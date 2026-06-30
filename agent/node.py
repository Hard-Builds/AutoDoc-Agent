import os
from typing import List

import google.genai as genai
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field

from agent.state import AgentState, MAX_ITERATIONS
from integrations import GithubClient

# Leave ~10% headroom below the model's 1M-token context window.
_MAX_TOKENS = 900_000


def _count_and_guard(text: str, label: str) -> int:
    """Count tokens via Gemini's count_tokens API and raise if over the limit."""
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = client.models.count_tokens(model=model_name, contents=text)
    count = result.total_tokens
    print(f"[token-count] {label}: {count:,} tokens")
    if count > _MAX_TOKENS:
        raise ValueError(
            f"{label} exceeds context limit: {count:,} > {_MAX_TOKENS:,} tokens. "
            "Truncate the diff or split the PR before processing."
        )
    return count


async def fetch_pr_details(state: AgentState):
    """Fetch the raw unified diff for the PR via the GitHub MCP tool."""
    git_tools = await GithubClient.get_tools()

    pr_tool = next(filter(
        lambda x: x.name == "pull_request_read",
        git_tools
    ))
    result = await pr_tool.ainvoke({
        "method": "get_diff",
        "owner": state["repo_owner"],
        "repo": state["repo_name"],
        "pullNumber": state["pr_number"]
    })

    raw_diff = next(
        item["text"] for item in result if item.get("type") == "text"
    )
    return {"raw_diff": raw_diff}


async def analyzer(state: AgentState):
    """Use Gemini to decide if the PR diff represents an architecturally significant change."""
    class ArchitecturalAnalysis(BaseModel):
        is_significant: bool = Field(
            description="True if the PR changes architecture, infra or core patterns"
        )
        reasoning: str = Field(
            description="Brief technical justification for the assignment"
        )
        affected_components: List[str] = Field(
            description="List of service or modules impacted"
        )

    _count_and_guard(state["raw_diff"], "analyzer/raw_diff")

    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))
    llm = llm.with_structured_output(ArchitecturalAnalysis)
    response = await llm.ainvoke([
        SystemMessage(
            "You are an expert software architecture, your role is to "
            "analyse the PR's code diff and "
            "decide if the changes affect the project at the architecture level."
        ),
        HumanMessage(state["raw_diff"])
    ])

    return {
        "is_significant": response.is_significant,
        "reasoning": response.reasoning,
        "affected_components": response.affected_components
    }


async def add_pr_review(state: AgentState):
    """Post a REQUEST_CHANGES review on the PR asking a human to approve the doc update."""
    git_tools = await GithubClient.get_tools()
    pr_review_tool = next(filter(
        lambda x: x.name == "pull_request_review_write",
        git_tools
    ))

    components = ", ".join(state["affected_components"])
    body = (
        f"## AutoDoc Analysis\n\n"
        f"I've reviewed this PR and found it to be architecturally significant.\n\n"
        f"**Reasoning**: {state['reasoning']}\n\n"
        f"**Affected components**: {components}\n\n"
        f"---\nPlease Comment `/approve-autodoc` and I'll update `ARCHITECTURE.md` accordingly."
    )
    _ = await pr_review_tool.ainvoke({
        "method": "create",
        "owner": state["repo_owner"],
        "repo": state["repo_name"],
        "pullNumber": state["pr_number"],
        "body": body,
        "event": "REQUEST_CHANGES"
    })
    return {}


async def output(state: AgentState):
    """Invoke Gemini with GitHub tools to read, update, and write back ARCHITECTURE.md."""
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        print(
            "Tool used: ",
            (await StrOutputParser().ainvoke(state["messages"][-1]))[:200]
        )

    tools = await GithubClient.get_tools()

    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(
        "You are an expert software architect. "
        "When given a PR, read the ARCHITECTURE.md file from the PR's target branch, "
        "update it to reflect the architectural changes described, "
        "then write it back to that same branch. \n"
        "ARCHITECTURE.md file stays in the root directory itself, don't iterate every folder for finding it.\n"
        "If ARCHITECTURE.md does not exist, just say so — no need to create a new one. "
        "Always use the exact repo owner, repo name, and PR number provided by the user — never substitute your own."
    )

    if state["messages"]:
        messages = [system_msg] + list(state["messages"])
        new_messages = []
    else:
        human_msg = HumanMessage(
            f"Repo: {state['repo_owner']}/{state['repo_name']}, PR number: {state['pr_number']}\n\n"
            f"PR diff:\n{state['raw_diff']}\n\n"
            f"Analysis:\n{state['reasoning']}\n\n"
            f"Affected components:\n{', '.join(state['affected_components'])}\n\n"
            f"Read PR #{state['pr_number']} from {state['repo_owner']}/{state['repo_name']} to get the target branch. "
            f"Then read the ARCHITECTURE.md file from that target branch, "
            f"update it to reflect these architectural changes, "
            f"then write it back to the same branch."
        )
        messages = [system_msg, human_msg]
        new_messages = [human_msg]

    iteration = state.get("iteration", 0)
    if iteration >= MAX_ITERATIONS:
        raise RuntimeError(
            f"Output node exceeded {MAX_ITERATIONS} iterations — possible tool error loop.")

    combined_text = "\n".join(
        m.content if hasattr(m, "content") and isinstance(m.content, str) else str(m)
        for m in messages
    )
    _count_and_guard(combined_text, f"output/iteration-{iteration}")

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": new_messages + [response], "iteration": iteration + 1}


def output_router(state: AgentState):
    """Route to the tools node if Gemini called a tool, otherwise to resolve_pr_review."""
    response = tools_condition(state)
    if response == "tools":
        return "tools"
    return "resolve_pr_review"


async def resolve_pr_review(state: AgentState):
    """Approve the PR to signal that ARCHITECTURE.md has been updated."""
    git_tools = await GithubClient.get_tools()
    pr_review_tool = next(filter(
        lambda x: x.name == "pull_request_review_write",
        git_tools
    ))
    _ = await pr_review_tool.ainvoke({
        "method": "create",
        "owner": state["repo_owner"],
        "repo": state["repo_name"],
        "pullNumber": state["pr_number"],
        "body": "AutoDoc has updated ARCHITECTURE.md.",
        "event": "APPROVE"
    })
    return {}
