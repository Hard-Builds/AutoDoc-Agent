import os
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field

from agent.state import AgentState, MAX_ITERATIONS
from integrations import GithubClient


async def fetch_pr_details(state: AgentState):
    # Add github mcp call
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
    # takes the raw_diff and check if this is a arch level diff
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

    response = await llm_with_tools.ainvoke(messages)
    return {"messages": new_messages + [response], "iteration": iteration + 1}


def output_router(state: AgentState):
    response = tools_condition(state)
    if response == "tools":
        return "tools"
    return "resolve_pr_review"


async def resolve_pr_review(state: AgentState):
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
