from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from agent import node
from agent.state import AgentState
from integrations import GithubClient


class Graph:
    @classmethod
    async def init(cls, checkpointer):
        builder = StateGraph(AgentState)

        builder.add_node("fetch", node.fetch_pr_details)
        builder.add_node("analysis", node.analyzer)
        builder.add_node("pr_review", node.add_pr_review)
        builder.add_node("output", node.output)
        builder.add_node("resolve_pr_review", node.resolve_pr_review)

        git_tool_node = ToolNode(await GithubClient.get_tools())
        builder.add_node("tools", git_tool_node)

        builder.add_edge(START, "fetch")
        builder.add_edge("fetch", "analysis")
        builder.add_conditional_edges(
            "analysis",
            lambda state: state["is_significant"],
            {True: "pr_review", False: END}
        )
        builder.add_edge("pr_review", "output")
        builder.add_conditional_edges(
            "output",
            node.output_router,
            {"tools": "tools", "resolve_pr_review": "resolve_pr_review"}
        )
        builder.add_edge("tools", "output")
        builder.add_edge("resolve_pr_review", END)

        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_after=["pr_review"]
        )
        return graph
