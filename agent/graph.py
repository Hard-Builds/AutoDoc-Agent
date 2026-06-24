from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent import node
from agent.state import AgentState
from integrations import GithubClient


class Graph:
    @classmethod
    async def init(cls, checkpointer):
        builder = StateGraph(AgentState)

        builder.add_node("fetch", node.fetch_pr_details)
        builder.add_node("analysis", node.analyzer)
        builder.add_node("comment", node.comment_analysis)
        builder.add_node("output", node.output)

        git_tool_node = ToolNode(await GithubClient.get_tools())
        builder.add_node("tools", git_tool_node)

        builder.add_edge(START, "fetch")
        builder.add_edge("fetch", "analysis")
        builder.add_conditional_edges(
            "analysis",
            lambda state: state["is_significant"],
            {True: "comment", False: END}
        )
        builder.add_edge("comment", "output")
        builder.add_conditional_edges("output", tools_condition)
        builder.add_edge("tools", "output")

        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_after=["comment"]
        )
        return graph
