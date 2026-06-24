import argparse
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.graph import Graph


async def analyze_mode(
        graph,
        owner: str,
        repo: str,
        pr_number: str,
        thread_id: str
):
    config = {"configurable": {"thread_id": thread_id}}
    final_state = await graph.ainvoke(
        input={
            "repo_owner": owner,
            "repo_name": repo,
            "pr_number": pr_number
        },
        config=config
    )
    return final_state


async def update_mode(graph, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    final_state = await graph.ainvoke(
        input=None,
        config=config
    )
    return final_state


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner")
    parser.add_argument("--repo")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--mode", choices=["analyze", "update"],
                        default="analyze")

    args = parser.parse_args()
    thread_id = f"{args.owner}-{args.repo}-{args.pr}"

    async with AsyncSqliteSaver.from_conn_string(
            "checkpoint.db"
    ) as checkpointer:
        graph = await Graph.init(checkpointer)
        if args.mode == "analyze":
            await analyze_mode(
                graph, args.owner, args.repo, args.pr, thread_id
            )
        else:
            await update_mode(graph, thread_id)

if __name__ == "__main__":
    asyncio.run(main())
