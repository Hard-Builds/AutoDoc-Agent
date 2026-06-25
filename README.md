# AutoDoc Agent

An agentic GitHub PR reviewer built with **LangGraph**, **MCP (Model Context Protocol)**, and **Gemini** — designed as a learning project to explore **Human-in-the-Loop (HITL)** workflows and MCP-based tool integration.

## What it does

AutoDoc Agent monitors pull requests and automatically:

1. Fetches the PR diff via the GitHub MCP server
2. Analyses whether the changes are architecturally significant using Gemini
3. If significant — posts a `REQUEST_CHANGES` review on the PR asking for human approval
4. Waits for a human to comment `/approve-autodoc` (HITL step)
5. Reads `ARCHITECTURE.md` from the target branch, updates it to reflect the architectural changes, and writes it back
6. Approves and dismisses the original review once the doc is updated

## Learning goals

- **HITL with LangGraph** — using `interrupt_after` to pause the graph and resume on human input via the GitHub webhook trigger
- **MCP integration** — connecting to the GitHub MCP server to perform real GitHub operations (read PR diff, post reviews, read/write files) without custom API wrappers

## Agent graph

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([<p>__start__</p>]):::first
    fetch(fetch)
    analysis(analysis)
    pr_review(pr_review<hr/><small><em>__interrupt = after</em></small>)
    output(output)
    resolve_pr_review(resolve_pr_review)
    tools(tools)
    __end__([<p>__end__</p>]):::last
    __start__ --> fetch;
    analysis -. &nbsp;False&nbsp; .-> __end__;
    analysis -. &nbsp;True&nbsp; .-> pr_review;
    fetch --> analysis;
    output -.-> resolve_pr_review;
    output -.-> tools;
    pr_review --> output;
    tools --> output;
    resolve_pr_review --> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

## Sample output

The agent posts an analysis review on the PR, waits for `/approve-autodoc`, updates `ARCHITECTURE.md`, then approves and closes out the review:

![Sample PR review](docs/sample-pr-review.png)
