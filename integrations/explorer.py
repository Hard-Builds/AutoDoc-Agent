from pathlib import Path

from integrations.base import MCPClientABC

dir_path = Path(__file__).parent.parent / "docs"
dir_path.mkdir(parents=True, exist_ok=True)


class FileExplorer(MCPClientABC):
    """MCP client for the local filesystem server, scoped to the docs directory."""

    _SERVERS = {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(dir_path)  # allow list
            ],
            "transport": "stdio"
        }
    }
