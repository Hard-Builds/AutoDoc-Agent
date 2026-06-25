from abc import ABC

from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPClientABC(ABC):
    """Abstract base for MCP-backed tool clients. Subclasses declare _SERVERS and _NEEDED_TOOLS."""

    _SERVERS = None
    _NEEDED_TOOLS = None

    _client = None
    _tools = None

    @classmethod
    async def get_client(cls):
        """Return the shared MCP client, initialising it on first call."""
        if cls._client is None:
            cls._client = MultiServerMCPClient(cls._SERVERS)
        return cls._client

    @classmethod
    async def get_tools(cls):
        """Return the filtered list of LangChain tools exposed by the MCP server."""
        client = await cls.get_client()
        if cls._tools is None:
            tools = await client.get_tools()
            tools = [t for t in tools if t.name in cls._NEEDED_TOOLS]
            cls._tools = tools
        return cls._tools
