import os
import time

import httpx
import jwt
from langchain_mcp_adapters.client import MultiServerMCPClient

from integrations.base import MCPClientABC


class GithubClient(MCPClientABC):
    _SERVERS = {
        "github": {
            "transport": "streamable_http",
            "url": "https://api.githubcopilot.com/mcp/",
        }
    }

    _NEEDED_TOOLS = (
        "get_file_contents",
        "create_or_update_file",
        "pull_request_read",
        "pull_request_review_write"
    )

    @classmethod
    async def get_client(cls):
        if cls._client is None:
            token = await cls.get_installation_token()
            cls._SERVERS["github"].update({
                "headers": {
                    "Authorization": f"Bearer {token}"
                }
            })
            cls._client = MultiServerMCPClient(cls._SERVERS)
        return cls._client

    @staticmethod
    async def get_installation_token() -> str:
        app_id = os.getenv("GH_APP_ID")
        installation_id = os.getenv("GH_APP_INSTALLATION_ID")
        private_key = os.getenv("GH_APP_PRIVATE_KEY", "").replace("\\n", "\n")

        now = int(time.time())
        payload = {
            "iat": now - 60,
            "exp": now + 600,
            "iss": app_id
        }
        jwt_token = jwt.encode(payload, private_key, algorithm="RS256")

        # Exchange JWT for installation access token
        url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers)
            resp.raise_for_status()
            return resp.json()["token"]
