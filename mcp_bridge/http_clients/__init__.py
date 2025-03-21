from httpx import AsyncClient
from mcp_bridge.config import config


# change this if you want to hard fork the repo
# its used to show ranking on openrouter and other inference providers
BRIDGE_REPO_URL = "https://github.com/SecretiveShell/MCP-Bridge"
BRIDGE_APP_TITLE = "MCP Bridge"


def get_client() -> AsyncClient:
    client: AsyncClient = AsyncClient(
        base_url=config.inference_server.base_url,
        headers={"Content-Type": "application/json"},
        timeout=10000,
    )

    # generic openai
    if config.inference_server.type == "openai":
        client.headers["Authorization"] = rf"Bearer {config.inference_server.api_key}"
        return client

    # openrouter
    if config.inference_server.type == "openrouter":
        client.headers["Authorization"] = rf"Bearer {config.inference_server.api_key}"
        client.headers["HTTP-Referer"] = BRIDGE_REPO_URL
        client.headers["X-Title"] = BRIDGE_APP_TITLE
        return client

    # gemini models
    if config.inference_server.type == "gemini":
        client.headers["Authorization"] = rf"Bearer {config.inference_server.api_key}"
        return client

    raise NotImplementedError("Inference Server Type not supported")
