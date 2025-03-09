from typing import Any
from fastapi import APIRouter, HTTPException
from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from mcp.types import ListPromptsResult, GetPromptResult

router = APIRouter(prefix="/prompts")


@router.get("")
async def get_prompts() -> dict[str, ListPromptsResult]:
    """Get all prompts from all MCP clients"""

    prompts = {}

    for name, client in ClientManager.get_clients():
        prompts[name] = await client.list_prompts()

    return prompts


@router.post("/{prompt_name}")
async def get_prompt(prompt_name: str, args: dict[str, Any] = {}) -> GetPromptResult:
    """Evaluate a prompt"""

    client = await ClientManager.get_client_from_prompt(prompt_name)
    if not client:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")

    result = await client.get_prompt(prompt_name, arguments=args)
    if not result:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")

    return result
