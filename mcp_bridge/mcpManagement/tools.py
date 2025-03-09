from typing import Any
from fastapi import APIRouter, HTTPException
from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from mcp.types import ListToolsResult, CallToolResult

router = APIRouter(prefix="/tools")


@router.get("")
async def get_tools() -> dict[str, ListToolsResult]:
    """Get all tools from all MCP clients"""

    tools = {}

    for name, client in ClientManager.get_clients():
        tools[name] = await client.list_tools()

    return tools


@router.post("/{tool_name}/call")
async def call_tool(tool_name: str, arguments: dict[str, Any] = {}) -> CallToolResult:
    """Call a tool"""

    client = await ClientManager.get_client_from_tool(tool_name)
    if not client:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return await client.call_tool(tool_name, arguments)
