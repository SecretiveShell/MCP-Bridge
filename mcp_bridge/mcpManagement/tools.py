from typing import Any

from fastapi import APIRouter, HTTPException
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel

from mcp_bridge.mcp_clients.tools import tools

router = APIRouter(prefix="/tools")


class ListTools(BaseModel):
    tools: list[Tool]


@router.get("")
async def get_tools() -> ListTools:
    """Get all tools from all MCP clients"""

    list_of_tools = await tools.get_tools()
    return ListTools(tools=list_of_tools)


@router.post("/{tool_name}/call")
async def call_tool(tool_name: str, arguments: dict[str, Any] = {}) -> CallToolResult:
    """Call a tool"""

    result = await tools.call_tool(tool_name, arguments)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    return result
