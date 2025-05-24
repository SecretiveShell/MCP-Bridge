from fastapi import APIRouter, HTTPException
from mcp.types import ListPromptsResult, ListToolsResult, ListResourcesResult
from mcp_bridge.models.mcpServerStatus import McpServerStatus
from mcp_bridge.mcp_clients.McpClientManager import ClientManager

router = APIRouter(prefix="/servers")


@router.get("/{server_name}/prompts")
async def get_server_prompts(server_name: str) -> ListPromptsResult:
    """Get all prompts from a specific MCP server"""

    client = ClientManager.get_client(server_name)
    if not client:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    return await client.session.list_prompts()


@router.get("/{server_name}/tools")
async def get_server_tools(server_name: str) -> ListToolsResult:
    """Get all tools from a specific MCP server"""

    client = ClientManager.get_client(server_name)
    if not client:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    return await client.session.list_tools()


@router.get("/{server_name}/resources")
async def get_server_resources(server_name: str) -> ListResourcesResult:
    """Get all resources from a specific MCP server"""

    client = ClientManager.get_client(server_name)
    if not client:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    return await client.session.list_resources()


@router.get("/{server_name}/status")
async def get_server_status(server_name: str) -> McpServerStatus:
    """Get the status of a specific MCP server"""

    client = ClientManager.get_client(server_name)
    if not client:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    return McpServerStatus(
        name=server_name,
        online=client.session is not None,
        enabled=True, # TODO: Implement actual check for server enablement
    )
