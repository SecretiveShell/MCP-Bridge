from fastapi import APIRouter, HTTPException
from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from mcp.types import ListResourcesResult

router = APIRouter(prefix="/resources")


@router.get("")
async def get_resources() -> dict[str, ListResourcesResult]:
    """Get all resources from all MCP clients"""

    resources = {}

    for name, client in ClientManager.get_clients():
        resources[name] = await client.session.list_resources()

    return resources
