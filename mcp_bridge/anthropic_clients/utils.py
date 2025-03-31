from typing import Dict, Any, List, Optional
from loguru import logger
import mcp.types
import json
import asyncio

from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from mcp_bridge.tool_mappers import mcp2anthropic


def format_tool_response_for_anthropic(tool_id: str, result_text: str) -> List[Dict[str, Any]]:
    """Format tool response in the correct structure for Anthropic API"""
    return [
        {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": result_text
        }
    ]


async def anthropic_get_tools() -> List[Dict[str, Any]]:
    """
    Get MCP tools and convert them to Anthropic format.
    Returns a list of tools in Anthropic format.
    """
    tools = []
    clients = ClientManager.get_clients()
    logger.info(f"Found {len(clients)} MCP clients")

    # Wait a bit for clients to initialize if needed
    for client_name, session in clients:
        if session.session is None:
            logger.info(f"Waiting for client {client_name} to initialize...")
            try:
                await asyncio.wait_for(session._wait_for_session(timeout=10, http_error=False), 10)
                logger.info(f"Client {client_name} initialized successfully")
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"Timed out waiting for client {client_name} to initialize: {e}")

    # Get updated clients after waiting
    clients = ClientManager.get_clients()
    for client_name, session in clients:
        logger.info(f"Processing client: {client_name}")
        # if session is None, then the client is not running
        if session.session is None:
            logger.error(f"Session is `None` for {client_name}")
            continue

        try:
            logger.info(f"Calling list_tools() for {client_name}")
            mcp_tools = await session.session.list_tools()
            logger.info(f"Client {client_name} returned {len(mcp_tools.tools)} tools")
            
            for tool in mcp_tools.tools:
                logger.info(f"Adding tool: {tool.name}")
                tools.append(mcp2anthropic(tool))
            logger.debug(f"Added {len(mcp_tools.tools)} tools from {client_name}")
        except Exception as e:
            logger.error(f"Error getting tools from {client_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info(f"Total Anthropic tools available: {len(tools)}")
    if tools:
        logger.debug(f"Tool names: {[t['name'] for t in tools]}")
    else:
        logger.warning("No tools were found from any client!")
    
    return tools


async def call_tool(
    tool_name: str, tool_input: Dict[str, Any], timeout: Optional[int] = None
) -> Optional[mcp.types.CallToolResult]:
    """
    Call a tool with the given name and input.
    Returns the tool call result or None if the call failed.
    """
    if not tool_name:
        logger.error("Tool name is empty")
        return None

    session = await ClientManager.get_client_from_tool(tool_name)

    if session is None:
        logger.error(f"Session is `None` for {tool_name}")
        return None

    return await session.call_tool(tool_name, tool_input, timeout) 