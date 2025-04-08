from mcp import Tool
from typing import Dict, Any
from loguru import logger


def mcp2anthropic(mcp_tool: Tool) -> Dict[str, Any]:
    """Convert a MCP Tool to an Anthropic tool format."""
    
    logger.debug(f"Converting MCP tool: {mcp_tool.name}")
    logger.debug(f"Tool schema: {mcp_tool.inputSchema}")
    
    return {
        "name": mcp_tool.name,
        "description": mcp_tool.description,
        "input_schema": mcp_tool.inputSchema,
    } 