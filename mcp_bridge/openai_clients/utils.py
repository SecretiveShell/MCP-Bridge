import json

import mcp.types
from lmos_openai_types import CreateChatCompletionRequest
from loguru import logger

from mcp_bridge.mcp_clients.tools import tools
from mcp_bridge.tool_mappers import mcp2openai


async def chat_completion_add_tools(request: CreateChatCompletionRequest):
    request.tools = []

    for tool in await tools.get_tools():
        request.tools.append(mcp2openai(tool))

    return request


async def call_tool(
    tool_call_name: str, tool_call_json: str
) -> mcp.types.CallToolResult | None:
    if tool_call_name == "" or tool_call_name is None:
        logger.error("tool call name is empty")
        return None

    if tool_call_json is None:
        logger.error("tool call json is empty")
        return None

    try:
        tool_call_args = json.loads(tool_call_json)
    except json.JSONDecodeError:
        logger.error(f"failed to decode json for {tool_call_name}")
        return None

    return await tools.call_tool(tool_call_name, tool_call_args)
