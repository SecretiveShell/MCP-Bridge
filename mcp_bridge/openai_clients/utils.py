from typing import Optional
from loguru import logger
from lmos_openai_types import CreateChatCompletionRequest
import mcp.types
import json
import traceback

from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from mcp_bridge.tool_mappers import mcp2openai


def json_pretty_print(obj) -> str:
    if type(obj) == bytes:
        obj = obj.decode()
    if type(obj) == str:
        obj = json.loads(obj)
    ret = json.dumps(obj, indent=4, ensure_ascii=False)
    return ret

def validate_if_json_object_parsable(content: str):
    try:
        json.loads(content)
        return True
    except ValueError:
        return False


def salvage_parsable_json_object(content: str):
    content = content.strip()
    for i in range(0, len(content)):
        snippet = content[: len(content) - i]
        if validate_if_json_object_parsable(snippet):
            return snippet
            
async def chat_completion_add_tools(request: CreateChatCompletionRequest):
    request.tools = []
    logger.info("adding tools to request")

    for _, session in ClientManager.get_clients():
        # if session is None, then the client is not running
        if session.session is None:
            logger.error(f"session is `None` for {session.name}") # Date:2025/01/25 why not running?
            continue
        logger.debug(f"session ready for {session.name}")
        tools = await session.session.list_tools()
        for tool in tools.tools:
            request.tools.append(mcp2openai(tool))
    
    if request.tools == []:
        logger.info("this request loads no tools")
        # raise Exception("no tools found. unable to initiate chat completion.")
        request.tools = None
    return request


async def call_tool(
    tool_call_name: str, tool_call_json: str, timeout: Optional[int] = None
) -> Optional[mcp.types.CallToolResult]:
    if tool_call_name == "" or tool_call_name is None:
        logger.error("tool call name is empty")
        return None

    if tool_call_json is None:
        logger.error("tool call json is empty")
        return None

    session = await ClientManager.get_client_from_tool(tool_call_name)

    if session is None:
        logger.error(f"session is `None` for {tool_call_name}")
        return None

    try:
        tool_call_args = json.loads(tool_call_json) # Date: 2025/01/26 cannot load this tool call json?
    except json.JSONDecodeError:
        logger.error(f"failed to decode json for {tool_call_name}")
        traceback.print_exc()
        return None

    return await session.call_tool(tool_call_name, tool_call_args, timeout)
