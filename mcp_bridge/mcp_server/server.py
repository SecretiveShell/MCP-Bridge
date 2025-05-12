from mcp import types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl
from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from loguru import logger
import asyncio

__all__ = ["server", "options"]

server = Server("MCP-Bridge")

## list functions


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    prompts = []
    logger.info("Aggregating prompts from all managed MCP clients concurrently.")
    
    active_clients = [(name, client) for name, client in ClientManager.get_clients() if client and client.session]
    if not active_clients:
        logger.info("No active clients to fetch prompts from.")
        return []

    coroutines = []
    for name, client in active_clients:
        logger.debug(f"Preparing to list prompts from client session: {name}")
        coroutines.append(client.session.list_prompts())

    logger.info(f"Gathering prompts from {len(coroutines)} active client sessions.")
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    for i, result in enumerate(results):
        name, client = active_clients[i] # Get corresponding client name
        if isinstance(result, Exception):
            logger.error(f"Error listing prompts for client '{name}': {result}", exc_info=result)
        elif result and result.prompts:
            logger.debug(f"Client '{name}' provided prompts: {[p.name for p in result.prompts]}")
            prompts.extend(result.prompts)
        else:
            logger.debug(f"Client '{name}' provided no prompts or an empty response.")
            
    logger.info(f"Finished prompt aggregation. Total prompts aggregated: {len(prompts)}. Names: {[p.name for p in prompts]}")
    return prompts


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    resources = []
    logger.info("Aggregating resources from all managed MCP clients concurrently.")

    active_clients = [(name, client) for name, client in ClientManager.get_clients() if client and client.session]
    if not active_clients:
        logger.info("No active clients to fetch resources from.")
        return []

    coroutines = []
    for name, client in active_clients:
        logger.debug(f"Preparing to list resources from client session: {name}")
        coroutines.append(client.session.list_resources())

    logger.info(f"Gathering resources from {len(coroutines)} active client sessions.")
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    for i, result in enumerate(results):
        name, client = active_clients[i] # Get corresponding client name
        if isinstance(result, Exception):
            logger.error(f"Error listing resources for client '{name}': {result}", exc_info=result)
        elif result and result.resources:
            logger.debug(f"Client '{name}' provided {len(result.resources)} resources.")
            resources.extend(result.resources)
        else:
            logger.debug(f"Client '{name}' provided no resources or an empty response.")

    logger.info(f"Finished resource aggregation. Total resources aggregated: {len(resources)}.")
    return resources


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    return []


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    tools = []
    logger.info("Aggregating tools from all managed MCP clients concurrently.")

    active_clients = [(name, client) for name, client in ClientManager.get_clients() if client and client.session]
    if not active_clients:
        logger.info("No active clients to fetch tools from.")
        return []

    coroutines = []
    for name, client in active_clients:
        logger.debug(f"Preparing to list tools from client session: {name}")
        coroutines.append(client.session.list_tools())

    logger.info(f"Gathering tools from {len(coroutines)} active client sessions.")
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    for i, result in enumerate(results):
        name, client = active_clients[i] # Get corresponding client name
        if isinstance(result, Exception):
            logger.error(f"Error listing tools for client '{name}': {result}", exc_info=result)
        elif result and result.tools:
            logger.debug(f"Client '{name}' provided tools: {[t.name for t in result.tools]}")
            tools.extend(result.tools)
        else:
            logger.debug(f"Client '{name}' provided no tools or an empty response.")

    logger.info(f"Finished tool aggregation. Total tools aggregated: {len(tools)}. Names: {[t.name for t in tools]}")
    return tools


## get functions


@server.get_prompt()
async def get_prompt(name: str, args: dict[str, str] | None) -> types.GetPromptResult:
    client = await ClientManager.get_client_from_prompt(name)

    # if client is None, then we cannot get the prompt
    if client is None:
        raise Exception(f"Prompt '{name}' not found")

    # if args is None, then we should use an empty dict
    if args is None:
        args = {}
    
    logger.info(f"Getting prompt '{name}' with args '{args}' from client: {client.name if client else 'Unknown'}")
    result = await client.get_prompt(name, args)
    if result is None:
        logger.error(f"Prompt '{name}' not found by client {client.name if client else 'Unknown'}")
        raise Exception(f"Prompt '{name}' not found")
    logger.info(f"Successfully retrieved prompt '{name}'.")
    return result


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str | bytes:
    for name, client in ClientManager.get_clients():
        try:
            client_resources = await client.list_resources()
            if str(uri) in map(lambda x: str(x.uri), client_resources.resources):
                response = await client.read_resource(uri)
                for resource in response:
                    if resource.mimeType == "text/plain":
                        assert isinstance(resource, types.TextResourceContents)
                        assert type(resource.text) is str
                        return resource.text

                    elif resource.mimeType == "application/octet-stream":
                        assert isinstance(resource, types.BlobResourceContents)
                        assert type(resource.blob) is bytes
                        return resource.blob

                    else:
                        raise Exception(
                            f"Unsupported resource type: {resource.mimeType}"
                        )

        except Exception as e:
            logger.error(f"Error listing resources for {name}: {e}")

    raise Exception(f"Resource '{uri}' not found")


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.info(f"Attempting to call tool '{name}' with arguments: {arguments}")
    client = await ClientManager.get_client_from_tool(name)

    if client is None:
        logger.error(f"Tool '{name}' could not be mapped to any managed client.")
        raise Exception(f"Tool '{name}' not found")

    if arguments is None:
        arguments = {}
    
    logger.info(f"Calling tool '{name}' on resolved client '{client.name}' with arguments: {arguments}")
    try:
        tool_result = await client.call_tool(name, arguments)
        logger.info(f"Tool '{name}' executed successfully by client '{client.name}'. Result content items: {len(tool_result.content) if tool_result else 'None'}")
        return tool_result.content
    except Exception as e:
        logger.error(f"Error calling tool '{name}' on client '{client.name}': {e}", exc_info=True)
        raise


# options

options = InitializationOptions(
    server_name="MCP-Bridge",
    server_version="0.2.0",
    capabilities=server.get_capabilities(
        notification_options=NotificationOptions(),
        experimental_capabilities={},
    ),
)
