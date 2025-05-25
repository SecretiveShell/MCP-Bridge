import asyncio
from dataclasses import dataclass
import time
from typing import Any
from loguru import logger
from mcp import McpError
from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from mcp.types import Cursor, Tool, CallToolResult
from mcp.client.session import ClientSession

__all__ = [
    "tools",
]

async def _list_tools(name: str, session: ClientSession) -> tuple[str, list[Tool]]:
    """List tools for a given session with pagination support."""
    tools = []
    done = False
    cursor: Cursor | None = None
    
    try:
        while not done:
            try:
                response = await session.list_tools(cursor=cursor)
                tools.extend(response.tools)
                if not response.nextCursor:
                    done = True
                else:
                    cursor = response.nextCursor
                if len(response.tools) == 0:
                    done = True
            except McpError as e:
                logger.error(f"Error listing tools for {name}: {e}")
                done = True
                break


    except Exception as e:
        logger.error(f"Error listing tools for {name}: {e}")

    return name, tools

@dataclass
class ToolsCacheObject:
    created: float
    tools: list[Tool]
    server_name: str

class ToolsRegistry:
    server_cache: dict[str, ToolsCacheObject] = {}

    async def prefill_tools(self) -> bool:
        """Prefill the tools cache"""
        tasks: list[asyncio.Task[tuple[str, list[Tool]]]] = []

        for name, session in ClientManager.get_clients():
            if session.session is None:
                continue

            if name in self.server_cache:
                continue

            task = asyncio.create_task(_list_tools(name, session.session))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for obj in results:
            if isinstance(obj, BaseException):
                logger.error(f"Error listing tools: {obj}")
                continue

            name, tools = obj
            self.server_cache[name] = ToolsCacheObject(
                created=time.time(),
                tools=tools,
                server_name=name,
            )

        return True
    

    async def get_tools(self, refresh: bool = False) -> list[Tool]:
        """list all tools from all clients"""
        if refresh:
            await self.prefill_tools()

        return [tool for obj in self.server_cache.values() for tool in obj.tools]
    

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] = {}) -> CallToolResult | None:
        """Call a tool"""
    
        toolname = None
        servername = None

        for server in self.server_cache.values():
            for tool in server.tools:
                if tool.name == tool_name:
                    toolname = tool
                    servername = server.server_name
                    break

        # if tool does not exist, return None
        if toolname is None:
            return None
        
        # if server does not exist, return None
        if servername is None:
            logger.error(f"Server for tool {tool_name} not found")
            return None

        client = ClientManager.get_client(servername)
        if client is None:
            logger.error(f"Client for server {servername} not found")
            return None
        
        try:
            return await client.session.call_tool(tool_name, arguments)
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return None

tools = ToolsRegistry()