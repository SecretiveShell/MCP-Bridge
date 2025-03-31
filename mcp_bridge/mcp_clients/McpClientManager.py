from typing import Union

from loguru import logger
from mcp import McpError, StdioServerParameters
from mcpx.client.transports.docker import DockerMCPServer

from mcp_bridge.config import config
from mcp_bridge.config.final import SSEMCPServer

from .DockerClient import DockerClient
from .SseClient import SseClient
from .StdioClient import StdioClient

client_types = Union[StdioClient, SseClient, DockerClient]


class MCPClientManager:
    clients: dict[str, client_types] = {}
    _client_tasks = {}  # Store tasks for cleanup

    async def initialize(self):
        """Initialize the MCP Client Manager and start all clients"""

        logger.log("DEBUG", "Initializing MCP Client Manager")

        for server_name, server_config in config.mcp_servers.items():
            self.clients[server_name] = await self.construct_client(
                server_name, server_config
            )

    async def shutdown(self):
        """Shutdown all MCP clients and cancel their tasks"""
        logger.log("DEBUG", "Shutting down all MCP clients")
        
        # Cancel all session maintainer tasks
        for name, client in list(self.clients.items()):
            if hasattr(client, "_task") and client._task is not None:
                logger.log("DEBUG", f"Cancelling task for {name}")
                client._task.cancel()
                try:
                    await client._task
                except:
                    pass  # Task was cancelled, ignore the exception
                
            # Clear client references
            self.clients[name] = None
        
        # Clear the clients dictionary
        self.clients.clear()
        logger.log("DEBUG", "All MCP clients shut down")

    async def construct_client(self, name, server_config) -> client_types:
        logger.log("DEBUG", f"Constructing client for {server_config}")

        if isinstance(server_config, StdioServerParameters):
            client = StdioClient(name, server_config)
            await client.start()
            return client

        if isinstance(server_config, SSEMCPServer):
            # TODO: implement sse client
            client = SseClient(name, server_config)  # type: ignore
            await client.start()
            return client
        
        if isinstance(server_config, DockerMCPServer):
            client = DockerClient(name, server_config)
            await client.start()
            return client

        raise NotImplementedError("Client Type not supported")

    def get_client(self, server_name: str):
        return self.clients[server_name]

    def get_clients(self):
        logger.debug(f"Getting clients, total: {len(self.clients)}")
        for name, client in self.clients.items():
            session_status = "initialized" if client.session else "not initialized"
            logger.debug(f"Client {name}: session {session_status}")
        return list(self.clients.items())

    async def get_client_from_tool(self, tool: str):
        logger.debug(f"Looking for client with tool: {tool}")
        for name, client in self.get_clients():
            
            # client cannot have tools if it is not connected
            if not client.session:
                logger.debug(f"Client {name} session is None, skipping")
                continue

            try:
                logger.debug(f"Calling list_tools on client {name}")
                list_tools = await client.session.list_tools()
                logger.debug(f"Client {name} has {len(list_tools.tools)} tools")
                for client_tool in list_tools.tools:
                    logger.debug(f"Client {name} has tool: {client_tool.name}")
                    if client_tool.name == tool:
                        logger.debug(f"Found tool {tool} in client {name}")
                        return client
            except McpError as e:
                logger.error(f"Error listing tools for client {name}: {e}")
                continue
        logger.warning(f"No client found with tool: {tool}")

    async def get_client_from_prompt(self, prompt: str):
        for name, client in self.get_clients():

            # client cannot have prompts if it is not connected
            if not client.session:
                continue

            try:
                list_prompts = await client.session.list_prompts()
                for client_prompt in list_prompts.prompts:
                    if client_prompt.name == prompt:
                        return client
            except McpError:
                continue


ClientManager = MCPClientManager()
