from typing import Union
import asyncio

from loguru import logger
from mcp import McpError, StdioServerParameters
from mcp_bridge.config.docker import DockerMCPServer

from mcp_bridge.config import config
from mcp_bridge.config.final import SSEMCPServer

from .DockerClient import DockerClient
from .SseClient import SseClient
from .StdioClient import StdioClient

client_types = Union[StdioClient, SseClient, DockerClient]


class MCPClientManager:
    clients: dict[str, client_types] = {}

    async def initialize(self):
        """Initialize the MCP Client Manager and start all clients concurrently."""
        logger.info("Initializing MCP Client Manager")

        construction_coroutines = []
        server_names_ordered = [] # To map results back to names

        for server_name, server_config in config.mcp_servers.items():
            logger.info(f"Preparing to initialize MCP client: {server_name} with config: {server_config}")
            construction_coroutines.append(self.construct_and_start_client(server_name, server_config))
            server_names_ordered.append(server_name)

        if not construction_coroutines:
            logger.info("No MCP clients configured to initialize.")
            return

        logger.info(f"Attempting to concurrently initialize {len(construction_coroutines)} MCP clients...")
        # The construct_and_start_client will return (name, client_instance) or (name, None) on failure
        results = await asyncio.gather(*construction_coroutines, return_exceptions=True)

        for i, result in enumerate(results):
            server_name = server_names_ordered[i]
            if isinstance(result, Exception):
                logger.error(f"Exception during initialization of MCP client: {server_name}. Error: {result}", exc_info=result)
            elif result is None: # Should not happen if construct_and_start_client returns (name, None)
                 logger.error(f"Failed to initialize or start MCP client: {server_name}. No client instance returned.")
            elif isinstance(result, tuple) and len(result) == 2:
                client_instance = result[1]
                if client_instance:
                    self.clients[server_name] = client_instance
                    logger.info(f"Successfully initialized and started MCP client: {server_name}")
                else: # Explicitly handled None for client_instance from construct_and_start_client
                    logger.error(f"Failed to initialize or start MCP client: {server_name}. Construction returned None.")
            else:
                logger.error(f"Unexpected result type during initialization of MCP client: {server_name}. Result: {result}")

    async def construct_and_start_client(self, name, server_config) -> tuple[str, client_types | None]:
        """Constructs and starts a single client, returns (name, client_instance | None)."""
        logger.info(f"Constructing client '{name}' for type: {type(server_config)}")
        try:
            client_instance: client_types | None = None
            if isinstance(server_config, StdioServerParameters):
                command_to_log = server_config.command if hasattr(server_config, 'command') else "Not specified"
                args_to_log = server_config.args if hasattr(server_config, 'args') else []
                logger.info(f"Creating StdioClient for '{name}' with command: '{command_to_log}' and args: '{args_to_log}'")
                client_instance = StdioClient(name, server_config)
                await client_instance.start()
                logger.info(f"StdioClient '{name}' started.")
            elif isinstance(server_config, SSEMCPServer):
                logger.info(f"Creating SseClient for '{name}' with URL: '{server_config.url}'")
                client_instance = SseClient(name, server_config)  # type: ignore
                await client_instance.start()
                logger.info(f"SseClient '{name}' started.")
            elif isinstance(server_config, DockerMCPServer):
                logger.info(f"Creating DockerClient for '{name}' with image: '{server_config.image_name}'")
                client_instance = DockerClient(name, server_config)
                await client_instance.start()
                logger.info(f"DockerClient '{name}' started.")
            else:
                logger.error(f"Unsupported MCP server config type for client '{name}': {type(server_config)}")
                # raise NotImplementedError("Client Type not supported") # Don't raise, return None
                return name, None
            return name, client_instance
        except Exception as e:
            logger.error(f"Error constructing or starting client {name}: {e}", exc_info=True)
            return name, None # Return name and None on failure

    def get_client(self, server_name: str):
        return self.clients[server_name]

    def get_clients(self):
        return list(self.clients.items())

    async def get_client_from_tool(self, tool: str):
        logger.info(f"Attempting to find client for tool: '{tool}'")
        for name, client in self.get_clients():
            logger.debug(f"Checking client '{name}' for tool '{tool}'")
            # client cannot have tools if it is not connected
            if not client.session:
                logger.warning(f"Client '{name}' session not active, skipping for tool '{tool}'")
                continue

            try:
                logger.debug(f"Calling list_tools() on client '{name}' for tool '{tool}'")
                list_tools_response = await client.session.list_tools()
                logger.debug(f"Client '{name}' returned tools: {list_tools_response.tools}")
                for client_tool in list_tools_response.tools:
                    if client_tool.name == tool:
                        logger.info(f"Found tool '{tool}' in client '{name}'")
                        return client
            except McpError as e:
                logger.error(f"McpError while listing tools for client '{name}': {e}", exc_info=True)
                continue
            except Exception as e:
                logger.error(f"Unexpected error while listing tools for client '{name}': {e}", exc_info=True)
                continue
        logger.warning(f"Tool '{tool}' not found in any active client.")
        return None # Explicitly return None if not found

    async def get_client_from_prompt(self, prompt: str):
        logger.info(f"Attempting to find client for prompt: '{prompt}'")
        for name, client in self.get_clients():
            logger.debug(f"Checking client '{name}' for prompt '{prompt}'")

            # client cannot have prompts if it is not connected
            if not client.session:
                logger.warning(f"Client '{name}' session not active, skipping for prompt '{prompt}'")
                continue

            try:
                logger.debug(f"Calling list_prompts() on client '{name}' for prompt '{prompt}'")
                list_prompts_response = await client.session.list_prompts()
                logger.debug(f"Client '{name}' returned prompts: {list_prompts_response.prompts}")
                for client_prompt in list_prompts_response.prompts:
                    if client_prompt.name == prompt:
                        logger.info(f"Found prompt '{prompt}' in client '{name}'")
                        return client
            except McpError as e:
                logger.error(f"McpError while listing prompts for client '{name}': {e}", exc_info=True)
                continue
            except Exception as e:
                logger.error(f"Unexpected error while listing prompts for client '{name}': {e}", exc_info=True)
                continue
        logger.warning(f"Prompt '{prompt}' not found in any active client.")
        return None # Explicitly return None if not found


ClientManager = MCPClientManager()
