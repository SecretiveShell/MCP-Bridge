import asyncio

from loguru import logger
from mcp import McpError, StdioServerParameters
from mcpx.client.transports.docker import DockerMCPServer

from mcp_bridge.config import config

from mcp_bridge.mcp_clients.sessionmaker import create_session, create_session_response

class MCPClientManager:
    clients: dict[str, create_session_response] = {}

    async def initialize(self):
        """Initialize the MCP Client Manager and start all clients"""

        logger.log("DEBUG", "Initializing MCP Client Manager")

        init_tasks = []

        for server_name, server_config in config.mcp_servers.items():
            logger.log("DEBUG", f"Initializing MCP client for server: {server_name}")

            instance = await create_session(server_name, server_config)
            init_task = asyncio.create_task(self.init_server_instance(instance))
            init_tasks.append(init_task)
        
        await asyncio.gather(*init_tasks)
        logger.log("DEBUG", "All MCP clients initialized")

    async def init_server_instance(self, session: create_session_response):
        completed = False
        for _ in range(3):
            await asyncio.sleep(1)
            try:
                async with asyncio.timeout(5):
                    await session.session.initialize()
                    completed = True
                    break
            except asyncio.TimeoutError:
                logger.warning(f"[{session.name}] Timeout during session init")
            except Exception as e:
                logger.exception(f"[{session.name}] Crash during session init: {e}")

        if not completed:
            logger.error(f"Failed to initialize session {session.name}")
            return
        
        self.clients[session.name] = session

    def get_client(self, server_name: str):
        return self.clients[server_name]

    def get_clients(self):
        return list(self.clients.items())

ClientManager = MCPClientManager()
