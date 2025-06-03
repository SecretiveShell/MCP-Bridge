import asyncio

from mcp_bridge.mcp_clients.session import McpClientSession
from mcp_bridge.config import config
from mcp_bridge.config.docker import docker_client, DockerMCPServer
from .AbstractClient import GenericMcpClient
from loguru import logger


class DockerClient(GenericMcpClient):
    config: DockerMCPServer

    def __init__(self, name: str, config: DockerMCPServer) -> None:
        super().__init__(name=name)

        self.config = config

    async def _maintain_session(self):
        async with docker_client(self.config) as client:
            logger.debug(f"made instance of docker client for {self.name}")
            async with McpClientSession(*client) as session:
                await session.initialize()
                logger.debug(f"finished initialise session for {self.name}")
                self.session = session

                try:
                    while True:
                        await asyncio.sleep(10)
                        if config.logging.log_server_pings:
                            logger.debug(f"pinging session for {self.name}")

                        await session.send_ping()

                except Exception as exc:
                    logger.error(f"ping failed for {self.name}: {exc}")
                    self.session = None

        logger.debug(f"exiting session for {self.name}")
