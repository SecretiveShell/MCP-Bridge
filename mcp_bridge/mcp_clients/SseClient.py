import asyncio
import httpx
from mcp.client.sse import sse_client
from mcp_bridge.config import config
from mcp_bridge.config.final import SSEMCPServer
from mcp_bridge.mcp_clients.session import McpClientSession
from .AbstractClient import GenericMcpClient
from loguru import logger


class SseClient(GenericMcpClient):
    config: SSEMCPServer

    def __init__(self, name: str, config: SSEMCPServer) -> None:
        super().__init__(name=name)

        self.config = config

    async def _maintain_session(self):
        ssl_verify = self.config.ssl_verify

        if ssl_verify is not True:
            # sse_client doesn't expose ssl options, so briefly patch httpx.AsyncClient
            # to inject the verify setting before it creates its internal client.
            _original_init = httpx.AsyncClient.__init__

            def _patched_init(client_self, *args, **kwargs):
                kwargs.setdefault("verify", ssl_verify)
                _original_init(client_self, *args, **kwargs)

            httpx.AsyncClient.__init__ = _patched_init

        try:
            async with sse_client(
                self.config.url, headers=self.config.headers or None
            ) as client:
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
        finally:
            if ssl_verify is not True:
                httpx.AsyncClient.__init__ = _original_init

        logger.debug(f"exiting session for {self.name}")
