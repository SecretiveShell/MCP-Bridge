import asyncio
from mcp import StdioServerParameters, stdio_client

from mcp_bridge.config import config
from mcp_bridge.mcp_clients.session import McpClientSession
from .AbstractClient import GenericMcpClient
from loguru import logger
import shutil
import os


# Keywords to identify virtual environment variables
venv_keywords = ["CONDA", "VIRTUAL", "PYTHON"]


class StdioClient(GenericMcpClient):
    config: StdioServerParameters

    def __init__(self, name: str, config: StdioServerParameters) -> None:
        super().__init__(name=name)

        # logger.debug(f"initializing settings for {name}: {config.command} {" ".join(config.args)}")

        own_config = config.model_copy(deep=True)

        env = dict(os.environ.copy())

        env = {
            key: value
            for key, value in env.items()
            if not any(key.startswith(keyword) for keyword in venv_keywords)
        }

        if config.env is not None:
            env.update(config.env)

        own_config.env = env

        command = shutil.which(config.command)
        if command is None:
            logger.error(f"could not find command {config.command}")
            exit(1)

        own_config.command = command

        # this changes the default to ignore
        if "encoding_error_handler" not in config.model_fields_set:
            own_config.encoding_error_handler = "ignore"

        self.config = own_config

    async def _maintain_session(self):
        logger.debug(f"starting maintain session for {self.name}")
        async with stdio_client(self.config) as client:
            logger.debug(f"entered stdio_client context manager for {self.name}")
            assert client[0] is not None, f"missing read stream for {self.name}"
            assert client[1] is not None, f"missing write stream for {self.name}"
            async with McpClientSession(*client) as session:
                logger.debug(f"entered client session context manager for {self.name}")
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
