import asyncio
import os
import weakref
from typing import Any, TypeVar, AsyncContextManager

from dataclasses import dataclass
from loguru import logger
from mcp import StdioServerParameters
from mcp.client.session import ClientSession, MessageHandlerFnT
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcpx.client.transports.docker import DockerMCPServer, docker_client

from mcp_bridge.config.final import ServerTypes, SSEMCPServer, StreamableHTTPServer
from mcp_bridge.sampling.sampler import sampling_callback

__all__ = [
    "create_session",
    "create_session_response",
]


@dataclass
class create_session_response:
    session: ClientSession
    transport: Any
    name: str
    id: str | None = None
    get_session_callback: MessageHandlerFnT | None = None

    _close_future: asyncio.Future | None = None
    _maintainer: asyncio.Task | None = None

    def __post_init__(self):
        weakref.finalize(self, lambda: logger.debug(f"SESSION GC'D: name={self.name}"))

    async def close(self):
        if self.session is not None:
            await self.session.__aexit__(None, None, None)

        if self._close_future is None:
            return
        
        if not self._close_future.done():
            self._close_future.set_result(None)

        if self._maintainer is not None and not self._maintainer.done():
            await self._maintainer


T = TypeVar("T")


async def transport_container(
    transport: AsyncContextManager[T],
    streams: asyncio.Future[T],
    future: asyncio.Future[None],
) -> None:
    async with transport as t:
        streams.set_result(t)
        await future


async def create_session(name: str, server: ServerTypes) -> create_session_response:
    transport = None
    streams = None
    sessionIDcallback: Any = None
    streams_future: asyncio.Future[tuple] = asyncio.Future[tuple]()
    end_future: asyncio.Future = asyncio.Future[None]()
    maintainer: asyncio.Task | None = None

    if isinstance(server, StdioServerParameters):
        env = os.environ.copy()
        env.update(server.env or {})
        env.update({"MCP_BRIDGE": "1"}) # allow servers to detect MCP Bridge
        server.env = env

        if not server.cwd:
            server.cwd = os.getcwd()

        transport = stdio_client(server)
        maintainer = asyncio.create_task(transport_container(transport, streams_future, end_future))
        streams = await streams_future

    elif isinstance(server, SSEMCPServer):
        transport = sse_client(
            url=server.url,
            headers=server.headers,
        )
        maintainer = asyncio.create_task(transport_container(transport, streams_future, end_future))
        streams = await streams_future

    elif isinstance(server, DockerMCPServer):
        raise NotImplementedError(
            "DockerMCPServer transport initialization not implemented"
        )

        # TODO: fix docker client to return correct streams
        transport = docker_client(server)
        streams = await transport.__aenter__()

    elif isinstance(server, StreamableHTTPServer):
        transport = streamablehttp_client(
            url=server.url,
            headers=server.headers,
            timeout=server.timeout,
            sse_read_timeout=server.sse_read_timeout,
            terminate_on_close=server.terminate_on_close,
        )
        maintainer = asyncio.create_task(transport_container(transport, streams_future, end_future))
        read_stream, write_stream, sessionIDcallback = await streams_future
        streams = (read_stream, write_stream)

    if streams is None:
        raise NotImplementedError(f"Server type {type(server)} not supported")

    client = ClientSession(
        read_stream=streams[0],
        write_stream=streams[1],
        read_timeout_seconds=None,
        sampling_callback=sampling_callback,
    )

    await client.__aenter__()

    weakref.finalize(
        maintainer, lambda: logger.debug(f"MAINTAINER GC'D: name={name}")
    )

    return create_session_response(
        name=name,
        session=client,
        transport=transport,
        get_session_callback=sessionIDcallback,
        _close_future=end_future,
        _maintainer=maintainer,
    )
