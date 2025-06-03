from aiodocker import Docker
from contextlib import asynccontextmanager
import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from loguru import logger
from pydantic import BaseModel, Field

from mcp import types

class DockerMCPServer(BaseModel):
    container_name: str | None = Field(default=None, description="Name of the docker container")
    image: str = Field(description="Image of the docker container")
    args: list[str] = Field(default_factory=list, description="Command line arguments for the docker container")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables for the docker container")


@asynccontextmanager
async def docker_client(server: DockerMCPServer):
    """
    Client transport for Docker: this will connect to a server by
    running a Docker container and communicating with it over its stdin/stdout.
    """
    read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

    write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
    write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    docker = Docker()

    try:
        # Pull the image if not available
        await docker.images.pull(server.image)

        # Create the container
        container = await docker.containers.create({
            "Image": server.image,
            "Args": server.args,
            "OpenStdin": True,
            "AttachStdout": True,
            "AttachStderr": True,
            "Tty": False,
            "HostConfig": {"AutoRemove": True},
        })

        await container.start()
        logger.debug(f"Started Docker container {container.id}")

        # Attach to the container's input/output streams
        attach_result = container.attach(stdout=True, stdin=True)

        async def read_from_stdout():
            try:
                async with read_stream_writer:
                    buffer = ""
                    while True: 
                        msg = await attach_result.read_out()
                        if msg is None:
                            continue
                        chunk = msg.data
                        if isinstance(chunk, bytes):
                            chunk = chunk.decode("utf-8")
                            lines = (buffer + chunk).split("\n")
                            buffer = lines.pop()

                            for line in lines:
                                try:
                                    json_message = types.JSONRPCMessage.model_validate_json(line)
                                    await read_stream_writer.send(json_message)
                                except Exception as exc:
                                    await read_stream_writer.send(exc)
            except anyio.ClosedResourceError:
                await anyio.lowlevel.checkpoint()

        async def write_to_stdin():
            try:
                async with write_stream_reader:
                    async for message in write_stream_reader:
                        json = message.model_dump_json(by_alias=True, exclude_none=True)
                        await attach_result.write_in(json.encode("utf-8") + b"\n")
            except anyio.ClosedResourceError:
                await anyio.lowlevel.checkpoint()

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(read_from_stdout)
                tg.start_soon(write_to_stdin)
                yield read_stream, write_stream
        finally:
            await container.stop()
            await container.delete()

    except Exception as e:
        logger.error(f"Error in docker client: {e}")
    finally:
        await docker.close()
        logger.debug("Docker client closed.")