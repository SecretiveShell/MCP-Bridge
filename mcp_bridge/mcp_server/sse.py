import asyncio
from anyio import BrokenResourceError
from fastapi.responses import StreamingResponse
from .sse_transport import SseServerTransport
from fastapi import APIRouter, Request
from pydantic import ValidationError
from loguru import logger

from .server import server, options

router = APIRouter(prefix="/sse")

sse = SseServerTransport("/mcp-server/sse/messages")


@router.get("/", response_class=StreamingResponse)
async def handle_sse(request: Request):
    logger.info("New incoming SSE connection established from client.")
    async with sse.connect_sse(request) as streams:
        try:
            logger.info(f"SSE streams acquired. Handing off to MCP server run method. Streams: {streams}")
            await server.run(streams[0], streams[1], options)
            logger.info("MCP server run method completed for SSE connection.")
        except BrokenResourceError:
            logger.warning("SSE connection BrokenResourceError.")
            pass
        except asyncio.CancelledError:
            logger.warning("SSE connection CancelledError.")
            pass
        except ValidationError as ve:
            logger.error(f"SSE handler ValidationError: {ve}", exc_info=True)
            pass
        except Exception as e:
            logger.error(f"Unexpected error in SSE handler: {e}", exc_info=True)
            raise
    await request.close()
    logger.info("SSE connection closed and request finalized.")


@router.post("/messages")
async def handle_messages(request: Request):
    client_host = request.client.host if request.client else "Unknown_Host"
    client_port = request.client.port if request.client else "Unknown_Port"
    logger.info(f"Incoming POST /messages from {client_host}:{client_port}.")
    
    # Removed explicit body reading here. Let sse.handle_post_message consume the body.
    # try:
    #     body_bytes = await request.body()
    #     try:
    #         body_str = body_bytes.decode('utf-8')
    #         logger.debug(f"POST /messages request body (decoded): {body_str}")
    #     except UnicodeDecodeError:
    #         logger.warning(f"POST /messages request body (raw bytes, could not decode as UTF-8): {body_bytes}")
    # except Exception as e:
    #     logger.error(f"Error reading request body for POST /messages: {e}", exc_info=True)

    logger.info(f"Forwarding POST /messages from {client_host}:{client_port} to sse.handle_post_message.")
    await sse.handle_post_message(request.scope, request.receive, request._send)
    logger.info(f"sse.handle_post_message completed for POST /messages from {client_host}:{client_port}.")
