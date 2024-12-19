import json
from typing import Optional
from lmos_openai_types import (
    ChatCompletionRequestMessage,
    CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse,
)
from models import SSEData
from .genericHttpxClient import client
from mcp_clients.McpClientManager import ClientManager
from tool_mappers import mcp2openai
from loguru import logger
from httpx_sse import aconnect_sse

from sse_starlette.sse import EventSourceResponse, ServerSentEvent


async def streaming_chat_completions(request: CreateChatCompletionRequest):
    # raise NotImplementedError("Streaming Chat Completion is not supported")

    try:
        return EventSourceResponse(
            content=chat_completions(request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    except Exception as e:
        logger.error(e)


async def chat_completions(request: CreateChatCompletionRequest):
    """performs a chat completion using the inference server"""

    request.stream = True

    request.tools = []

    for _, session in ClientManager.get_clients():
        tools = await session.session.list_tools()
        for tool in tools.tools:
            request.tools.append(mcp2openai(tool))

    fully_done = False
    while not fully_done:
        json_data = request.model_dump_json(
            exclude_defaults=True, exclude_none=True, exclude_unset=True
        )

        logger.debug(json_data)

        last: Optional[CreateChatCompletionStreamResponse] = None  # last message

        async with aconnect_sse(
            client, "post", "/chat/completions", content=json_data
        ) as event_source:
            async for sse in event_source.aiter_sse():
                event = sse.event
                data = sse.data
                id = sse.id
                retry = sse.retry

                logger.debug(
                    f"event: {event},\ndata: {data},\nid: {id},\nretry: {retry}"
                )

                # handle if the SSE stream is done
                if data == "[DONE]":
                    break

                parsed_data = CreateChatCompletionStreamResponse.model_validate_json(
                    data
                )

                yield SSEData.model_validate_json(sse.data).model_dump_json()

                last = parsed_data

        # ideally we should check this properly
        assert last is not None
        assert last.choices[0].finish_reason is not None

        if last.choices[0].finish_reason.value in ["stop", "length"]:
            logger.debug("no tool calls found")
            fully_done = True

    # when done, send the final event
    logger.debug("sending final event")
    yield ServerSentEvent(event="message", data="[DONE]", id=None, retry=None)
