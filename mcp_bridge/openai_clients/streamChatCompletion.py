import datetime
import json
import os
import secrets
import time
import traceback
from typing import Optional
from secrets import token_hex
from lmos_openai_types import (
    ChatCompletionMessageToolCall,
    ChatCompletionRequestMessage,
    CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse,
    Function1,
    FinishReason1,
)

from mcp_bridge.inference_engine_mappers.chat.requester import chat_completion_requester
from mcp_bridge.inference_engine_mappers.chat.stream_responder import (
    chat_completion_stream_responder,
)
from .utils import (
    call_tool,
    chat_completion_add_tools,
    json_pretty_print,
    salvage_parsable_json_object,
    validate_if_json_object_parsable,
)
from mcp_bridge.models import SSEData, upstream_error
from mcp_bridge.http_clients import get_client
from loguru import logger
from httpx_sse import aconnect_sse

from sse_starlette.sse import EventSourceResponse
from sse_starlette.event import ServerSentEvent


async def streaming_chat_completions(request: CreateChatCompletionRequest):
    # raise NotImplementedError("Streaming Chat Completion is not supported")

    return EventSourceResponse(
        content=chat_completions(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def format_error_as_sse(message: str) -> str:
    return SSEData.model_validate(
        {
            "id": f"error-{token_hex(16)}",
            "provider": "MCP-Bridge",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "MCP-Bridge",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": message,
                    },
                }
            ],
        }
    ).model_dump_json()


async def chat_completions(request: CreateChatCompletionRequest):
    """performs a chat completion using the inference server"""

    request.stream = True

    request = await chat_completion_add_tools(request)

    fully_done = False
    while not fully_done:
        # json_data = request.model_dump_json(
        #     exclude_defaults=True, exclude_none=True, exclude_unset=True
        # )

        json_data = json_pretty_print(chat_completion_requester(request))

        logger.debug("Request JSON:\n%s" % json_data)

        last: Optional[CreateChatCompletionStreamResponse] = None  # last message

        tool_call_name: str = ""
        tool_call_json: str = ""
        has_tool_calls: bool = False
        should_forward: bool = True
        response_content: str = ""
        tool_call_id: str = ""

        async with aconnect_sse(
            get_client(), "post", "/chat/completions", content=json_data
        ) as event_source:
            logger.debug(event_source.response.status_code)

            # check if the content type is correct because the aiter_sse method
            # will raise an exception if the content type is not correct
            if "Content-Type" in event_source.response.headers:
                content_type = event_source.response.headers["Content-Type"]
                if "application/json" in content_type:
                    logger.error(f"Unexpected Content-Type: {content_type}")
                    error_data = await event_source.response.aread()
                    # logger.error(f"Request URL: {event_source.response.url}")
                    # logger.error(f"Response Status: {event_source.response.status_code}")
                    # logger.error(f"Response Data: {error_data.decode(event_source.response.encoding or 'utf-8')}")
                    # raise HTTPException(status_code=500, detail="Unexpected Content-Type")
                    data = json.loads(
                        error_data.decode(event_source.response.encoding or "utf-8")
                    )
                    if message := data.get("error", {}).get("message"):
                        logger.error(f"Upstream error: {message}")
                        yield format_error_as_sse(message)
                        yield [
                            "DONE"
                        ]  # ServerSentEvent(event="message", data="[DONE]", id=None, retry=None)
                        return

                if "text/event-stream" not in content_type:
                    logger.error(f"Unexpected Content-Type: {content_type}")
                    error_data = await event_source.response.aread()
                    logger.error(f"Request URL: {event_source.response.url}")
                    logger.error(f"Request Data: {json_data}")
                    logger.error(
                        f"Response Status: {event_source.response.status_code}"
                    )
                    logger.error(
                        f"Response Data: {error_data.decode(event_source.response.encoding or 'utf-8')}"
                    )
                    yield format_error_as_sse("Upsteam error: Unexpected Content-Type")
                    yield ServerSentEvent(
                        event="message", data="[DONE]", id=None, retry=None
                    )
                    return

            # iterate over the SSE stream
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
                    logger.debug("inference serverstream done")
                    break

                # try to parse the data as json, if this fails we assume it is an error message
                # if parsing fails we send the error message to the client
                dict_data = json.loads(data)
                try:
                    parsed_data = chat_completion_stream_responder(dict_data)
                except Exception:
                    logger.debug(data)
                    try:
                        parsed_error_data = upstream_error.UpstreamError.model_validate_json(data)
                        yield format_error_as_sse(parsed_error_data.error.message)
                    except Exception:
                        yield format_error_as_sse(f"Error parsing response: {json.loads(data)}")

                    yield ServerSentEvent(event="message", data="[DONE]", id=None, retry=None)
                    return

                # handle empty response (usually caused by "usage" reporting)
                if len(parsed_data.choices) == 0:
                    logger.debug("no choices found in response")
                    continue

                # add the delta to the response content
                content = parsed_data.choices[0].delta.content
                content = content if content is not None else ""
                response_content += content

                # handle stop reasons
                if parsed_data.choices[0].finish_reason is not None:
                    if parsed_data.choices[0].finish_reason.value in [
                        "stop",
                        "length",
                    ]:
                        fully_done = True
                    else:
                        should_forward = False
                    
                    if parsed_data.choices[0].finish_reason.value == "tool_calls":
                        has_tool_calls = True

                # this manages the incoming tool call schema
                # most of this is assertions to please mypy
                if parsed_data.choices[0].delta.tool_calls is not None:
                    should_forward = False
                    assert (
                        parsed_data.choices[0].delta.tool_calls[0].function is not None
                    )

                    name = parsed_data.choices[0].delta.tool_calls[0].function.name
                    name = name if name is not None else ""
                    tool_call_name = name if tool_call_name == "" else tool_call_name

                    logger.debug(f"ARGS: {parsed_data.choices[0].delta.tool_calls[0].function.arguments}")

                    call_id = parsed_data.choices[0].delta.tool_calls[0].id
                    call_id = call_id if call_id is not None else ""
                    tool_call_id = id if tool_call_id == "" else tool_call_id

                    arg = parsed_data.choices[0].delta.tool_calls[0].function.arguments
                    tool_call_json += arg if arg is not None else ""

                # forward SSE messages to the client
                logger.debug(f"{should_forward=}")
                if should_forward:
                    # we do not want to forward tool call json to the client
                    logger.debug("forwarding message")
                    yield SSEData.model_validate_json(sse.data).model_dump_json()

                # save the last message
                last = parsed_data

                # perform early stopping on parsable tool_call_json
                if tool_call_json:
                    if tool_call_json.strip().startswith("{"):
                        if validate_if_json_object_parsable(tool_call_json):
                            logger.debug(
                                f"tool call json '{tool_call_json}' is parsable now."
                            )
                            logger.debug("exiting message receive loop")
                            last.choices[0].finish_reason = FinishReason1.tool_calls
                            break
                        salvaged_json_object = salvage_parsable_json_object(
                            tool_call_json
                        )
                        if salvaged_json_object:
                            tool_call_json = salvaged_json_object
                            logger.debug(
                                f"tool call json '{tool_call_json}' is salvagable now."
                            )
                            logger.debug("salvaged json content:", tool_call_json)
                            logger.debug("exiting message receive loop")
                            last.choices[0].finish_reason = FinishReason1.tool_calls
                            break

        # ideally we should check this properly
        assert last is not None

        if last.choices[0].finish_reason:
            if last.choices[0].finish_reason.value in ["stop", "length"]:
                logger.debug("no tool calls found")
                fully_done = True
                continue

        if last.choices[0].finish_reason is None and not has_tool_calls:
            logger.debug("no finish reason found")
            continue

        logger.debug("tool calls found")
        logger.debug(
            f"{tool_call_name=} {tool_call_json=}"
        )  # this should not be error but its easier to debug

        logger.debug("clearing tool contexts to prevent tool call loops")
        request.tools = None

        if tool_call_id is None or tool_call_id == "":
            tool_call_id = secrets.token_hex(16)

        # add received message to the history
        msg = ChatCompletionRequestMessage(
            role="assistant",
            content=response_content,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id=tool_call_id,
                    type="function",
                    function=Function1(name=tool_call_name, arguments=tool_call_json),
                )
            ],
        )  # type: ignore
        request.messages.append(msg)

        #### MOST OF THIS IS COPY PASTED FROM CHAT_COMPLETIONS
        # FIXME: this can probably be done in parallel using asyncio gather
        tool_call_result = await call_tool(tool_call_name, tool_call_json)
        if tool_call_result is None:
            continue

        logger.debug(
            f"tool call result for {tool_call_name}: {tool_call_result.model_dump()}"
        )

        logger.debug(f"tool call result content: {tool_call_result.content}")

        tools_content = [
            {"type": "text", "text": part.text}
            for part in filter(lambda x: x.type == "text", tool_call_result.content)
        ]
        if len(tools_content) == 0:
            tools_content = [{"type": "text", "text": "the tool call result is empty"}]
        request.messages.append(
            ChatCompletionRequestMessage.model_validate(
                {
                    "role": "tool",
                    "content": tools_content,
                    "tool_call_id": tool_call_id,
                }
            )
        )

        logger.debug("sending next iteration of chat completion request")

    # when done, send the final event
    logger.debug("sending final event")
    yield ServerSentEvent(event="message", data="[DONE]", id=None, retry=None)
