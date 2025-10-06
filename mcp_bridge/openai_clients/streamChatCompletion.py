import json
from typing import Optional
from fastapi import HTTPException, Request
from lmos_openai_types import (
    ChatCompletionMessageToolCall,
    ChatCompletionRequestMessage,
    CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse,
    Function1,
)
from .utils import call_tool, chat_completion_add_tools
from mcp_bridge.models import SSEData
from .genericHttpxClient import get_client
from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from mcp_bridge.tool_mappers import mcp2openai
from loguru import logger
from httpx_sse import aconnect_sse

from sse_starlette.sse import EventSourceResponse, ServerSentEvent


async def streaming_chat_completions(request: CreateChatCompletionRequest, http_request: Request):
    # raise NotImplementedError("Streaming Chat Completion is not supported")

    try:
        return EventSourceResponse(
            content=chat_completions(request, http_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    except Exception as e:
        logger.error(e)


async def chat_completions(request: CreateChatCompletionRequest, http_request: Request):
    """performs a chat completion using the inference server"""

    request.stream = True

    request = await chat_completion_add_tools(request)

    fully_done = False
    while not fully_done:
        # json_data = request.model_dump_json(
        #     exclude_defaults=True, exclude_none=True, exclude_unset=True
        # )

        json_data = json.dumps(request.model_dump(
            exclude_defaults=True, exclude_none=True, exclude_unset=True
        ))

        # logger.debug(json_data)

        last: Optional[CreateChatCompletionStreamResponse] = None  # last message

        tool_call_name: str = ""
        tool_call_json: str = ""
        should_forward: bool = True
        response_content: str = ""
        tool_call_id: str = ""

        async with get_client(http_request) as client:
            async with aconnect_sse(
                client, "post", "/chat/completions", content=json_data
            ) as event_source:
                
                # check if the content type is correct because the aiter_sse method
                # will raise an exception if the content type is not correct
                if "Content-Type" in event_source.response.headers:
                    content_type = event_source.response.headers["Content-Type"]
                    if "text/event-stream" not in content_type:
                        logger.error(f"Unexpected Content-Type: {content_type}")
                        error_data = await event_source.response.aread()
                        logger.error(f"Request URL: {event_source.response.url}")
                        logger.error(f"Request Data: {json_data}")
                        logger.error(f"Response Status: {event_source.response.status_code}")
                        logger.error(f"Response Data: {error_data.decode(event_source.response.encoding or 'utf-8')}")
                        raise HTTPException(status_code=500, detail="Unexpected Content-Type")

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

                    # for some reason openrouter uses uppercase for finish_reason
                    try:
                        data['choices'][0]['finish_reason'] = data['choices'][0]['finish_reason'].lower() # type: ignore
                    except Exception as e:
                        logger.debug(f"failed to lowercase finish_reason: {e}")

                    try:
                        parsed_data = CreateChatCompletionStreamResponse.model_validate_json(
                            data
                        )
                    except Exception as e:
                        logger.debug(data)
                        raise e

                    # add the delta to the response content
                    content = parsed_data.choices[0].delta.content if len(parsed_data.choices) > 0 else ""
                    content = content if content is not None else ""
                    response_content += content

                    # handle stop reasons
                    if  len(parsed_data.choices) > 0 and parsed_data.choices[0].finish_reason is not None:
                        if parsed_data.choices[0].finish_reason.value in [
                            "stop",
                            "length",
                        ]:
                            fully_done = True
                        else:
                            should_forward = False

                    # this manages the incoming tool call schema
                    # most of this is assertions to please mypy
                    if len(parsed_data.choices) > 0 and parsed_data.choices[0].delta.tool_calls is not None:
                        should_forward = False
                        assert (
                            parsed_data.choices[0].delta.tool_calls[0].function is not None
                        )

                        name = parsed_data.choices[0].delta.tool_calls[0].function.name
                        name = name if name is not None else ""
                        tool_call_name = name if tool_call_name == "" else tool_call_name

                        call_id = parsed_data.choices[0].delta.tool_calls[0].id
                        call_id = call_id if call_id is not None else ""
                        tool_call_id = call_id if tool_call_id == "" else tool_call_id

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

        # ideally we should check this properly
        assert last is not None
        if len(last.choices) > 0:
            assert last.choices[0].finish_reason is not None

        if len(last.choices) > 0 and last.choices[0].finish_reason.value in ["stop", "length"]:
            logger.debug("no tool calls found")
            fully_done = True
            continue

        logger.debug("tool calls found")
        logger.debug(
            f"{tool_call_name=} {tool_call_json=}"
        )  # this should not be error but its easier to debug

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
