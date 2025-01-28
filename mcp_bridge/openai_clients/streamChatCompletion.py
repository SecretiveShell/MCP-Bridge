import json
from typing import Optional
from fastapi import HTTPException
from lmos_openai_types import (
    ChatCompletionMessageToolCall,
    ChatCompletionRequestMessage,
    CreateChatCompletionRequest,
    CreateChatCompletionStreamResponse,
    Function1,
    FinishReason1,
    ChatCompletionToolChoiceOption1,
    ChatCompletionToolChoiceOption,
)
from .utils import call_tool, chat_completion_add_tools
from mcp_bridge.models import SSEData
from .genericHttpxClient import client
from mcp_bridge.mcp_clients.McpClientManager import ClientManager
from mcp_bridge.tool_mappers import mcp2openai
from loguru import logger
from httpx_sse import aconnect_sse
import datetime
import os

from sse_starlette.sse import EventSourceResponse, ServerSentEvent
import json
import traceback


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


def validate_if_json_object_parsable(content: str):
    try:
        json.loads(content)
        return True
    except ValueError:
        return False


def salvage_parsable_json_object(content: str):
    content = content.strip()
    for i in range(0, len(content)):
        snippet = content[: len(content) - i]
        if validate_if_json_object_parsable(snippet):
            return snippet


async def chat_completions(request: CreateChatCompletionRequest):
    """performs a chat completion using the inference server"""

    request.stream = True

    request = await chat_completion_add_tools(
        request
    )  # Date: 2025/01/27 ChatMCP clear tools after first tool call.

    fully_done = False
    while not fully_done:
        # json_data = request.model_dump_json(
        #     exclude_defaults=True, exclude_none=True, exclude_unset=True
        # )
        if request.tools:
            request.tool_choice = ChatCompletionToolChoiceOption(
                root=ChatCompletionToolChoiceOption1.auto
            )

        json_data = json.dumps(
            request.model_dump(
                exclude_defaults=True,
                exclude_none=True,
                exclude_unset=True,
            ),
            indent=4,
            ensure_ascii=False,
        )

        logger.debug("Request JSON:\n%s" % json_data)  # empty?

        last: Optional[CreateChatCompletionStreamResponse] = None  # last message

        tool_call_name: str = ""
        tool_call_json: str = ""
        should_forward: bool = True
        response_content: str = ""
        tool_call_id: str = ""

        async with aconnect_sse(
            client, "post", "/chat/completions", content=json_data
        ) as event_source:

            # check if the content type is correct because the aiter_sse method
            # will raise an exception if the content type is not correct
            if "Content-Type" in event_source.response.headers:  # error here.
                content_type = event_source.response.headers["Content-Type"]
                if "text/event-stream" not in content_type:
                    logger.error(f"Unexpected Content-Type: {content_type}")
                    error_data = await event_source.response.aread()
                    logger.error(f"Request URL: {event_source.response.url}")
                    log_dir = os.path.join(os.getcwd(), "logs")
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    request_data_path = f"{log_dir}/request_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    request_data_formatted = json.dumps(
                        json.loads(json_data), indent=4, ensure_ascii=False
                    )
                    with open(request_data_path, "w+") as f:
                        f.write(request_data_formatted)
                    logger.error(f"Request Data saved to: {request_data_path}")
                    logger.error(f"Request Data:\n{request_data_formatted}")
                    logger.error(
                        f"Response Status: {event_source.response.status_code}"
                    )
                    error_data_decoded = error_data.decode(
                        event_source.response.encoding or "utf-8"
                    )
                    error_data_path = f"{log_dir}/error_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    logger.error(f"Response Data saved to: {error_data_path}")
                    logger.error(f"Response Data:\n{error_data_decoded}")
                    with open(error_data_path, "w+") as f:
                        f.write(error_data_decoded)
                    raise HTTPException(
                        status_code=500, detail="Unexpected Content-Type"
                    )

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
                    mjson_data = json.loads(data)
                    
                    # Date: 2025/01/26 failed to lowercase finish_reason: string indices must be integers, not 'str'
                    if mjson_data["choices"][0].keys().__contains__("finish_reason"):  # type: ignore
                        mjson_data["choices"][0]["finish_reason"] = mjson_data["choices"][0]["finish_reason"].lower()  # type: ignore
                        
                    data = json.dumps(mjson_data, ensure_ascii=False)
                except Exception as e:
                    traceback.print_exc()
                    logger.debug(f"failed to lowercase finish_reason: {e}")

                try:
                    parsed_data = (
                        CreateChatCompletionStreamResponse.model_validate_json(data)
                    )
                except Exception as e:
                    logger.debug(data)
                    raise e


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

                    call_id = parsed_data.choices[0].delta.tool_calls[0].id
                    call_id = call_id if call_id is not None else ""
                    tool_call_id = id if tool_call_id == "" else tool_call_id

                    arg = parsed_data.choices[0].delta.tool_calls[0].function.arguments

                    tool_call_json += arg if arg is not None else ""
                    # Date: 2025/01/26 validate the tool call json.

                # forward SSE messages to the client
                logger.debug(f"{should_forward=}")
                if should_forward:
                    # we do not want to forward tool call json to the client
                    logger.debug("forwarding message")
                    yield SSEData.model_validate_json(sse.data).model_dump_json()

                # save the last message
                last = parsed_data

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
        assert last.choices[0].finish_reason is not None

        if last.choices[0].finish_reason.value in ["stop", "length"]:
            logger.debug("no tool calls found")
            fully_done = True
            continue

        logger.debug("tool calls found")
        logger.debug(
            f"{tool_call_name=} {tool_call_json=}"
        )  # this should not be error but its easier to debug

        logger.debug("clearing tool contexts to prevent tool call loops")
        request.tools = None

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
        # Date: 2025/01/26 decoding error?
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

        # Date: 2025/01/26 crucial! we have to ensure the llm does not end up with infinite loop.

        # request.messages.append(
        #     ChatCompletionRequestMessage.model_validate(
        #         {"role": "user", "content": "Do you consider you have done enough tool calls? If not, please continue the rest of the tool calls. If yes, please respond to the user and end the conversation."}
        #     )
        # )

        logger.debug("sending next iteration of chat completion request")

    # when done, send the final event
    logger.debug("sending final event")
    yield ServerSentEvent(event="message", data="[DONE]", id=None, retry=None)
