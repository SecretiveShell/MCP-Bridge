import secrets
import time
from turtle import st
from lmos_openai_types import (
    ChatCompletionResponseMessage,
    Choice1,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    ChatCompletionRequestMessage,
    FinishReason1,
)

from .utils import call_tool, chat_completion_add_tools, validate_if_json_object_parsable, json_pretty_print
from mcp_bridge.http_clients import get_client
from mcp_bridge.inference_engine_mappers.chat.requester import chat_completion_requester
from mcp_bridge.inference_engine_mappers.chat.responder import chat_completion_responder
from loguru import logger
import json

def format_error_as_chat_completion(message: str) -> CreateChatCompletionResponse:
    return CreateChatCompletionResponse.model_validate(
        {
            "model": "MCP-Bridge",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "content": message,
                        "role": "assistant",
                    }
                }
            ],
            "id": secrets.token_hex(16),
            "created": int(time.time()),
            "object": "chat.completion",
        }
    )

async def chat_completions(
    request: CreateChatCompletionRequest,
) -> CreateChatCompletionResponse:
    """performs a chat completion using the inference server"""

    request = await chat_completion_add_tools(request)

    while True:
        # logger.debug(request.model_dump_json())

        response = await get_client().post(
                "/chat/completions",
                json=chat_completion_requester(request),
            )
        text = response.text
        logger.debug(text)
        try:
            response = chat_completion_responder(json.loads(text))
        except Exception as e:
            logger.error(f"Error parsing response: {text}")
            logger.error(e)
            
            # openrouter returns a json error message
            try :
                response = json.loads(text)
                return format_error_as_chat_completion(f"Upstream error: {response['error']['message']}")
            except Exception:
                pass

            return format_error_as_chat_completion(f"Error parsing response: {text}")

        if not response.choices:
            logger.error("no choices found in response")
            return format_error_as_chat_completion("no choices found in response")

        msg = response.choices[0].message
        msg = ChatCompletionRequestMessage(
            role="assistant",
            content=msg.content,
            tool_calls=msg.tool_calls,
        )  # type: ignore
        request.messages.append(msg)

        logger.debug(f"finish reason: {response.choices[0].finish_reason}")
        if response.choices[0].finish_reason.value in ["stop", "length"]:
            logger.debug("no tool calls found")
            return response

        logger.debug("tool calls found")
        
        logger.debug("clearing tool contexts to prevent tool call loops")
        request.tools = None
        
        for tool_call in response.choices[0].message.tool_calls.root:
            logger.debug(
                f"tool call: {tool_call.function.name}"
            )

            if validate_if_json_object_parsable(tool):
                logger.debug(f"arguments:\n{json_pretty_print(tool_call.function.arguments)}")
            else:
                logger.debug("non-json arguments given: %s" % tool_call.function.arguments)
                logger.debug("unable to parse tool call argument as json. skipping...")
                continue

            # FIXME: this can probably be done in parallel using asyncio gather
            tool_call_result = await call_tool(
                tool_call.function.name, tool_call.function.arguments
            )
            if tool_call_result is None:
                continue

            logger.debug(
                f"tool call result for {tool_call.function.name}: {tool_call_result.model_dump()}"
            )

            logger.debug(f"tool call result content: {tool_call_result.content}")

            tools_content = [
                {"type": "text", "text": part.text}
                for part in filter(lambda x: x.type == "text", tool_call_result.content)
            ]
            if len(tools_content) == 0:
                tools_content = [
                    {"type": "text", "text": "the tool call result is empty"}
                ]
            request.messages.append(
                ChatCompletionRequestMessage.model_validate(
                    {
                        "role": "tool",
                        "content": tools_content,
                        "tool_call_id": tool_call.id or secrets.token_hex(16),
                    }
                )
            )

        logger.debug("sending next iteration of chat completion request")
