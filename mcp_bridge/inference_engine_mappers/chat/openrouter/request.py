import json
import secrets
from typing import Any, cast
from lmos_openai_types import CreateChatCompletionRequest
from loguru import logger


def chat_completion_openrouter_request(data: CreateChatCompletionRequest) -> dict:

    dumped_data = data.model_dump(exclude_defaults=True, exclude_none=True, exclude_unset=True)

    # make sure we have a tool call id for each tool call
    try:
        for message in dumped_data["messages"]:
            
            message = cast(dict[str, Any], message)

            if message["role"] == "assistant":
                if message.get("tool_calls") is None:
                    continue
                for tool_call in message["tool_calls"]:
                    tool_call["tool_call_id"] = tool_call.get("id", secrets.token_hex(16))

            if message["role"] == "tool":
                if message.get("tool_call_id") is None:
                    message["tool_call_id"] = secrets.token_hex(16)
                if message.get("id") is None:
                    message["id"] = message["tool_call_id"]

    except Exception as e:
        print(e)

    logger.debug(f"dumped data: {dumped_data}")
    logger.debug(f"json dumped data: {json.dumps(dumped_data)}")

    return dumped_data
