import secrets
from typing import cast
from lmos_openai_types import CreateChatCompletionResponse
from loguru import logger


def chat_completion_openrouter_response(data: dict) -> CreateChatCompletionResponse:
    validated_data = CreateChatCompletionResponse.model_validate(data)
    
    # make sure tool call ids are not none
    for choice in validated_data.choices:
        if choice.message.tool_calls is None:
            continue
        for tool_call in choice.message.tool_calls:
            logger.error(f"tool call: {tool_call[1]}")
            for calls in tool_call[1]:
                if calls.id is None:
                    calls.id = secrets.token_hex(16)

    logger.debug(f"validated data: {validated_data.model_dump_json()}")
    return validated_data