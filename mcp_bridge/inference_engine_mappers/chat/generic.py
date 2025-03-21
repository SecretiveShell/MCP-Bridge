from lmos_openai_types import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)


def chat_completion_generic_request(data: CreateChatCompletionRequest) -> dict:
    return data.model_dump(exclude_defaults=True, exclude_none=True, exclude_unset=True)


def chat_completion_generic_response(data: dict) -> CreateChatCompletionResponse:
    return CreateChatCompletionResponse.model_validate(data)


def chat_completion_generic_stream_response(
    data: dict,
) -> CreateChatCompletionStreamResponse:
    return CreateChatCompletionStreamResponse.model_validate(data)
