from lmos_openai_types import CreateChatCompletionResponse


def chat_completion_openrouter_response(data: dict) -> CreateChatCompletionResponse:
    return CreateChatCompletionResponse.model_validate(data)
