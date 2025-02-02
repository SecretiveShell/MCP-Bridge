from lmos_openai_types import CreateChatCompletionRequest


def chat_completion_openrouter_request(data: CreateChatCompletionRequest) -> dict:
    return data.model_dump(exclude_defaults=True, exclude_none=True, exclude_unset=True)
