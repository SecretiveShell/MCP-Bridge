from lmos_openai_types import CreateChatCompletionRequest


def chat_completion_gemini_request(data: CreateChatCompletionRequest) -> dict:

    dumped_data = data.model_dump(exclude_defaults=True, exclude_none=True, exclude_unset=True)

    return dumped_data
