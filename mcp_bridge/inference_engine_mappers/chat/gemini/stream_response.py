from lmos_openai_types import CreateChatCompletionStreamResponse
from loguru import logger


def chat_completion_gemini_stream_response(
    data: dict,
) -> CreateChatCompletionStreamResponse:  # type: ignore
    
    logger.debug(f"data: {data}")
    
    if "id" not in data or data["id"] == "":
        data["id"] = "default-id"

    return CreateChatCompletionStreamResponse.model_validate(data)
