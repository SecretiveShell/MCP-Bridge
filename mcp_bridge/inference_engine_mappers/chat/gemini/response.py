from lmos_openai_types import CreateChatCompletionResponse


def chat_completion_gemini_response(data: dict) -> CreateChatCompletionResponse:
    
    if "id" not in data or data["id"] is "":
        data["id"] = "default-id"

    validated_data = CreateChatCompletionResponse.model_validate(data)
    return validated_data