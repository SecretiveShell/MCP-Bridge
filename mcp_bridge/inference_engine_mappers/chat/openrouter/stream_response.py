from lmos_openai_types import CreateChatCompletionStreamResponse


def chat_completion_openrouter_stream_response(
    data: dict,
) -> CreateChatCompletionStreamResponse:  # type: ignore
    try:
        data["choices"][0]["finish_reason"] = data["choices"][0][
            "finish_reason"
        ].lower()  # type: ignore
    except Exception:
        pass
    return CreateChatCompletionStreamResponse.model_validate(data)
