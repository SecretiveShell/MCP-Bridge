from .generic import chat_completion_generic_stream_response
from .openrouter.stream_response import chat_completion_openrouter_stream_response
from lmos_openai_types import CreateChatCompletionStreamResponse
from mcp_bridge.config import config


def chat_completion_stream_responder(data: dict) -> CreateChatCompletionStreamResponse:
    client_type = config.inference_server.type

    match client_type:
        # apply incoming data mappers
        case "openai":
            return chat_completion_generic_stream_response(data)
        case "openrouter":
            # TODO: implement openrouter responser
            return chat_completion_openrouter_stream_response(data)
        case _:
            return chat_completion_generic_stream_response(data)
