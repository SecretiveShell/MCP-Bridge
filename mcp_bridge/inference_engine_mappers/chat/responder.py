from mcp_bridge.inference_engine_mappers.chat.gemini.response import chat_completion_gemini_response
from .generic import chat_completion_generic_response
from .openrouter.response import chat_completion_openrouter_response
from lmos_openai_types import CreateChatCompletionResponse
from mcp_bridge.config import config


def chat_completion_responder(data: dict) -> CreateChatCompletionResponse:
    client_type = config.inference_server.type

    match client_type:
        # apply incoming data mappers
        case "openai":
            return chat_completion_generic_response(data)
        case "openrouter":
            # TODO: implement openrouter responser
            return chat_completion_openrouter_response(data)
        case "gemini":
            return chat_completion_gemini_response(data)
        case _:
            return chat_completion_generic_response(data)
