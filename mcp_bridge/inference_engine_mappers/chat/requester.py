from .generic import chat_completion_generic_request
from .openrouter.request import chat_completion_openrouter_request
from lmos_openai_types import CreateChatCompletionRequest
from mcp_bridge.config import config


def chat_completion_requester(data: CreateChatCompletionRequest) -> dict:
    client_type = config.inference_server.type

    match client_type:
        # apply incoming data mappers
        case "openai":
            return chat_completion_generic_request(data)
        case "openrouter":
            # TODO: implement openrouter requester
            return chat_completion_openrouter_request(data)
        case _:
            return chat_completion_generic_request(data)