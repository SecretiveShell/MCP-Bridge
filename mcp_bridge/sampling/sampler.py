import json
from loguru import logger
from mcp import SamplingMessage
import mcp.types as types
from mcp.types import CreateMessageRequestParams, CreateMessageResult

from mcp_bridge.config import config
from mcp_bridge.http_clients import get_client
from mcp_bridge.sampling.modelSelector import find_best_model
from mcp_bridge.inference_engine_mappers.chat.generic import chat_completion_generic_response


def make_message(x: SamplingMessage):
    if x.content.type == "text":
        return {
            "role": x.role,
            "content": [
                {
                    "type": "text",
                    "text": x.content.text,
                }
            ],
        }
    if x.content.type == "image":
        return {
            "role": x.role,
            "content": [
                {
                    "type": "image",
                    "image_url": x.content.data,
                }
            ],
        }


async def handle_sampling_message(
    message: CreateMessageRequestParams,
) -> CreateMessageResult:
    """perform sampling"""

    logger.debug(f"sampling message: {message.modelPreferences}")

    # select model
    model = config.sampling.models[0]
    if message.modelPreferences is not None:
        model = find_best_model(message.modelPreferences)

    logger.debug(f"selected model: {model.model}")

    logger.debug("sending sampling request to endpoint")
    # request = CreateChatCompletionRequest(model=model.model, messages=message.messages, stream=False)  # type: ignore
    request = {
        "model": model.model,
        "messages": [make_message(x) for x in message.messages],
        "stream": False,
    }

    logger.debug(f"request: {request}")

    logger.debug(request)

    resp = await get_client().post(
        "/chat/completions",
        json=request,
        timeout=config.sampling.timeout,
    )

    logger.debug("parsing json")
    text = resp.text
    logger.debug(text)

    response = chat_completion_generic_response(json.loads(text))

    logger.debug("sampling request received from endpoint")

    assert response.choices is not None
    assert len(response.choices) > 0
    assert response.choices[0].message is not None
    assert response.choices[0].message.content is not None

    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text=response.choices[0].message.content,
        ),
        model=model.model,
        stopReason=response.choices[0].finish_reason,
    )
