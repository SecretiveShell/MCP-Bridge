from typing import Any

from lmos_openai_types import CreateChatCompletionResponse
from loguru import logger
from mcp import ClientSession, ErrorData, SamplingMessage
import mcp.types as types
from mcp.shared.context import RequestContext
from mcp.types import CreateMessageRequestParams, CreateMessageResult

from mcp_bridge.config import config
from mcp_bridge.openai_clients.genericHttpxClient import get_client
from mcp_bridge.sampling.modelSelector import find_best_model


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


async def sampling_callback(
    context: RequestContext[ClientSession, Any], params: CreateMessageRequestParams
) -> CreateMessageResult | ErrorData:
    """perform sampling"""

    logger.debug(f"sampling message: {params.modelPreferences}")

    # select model
    model = config.sampling.models[0]
    if params.modelPreferences is not None:
        model = find_best_model(params.modelPreferences)

    logger.debug(f"selected model: {model.model}")

    logger.debug("sending sampling request to endpoint")
    # request = CreateChatCompletionRequest(model=model.model, messages=message.messages, stream=False)  # type: ignore
    request = {
        "model": model.model,
        "messages": [make_message(x) for x in params.messages],
        "stream": False,
    }

    logger.debug(f"request: {request}")

    logger.debug(request)

    async with get_client() as client:
        resp = await client.post(
            "/chat/completions",
            json=request,
            timeout=config.sampling.timeout,
        )

    logger.debug("parsing json")
    text = resp.text
    logger.debug(text)

    response = CreateChatCompletionResponse.model_validate_json(text)

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
