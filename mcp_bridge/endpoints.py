from fastapi import APIRouter

from lmos_openai_types import CreateChatCompletionRequest, CreateCompletionRequest

from openai_clients import (
    client,
    completions,
    chat_completions,
    streaming_chat_completions,
)

from openapi_tags import Tag

router = APIRouter(prefix="/v1", tags=[Tag.openai])


@router.post("/completions")
async def openai_completions(request: CreateCompletionRequest):
    """Completions endpoint"""
    if request.stream:
        raise NotImplementedError("Streaming Completion is not supported")
    else:
        return await completions(request)


@router.post("/chat/completions")
async def openai_chat_completions(request: CreateChatCompletionRequest):
    """Chat Completions endpoint"""
    if request.stream:
        return await streaming_chat_completions(request)
    else:
        return await chat_completions(request)


@router.get("/models")
async def models():
    """List models"""
    response = await client.get("/models")
    return response.json()
