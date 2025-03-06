from fastapi import APIRouter, Depends
from .sse import router as sse_router
from mcp_bridge.openapi_tags import Tag
from mcp_bridge.auth import get_api_key

__all__ = ["router"]

router = APIRouter(prefix="/mcp-server", tags=[Tag.mcp_server])
router.include_router(sse_router)
