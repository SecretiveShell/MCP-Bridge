from httpx import AsyncClient
from mcp_bridge.config import config
from fastapi import Request
from contextlib import asynccontextmanager

async def create_client(request: Request = None):
    """Tworzy nową instancję klienta z odpowiednimi nagłówkami"""
    client = AsyncClient(
        base_url=config.inference_server.base_url,
        headers={
            "Authorization": f"Bearer {config.inference_server.api_key}",
            "Content-Type": "application/json"
        },
        timeout=10000,
    )
    
    if request:
        # Dodaj nagłówki z żądania
        headers = {k.lower(): v for k, v in request.headers.items()}
        
        openwebui_headers = [
            "x-openwebui-user-name",
            "x-openwebui-user-id",
            "x-openwebui-user-email",
            "x-openwebui-user-role"
        ]
        
        for header in openwebui_headers:
            if header in headers:
                client.headers[header] = headers[header]
    
    return client

@asynccontextmanager
async def get_client(request: Request = None):
    """Context manager dla klienta HTTP"""
    client = await create_client(request)
    try:
        yield client
    finally:
        await client.aclose()
