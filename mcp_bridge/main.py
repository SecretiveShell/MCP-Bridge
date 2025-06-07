import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from mcp_bridge import __version__ as version
from mcp_bridge.config import config
from mcp_bridge.routers import secure_router, public_router
from mcp_bridge.lifespan import lifespan
from mcp_bridge.openapi_tags import tags_metadata
from mcp_bridge.telemetry import setup_tracing


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="MCP Bridge",
        description="A middleware application to add MCP support to OpenAI-compatible APIs",
        version=version,
        lifespan=lifespan,
        openapi_tags=tags_metadata,
    )

    # setup tracing
    setup_tracing(app)

    # show auth data
    if config.security.auth.enabled:
        logger.info("Authentication is enabled")
    else:
        logger.info("Authentication is disabled")
    
    # Add CORS middleware
    if config.security.CORS.enabled:
        if config.security.CORS.allow_origins == ["*"]:
            logger.warning("CORS middleware is enabled with wildcard origins")
        else:
            logger.info("CORS middleware is enabled")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.security.CORS.allow_origins,
            allow_credentials=config.security.CORS.allow_credentials,
            allow_methods=config.security.CORS.allow_methods,
            allow_headers=config.security.CORS.allow_headers,
        )
    else:
        logger.info("CORS middleware is disabled")

    app.include_router(secure_router)
    app.include_router(public_router)

    return app

app = create_app()

def run():
    import uvicorn
    uvicorn.run(
        app,
        host=config.network.host,
        port=config.network.port,
        root_path=config.network.root_path,
    )

if __name__ == "__main__":
    run()