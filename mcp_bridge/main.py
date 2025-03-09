from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp_bridge.endpoints import router as endpointRouter
from mcp_bridge.mcpManagement import router as mcpRouter
from mcp_bridge.health import router as healthRouter
from mcp_bridge.mcp_server import router as mcp_server_router
from mcp_bridge.lifespan import lifespan
from mcp_bridge.openapi_tags import tags_metadata
from mcp_bridge import __version__ as version
from mcp_bridge.config import config
from loguru import logger


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

    app.include_router(endpointRouter)
    app.include_router(mcpRouter)
    app.include_router(healthRouter)
    app.include_router(mcp_server_router)

    return app


app = create_app()


def run():
    import uvicorn
    from mcp_bridge.config import config

    uvicorn.run(app, host=config.network.host, port=config.network.port)


if __name__ == "__main__":
    run()
