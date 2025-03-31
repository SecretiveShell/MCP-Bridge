import anthropic
from mcp_bridge.config import config
from loguru import logger

# Create a singleton Anthropic client using the config
try:
    api_key = config.inference_server.api_key
    if not api_key:
        logger.error("No API key found in config.json inference_server.api_key")
        client = None
    else:
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Initialized Anthropic client successfully")
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {e}")
    client = None 