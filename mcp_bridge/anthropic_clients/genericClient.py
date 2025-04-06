import anthropic
from mcp_bridge.config import config
from loguru import logger
import boto3
import json
from typing import Dict, Any, Optional

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

# Create AWS Bedrock client if configured
bedrock_client = None
try:
    if hasattr(config.inference_server, "use_bedrock") and config.inference_server.use_bedrock:
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=getattr(config.inference_server, "aws_region", "us-east-1"),
            aws_access_key_id=getattr(config.inference_server, "aws_access_key_id", None),
            aws_secret_access_key=getattr(config.inference_server, "aws_secret_access_key", None),
        )
        logger.info("Initialized AWS Bedrock client successfully")
except Exception as e:
    logger.error(f"Failed to initialize AWS Bedrock client: {e}")
    bedrock_client = None

async def create_messages(**params):
    """
    Common wrapper for creating messages with either Anthropic or AWS Bedrock.
    
    Args:
        params: Parameters for the API call
        
    Returns:
        API response from either Anthropic or AWS Bedrock
    """
    # Determine which client to use
    if bedrock_client is not None and hasattr(config.inference_server, "use_bedrock") and config.inference_server.use_bedrock:
        logger.info("Using AWS Bedrock for Claude API call")
        return await _create_messages_bedrock(**params)
    else:
        logger.info("Using Anthropic direct API for Claude API call")
        return client.beta.messages.create(**params)

async def _create_messages_bedrock(**params):
    """
    Create messages using AWS Bedrock API for Claude models.
    
    Args:
        params: Parameters for the API call
        
    Returns:
        API response converted to match Anthropic's format
    """
    if bedrock_client is None:
        raise ValueError("AWS Bedrock client is not initialized")
    
    # Extract model ID from the full model string (bedrock format uses model IDs without anthropic. prefix)
    # check https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
    model_id = params.get("model", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    
    # Convert params to Bedrock format
    # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking

    bedrock_params = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": params.get("max_tokens", 1024),
        "temperature": 1, # `temperature` may only be set to 1 when thinking is enabled.
        # "top_p": params.get("top_p", 0.7), # `top_p` must be unset when thinking is enabled.
        "thinking": params["thinking"]
    }

    reasoning_config = {
        "thinking": params["thinking"]
    }
    
    # Add system if provided
    if "system" in params:
        bedrock_params["system"] = params["system"]
    
    # Add tools if provided
    if "tools" in params and params["tools"]:
        bedrock_params["tools"] = params["tools"]

    # Add messages to params
    bedrock_params["messages"] = params['messages']
    
    # Convert to JSON
    request_body = json.dumps(bedrock_params)
    # logger.info(f"Bedrock request body: {request_body}")
    
    # Call Bedrock API
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=request_body,
        contentType="application/json",
        accept="application/json",
    )
    
    # Parse response
    response_body = json.loads(response["body"].read().decode())
    logger.info(f"Bedrock response body: {response_body}")
    # Convert Bedrock response to match Anthropic response format
    anthropic_response = type('AnthropicResponse', (), {})()
    anthropic_response.id = response_body.get("id", "bedrock-response")
    anthropic_response.model = model_id
    anthropic_response.stop_reason = response_body.get("stop_reason", "stop")
    anthropic_response.stop_sequence = response_body.get("stop_sequence", None)
    
    # Extract content
    anthropic_response.content = []
    if "content" in response_body:
        for item in response_body["content"]:
            if item.get("type") == "text":
                text_block = type('TextBlock', (), {})()
                text_block.type = "text"
                text_block.text = item.get("text", "")
                anthropic_response.content.append(text_block)
            elif item.get("type") == "tool_use":
                tool_block = type('ToolUseBlock', (), {})()
                tool_block.type = "tool_use"
                tool_block.id = item.get("id", "")
                tool_block.name = item.get("name", "")
                tool_block.input = item.get("input", {})
                anthropic_response.content.append(tool_block)
            elif item.get("type") in ["thinking", "redacted_thinking"]:
                thinking_block = type('ThinkingBlock', (), {})()
                thinking_block.type = item.get("type")
                thinking_block.thinking = item.get("thinking", "")
                thinking_block.signature = item.get("signature", "")
                anthropic_response.content.append(thinking_block)
    
    # Set usage information
    usage = type('Usage', (), {})()
    if "usage" in response_body:
        usage.input_tokens = response_body["usage"].get("input_tokens", 0)
        usage.output_tokens = response_body["usage"].get("output_tokens", 0)
    else:
        usage.input_tokens = 0
        usage.output_tokens = 0
    anthropic_response.usage = usage
    
    return anthropic_response 