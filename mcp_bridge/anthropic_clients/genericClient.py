import anthropic
from mcp_bridge.config import config
from loguru import logger
import boto3
import json
import time
from typing import Dict, Any, Optional
from botocore.config import Config
from collections import deque
import asyncio

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
        # Configure Bedrock client with retry settings
        boto_config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'  # Uses adaptive retry mode with backoff
            },
            connect_timeout=5,
            read_timeout=120
        )
        
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=getattr(config.inference_server, "aws_region", "us-east-1"),
            aws_access_key_id=getattr(config.inference_server, "aws_access_key_id", None),
            aws_secret_access_key=getattr(config.inference_server, "aws_secret_access_key", None),
            config=boto_config
        )
        logger.info("Initialized AWS Bedrock client successfully with retry configuration")
except Exception as e:
    logger.error(f"Failed to initialize AWS Bedrock client: {e}")
    bedrock_client = None

# Token rate limiter for Bedrock (1M tokens per minute)
class TokenRateLimiter:
    def __init__(self, limit=1000000, window_size=60):
        self.limit = limit  # Maximum tokens per minute
        self.window_size = window_size  # Time window in seconds
        self.usage_history = deque()  # Store (timestamp, tokens) pairs
        self.total_tokens = 0  # Current total in the window

    async def add_usage(self, tokens):
        """Add token usage and wait if the rate limit would be exceeded"""
        current_time = time.time()
        
        # Remove old entries outside the window
        while self.usage_history and self.usage_history[0][0] < current_time - self.window_size:
            _, old_tokens = self.usage_history.popleft()
            self.total_tokens -= old_tokens
        
        # Check if adding these tokens would exceed the limit
        if self.total_tokens + tokens > self.limit:
            # Calculate wait time needed
            needed_reduction = self.total_tokens + tokens - self.limit
            
            if self.usage_history:
                # Estimate time until enough tokens are freed
                oldest_time = self.usage_history[0][0]
                wait_time = (oldest_time + self.window_size) - current_time
                wait_time = max(1, min(wait_time, self.window_size))  # Bound wait time
                
                logger.warning(f"Rate limit would be exceeded. Waiting {wait_time:.2f}s for token budget to free up")
                await asyncio.sleep(wait_time)
                
                # Recursive call to check again after waiting
                await self.add_usage(tokens)
                return
        
        # Add the usage to our window
        self.usage_history.append((current_time, tokens))
        self.total_tokens += tokens
        logger.info(f"Added {tokens} tokens. Current usage: {self.total_tokens}/{self.limit} tokens in the last {self.window_size}s")

# Initialize rate limiter
token_limiter = TokenRateLimiter()

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
    
    # Predict token usage (simple heuristic - can be improved)
    # Rough estimate of input tokens based on message content
    estimated_input_tokens = 0
    for msg in params['messages']:
        if isinstance(msg.get('content', ''), str):
            estimated_input_tokens += int(len(msg.get('content', '')) / 4)  # Rough estimate
        elif isinstance(msg.get('content', []), list):
            for content in msg.get('content', []):
                if isinstance(content, dict) and 'text' in content:
                    estimated_input_tokens += len(content.get('text', '')) / 4
    
    # Add estimated tokens for system prompt if present
    if "system" in params:
        estimated_input_tokens += len(params["system"]) / 4
    
    # Add tokens for tools if present
    if "tools" in params and params["tools"]:
        estimated_input_tokens += 500  # Rough estimate for tools
    
    # Add the max_tokens as the upper bound for output
    estimated_total_tokens = estimated_input_tokens + params.get("max_tokens", 1024)
    
    # Check rate limit before making the API call
    await token_limiter.add_usage(estimated_total_tokens)
    
    # Retry mechanism for throttling exceptions
    max_retries = 5
    base_delay = 2  # Starting delay in seconds
    
    for retry in range(max_retries + 1):
        try:
            # Call Bedrock API
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=request_body,
                contentType="application/json",
                accept="application/json",
            )
            
            # Parse response
            response_body = json.loads(response["body"].read().decode())
            
            # Track actual token usage if available in the response
            if "usage" in response_body:
                actual_tokens = response_body["usage"].get("input_tokens", 0) + response_body["usage"].get("output_tokens", 0)
                # Update with actual usage (subtracting the estimate first since we already counted it)
                await token_limiter.add_usage(actual_tokens - estimated_total_tokens)
                logger.info(f"API call used {actual_tokens} tokens (input: {response_body['usage'].get('input_tokens', 0)}, output: {response_body['usage'].get('output_tokens', 0)})")
            
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
        
        except Exception as e:
            # Check if it's a throttling exception
            if "ThrottlingException" in str(e) or "Too many requests" in str(e) or "Rate exceeded" in str(e):
                if retry < max_retries:
                    # Calculate exponential backoff delay: 2^retry * base_delay (with jitter)
                    delay = min(60, (2 ** retry) * base_delay * (0.5 + 0.5 * (time.time() % 1)))
                    logger.warning(f"Request throttled. Retrying in {delay:.2f}s (attempt {retry+1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries reached for throttling exception: {e}")
                    raise
            else:
                logger.error(f"Error calling Bedrock API: {e}")
                raise 