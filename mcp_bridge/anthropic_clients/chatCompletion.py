from typing import Dict, Any, List, Optional
import json
from loguru import logger

from .genericClient import client
from .utils import anthropic_get_tools, call_tool


async def anthropic_chat_completions(
    messages: List[Dict[str, Any]],
    model: str = "claude-3-7-sonnet-20250219",
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Perform chat completion using the Anthropic API with MCP tools.
    """
    # Check if client is available
    if client is None:
        logger.error("Anthropic client not initialized")
        return {
            "id": "error",
            "model": model,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Error: Anthropic client not initialized. Check API key in config.json.",
                },
                "finish_reason": "error"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    # Fetch tools from MCP clients and convert to Anthropic format
    tools = await anthropic_get_tools()
    
    # Configure parameters for the request
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    
    # Only add tools parameter if we have tools available
    if tools:
        params["tools"] = tools
        params["betas"] = ["token-efficient-tools-2025-02-19"]
    
    # Add optional parameters if provided
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    
    try:
        # Initial request to Anthropic
        logger.info(f"Calling Anthropic API with {len(tools)} tools")
        if tools:
            logger.info(f"Tool names: {[t['name'] for t in tools]}")
        response = client.beta.messages.create(**params)
        
        # Process tool calls if any
        while hasattr(response, "stop_reason") and response.stop_reason == "tool_use":
            logger.info("Tool use detected in Anthropic response")
            
            if not hasattr(response, "content") or not response.content:
                logger.error("No content in tool use response")
                break
            
            # Process each tool call
            for content_item in response.content:
                if not hasattr(content_item, "type") or content_item.type != "tool_use":
                    continue
                
                logger.info(f"Processing tool call: {content_item.name}")
                
                tool_name = content_item.name
                tool_input = content_item.input
                tool_id = content_item.id
                
                # Call the tool using the utility function
                tool_result = await call_tool(tool_name, tool_input)
                
                if tool_result is None:
                    # Add error response for this tool call
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
                        ]
                    })
                    messages.append({
                        "role": "user", 
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": f"Error: Tool {tool_name} call failed."
                            }
                        ]
                    })
                    continue
                
                # Extract the text content from the tool result
                tool_content = ""
                if hasattr(tool_result, "content"):
                    for part in tool_result.content:
                        if hasattr(part, "text") and part.text:
                            tool_content += part.text
                
                if not tool_content:
                    tool_content = "The tool call result is empty"
                
                # Add the tool result to messages
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
                    ]
                })
                
                # Use the correct Anthropic tool result format
                messages.append({
                    "role": "user", 
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": tool_content
                        }
                    ]
                })
            
            # Make a follow-up request with the updated messages
            params["messages"] = messages
            try:
                response = client.beta.messages.create(**params)
            except Exception as e:
                logger.error(f"Error from Anthropic API during tool processing: {e}")
                return {
                    "id": "error",
                    "model": model,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": f"Error from Anthropic API during tool processing: {str(e)}",
                        },
                        "finish_reason": "error"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
        
        # Format the response to match expected structure
        try:
            content_text = ""
            if hasattr(response, "content"):
                content_text = "".join([item.text for item in response.content if hasattr(item, "text")])
            
            # Structure the response to match OpenAI format for compatibility
            formatted_response = {
                "id": response.id,
                "model": response.model,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content_text,
                    },
                    "finish_reason": response.stop_reason
                }],
                "usage": {
                    "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                    "completion_tokens": getattr(response.usage, "output_tokens", 0),
                    "total_tokens": getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)
                }
            }
            
            # Add the original content for tool processing
            if hasattr(response, "content"):
                formatted_response["content"] = response.content
                
            return formatted_response
        except Exception as e:
            logger.error(f"Error formatting Anthropic response: {e}")
            return {
                "id": getattr(response, "id", "error"),
                "model": model,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Error formatting Anthropic response",
                    },
                    "finish_reason": "error"
                }],
                "usage": {
                    "prompt_tokens": 0, 
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
    except Exception as e:
        logger.error(f"Error from Anthropic API: {e}")
        return {
            "id": "error",
            "model": model,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"Error from Anthropic API: {str(e)}",
                },
                "finish_reason": "error"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        } 