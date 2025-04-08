from typing import Dict, Any, List, Optional, Union
import json
from loguru import logger

from .genericClient import client, create_messages
from .utils import anthropic_get_tools, call_tool


def _build_error_response(model: str, error_message: str) -> Dict[str, Any]:
    """Build a standardized error response"""
    return {
        "id": "error",
        "model": model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": error_message,
            },
            "finish_reason": "error"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def _build_request_params(
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    system: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]],
    tools: List[Dict[str, Any]],
    budget_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """Build the parameters for Anthropic API request"""
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    
    # Add system prompt if provided
    if system:
        params["system"] = system
    
    # Only add tools parameter if we have tools available
    if tools:
        params["tools"] = tools
    
    # Add thinking configuration if budget_tokens is provided
    if budget_tokens is not None:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens
        }
    
    # Add optional parameters if provided
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
        
    return params


def _process_tool_result(tool_name: str, tool_result: Any) -> tuple[str, Optional[Dict[str, Any]]]:
    """Process a tool result and extract text content and image data"""
    tool_content = ""
    image_data = None
    
    if hasattr(tool_result, "content"):
        for part in tool_result.content:
            part_type_name = type(part).__name__
            
            # Handle text content
            if hasattr(part, "text") and part.text:
                tool_content += part.text
            # Handle ImageContent objects (main approach for MCP image tools)
            elif part_type_name == "ImageContent" or (hasattr(part, "type") and getattr(part, "type") == "image"):
                # Extract image data if available
                try:
                    # Look for data attribute which contains the base64 image
                    if hasattr(part, "data"):
                        # Get the image data and MIME type
                        image_data_raw = part.data
                        mime_type = "image/png"  # Default
                        if hasattr(part, "mimeType"):
                            mime_type = part.mimeType

                        image_data = {
                            "type": "image",
                            "source": {
                                "type": "base64", 
                                "media_type": mime_type,
                                "data": image_data_raw
                            }
                        }
                        logger.info(f"Found image in {tool_name} result")
                except Exception as e:
                    logger.error(f"Error extracting image: {e}")
            # Legacy approach for image attribute
            elif hasattr(part, "image") and part.image:
                # Extract image data if available
                try:
                    image_data = {
                        "type": "image",
                        "source": {
                            "type": "base64", 
                            "media_type": "image/png",
                            "data": part.image
                        }
                    }
                    logger.info(f"Found image in {tool_name} result")
                except Exception as e:
                    logger.error(f"Error extracting image: {e}")
            # Check if part might be a dictionary
            elif isinstance(part, dict):
                # Check for image or data keys in dictionary
                for key in ["image", "data"]:
                    if hasattr(part, 'keys') and key in part and isinstance(part[key], str) and len(part[key]) > 1000:
                        try:
                            image_data = {
                                "type": "image",
                                "source": {
                                    "type": "base64", 
                                    "media_type": "image/png",
                                    "data": part[key]
                                }
                            }
                            logger.info(f"Found image in {tool_name} dictionary result")
                            break  # Found image data, no need to check other keys
                        except Exception as e:
                            logger.error(f"Error extracting image from dictionary: {e}")
    
    if not tool_content:
        tool_content = "The tool call result is empty"
        
    return tool_content, image_data


def _add_tool_messages(
    messages: List[Dict[str, Any]], 
    tool_name: str, 
    tool_input: Any, 
    tool_id: str, 
    tool_content: str,
    image_data: Optional[Dict[str, Any]]
) -> None:
    """Add tool call and tool result messages to the message list"""
    # Add the tool call message
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
        ]
    })
    
    # Add the tool result message with image if available
    if image_data:
        messages.append({
            "role": "user", 
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": tool_content
                },
                image_data
            ]
        })
        logger.info(f"Added image to message for {tool_name}")
    else:
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


def _format_final_response(response: Any, model: str, thinking_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format the Anthropic response to match expected structure"""
    content_text = ""
    thinking_blocks = []
    
    if hasattr(response, "content"):
        # Extract thinking blocks and text content
        for item in response.content:
            if hasattr(item, "type"):
                if item.type in ["thinking", "redacted_thinking"]:
                    thinking_blocks.append(item)
                elif item.type == "text" and hasattr(item, "text"):
                    content_text += item.text
            elif hasattr(item, "text"):
                content_text += item.text
    
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
        
    # Add thinking blocks if available
    if thinking_blocks:
        formatted_response["thinking_blocks"] = thinking_blocks
    
    if 'thinking_blocks' in formatted_response:
        logger.info(f"Returning response with {len(formatted_response['thinking_blocks'])} thinking blocks.")
    else:
        logger.info("No thinking blocks in response")

    return formatted_response


async def _process_tool_calls(response: Any, messages: List[Dict[str, Any]], params: Dict[str, Any], model: str, thinking_blocks: List[Dict[str, Any]], customer_logger: Optional[Any] = None) -> Any:
    """Process tool calls in the response and make follow-up requests"""
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

            if customer_logger:
                customer_logger.log_tool_call(content_item.name, content_item.input, content_item.id)
            
            # Call the tool using the utility function
            tool_result = await call_tool(tool_name, tool_input)
            
            if tool_result is None:
                # Add error response for this tool call
                _add_tool_messages(
                    messages, 
                    tool_name, 
                    tool_input, 
                    tool_id, 
                    f"Error: Tool {tool_name} call failed.",
                    None
                )
                continue
            
            # Process tool result to extract text and images
            tool_content, image_data = _process_tool_result(tool_name, tool_result)

            if customer_logger:
                customer_logger.log_tool_result(tool_id, tool_content, image_data is not None)
            
            # Add the tool messages to the conversation
            _add_tool_messages(messages, tool_name, tool_input, tool_id, tool_content, image_data)

                # Update messages with new thinking blocks for tool processing
        if thinking_blocks:
            messages = _insert_thinking_into_messages(messages, thinking_blocks)
            logger.info(f"Added {len(thinking_blocks)} thinking blocks to tool call messages.")
            params["messages"] = messages

            if customer_logger:
                customer_logger.log_thinking(_format_thinking_blocks(thinking_blocks))
    
        response = await create_messages(**params)

        if customer_logger:
            customer_logger.log_system_event("anthropic_tool_call_response", {
                "response": json.dumps(response, default=str)
            })
        
        total_new_thinking_blocks = 0
        thinking_blocks = []
        if hasattr(response, "content"):
            for item in response.content:
                if hasattr(item, "type") and item.type in ["thinking", "redacted_thinking"]:
                    thinking_blocks.append(item)
                    total_new_thinking_blocks += 1
        logger.info(f"Added {total_new_thinking_blocks} thinking blocks from tool use response to the tool use messages.")
    
    return response


def _format_thinking_blocks(thinking_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format thinking blocks to ensure they're properly structured.
    
    Args:
        thinking_blocks: Raw thinking blocks
        
    Returns:
        Properly formatted thinking blocks ready to be added to content
    """
    if not thinking_blocks:
        return []
        
    # Ensure thinking blocks are properly formatted
    formatted_blocks = []
    for block in thinking_blocks:
        # Check if the block is already properly formatted
        formatted_blocks.append({"type": "thinking", "thinking": block.thinking, "signature": block.signature})
            
    return formatted_blocks


def _add_thinking_to_message_content(message: Dict[str, Any], thinking_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add thinking blocks to a message's content.
    
    Args:
        message: Message to add thinking blocks to
        thinking_blocks: Thinking blocks to add
        
    Returns:
        Updated message with thinking blocks added to content
    """
    if not thinking_blocks:
        return message.copy()
        
    # Create a new message with the same role
    updated_message = {"role": message.get("role", "assistant")}
    
    # Initialize content based on original message
    if "content" not in message:
        # Create new content with just thinking blocks
        updated_message["content"] = thinking_blocks
    elif isinstance(message["content"], list):
        # Add thinking blocks to the beginning of existing content list
        updated_message["content"] = thinking_blocks + message["content"]
    else:
        # Convert string content to a content list with thinking blocks at the beginning
        # followed by the original content as a text block
        updated_message["content"] = thinking_blocks + [{"type": "text", "text": message["content"]}]
    
    return updated_message


def _insert_thinking_into_messages(
    messages: List[Dict[str, Any]], 
    thinking_blocks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Insert thinking blocks into a messages list.
    This function decides where to put the thinking blocks.
    
    Args:
        messages: Original messages list
        thinking_blocks: Thinking blocks to insert
        
    Returns:
        Updated messages list with thinking blocks inserted
    """
    if not thinking_blocks:
        return messages.copy()
    
    formatted_blocks = _format_thinking_blocks(thinking_blocks)
    if not formatted_blocks:
        return messages.copy()
        
    updated_messages = messages.copy()
    
    # Find the last assistant message if it exists
    last_assistant_message = None
    last_assistant_message_index = None
    for i, message in enumerate(updated_messages):
        if message.get("role") == "assistant":
            last_assistant_message = message
            last_assistant_message_index = i
    
    if last_assistant_message:
        # Add thinking blocks to this existing assistant message
        updated_messages[last_assistant_message_index] = _add_thinking_to_message_content(last_assistant_message, formatted_blocks)
        return updated_messages
    
    # If no assistant message found, create one at the beginning
    new_assistant_message = {
        "role": "assistant",
        "content": formatted_blocks
    }
    
    # Insert new assistant message at the beginning
    return updated_messages + [new_assistant_message]


async def anthropic_chat_completions(
    messages: List[Dict[str, Any]],
    model: str = "claude-3-7-sonnet-20250219",
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    system: Optional[Union[str, List[Dict[str, Any]]]] = None,
    thinking_blocks: Optional[List[Dict[str, Any]]] = None,
    budget_tokens: Optional[int] = None,
    customer_logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Perform chat completion using the Anthropic API with MCP tools.
    
    Args:
        messages: List of message objects to send to the API
        model: Model ID to use
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0-1)
        top_p: Nucleus sampling parameter
        system: System prompt as string or list of content blocks with caching controls
        thinking_blocks: Previous thinking blocks from Claude to maintain thought continuity
        budget_tokens: Number of tokens to allocate for Claude's thinking process
    """
    # Check if client is None
    if client is None:
        logger.error("Anthropic client not initialized")
        return _build_error_response(model, "Error: Anthropic client not initialized. Check API key in config.json.")
    
    # Fetch tools from MCP clients and convert to Anthropic format
    tools = await anthropic_get_tools()

    # Add thinking blocks to messages if provided
    if thinking_blocks:
        messages = _insert_thinking_into_messages(messages, thinking_blocks)
        logger.info(f"Added {len(thinking_blocks)} thinking blocks to chat completion messages.")
    
    # Configure parameters for the request
    params = _build_request_params(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        system=system,
        tools=tools,
        budget_tokens=budget_tokens,
    )

    params["betas"] = ["computer-use-2025-01-24"]
    
    # Initial request to Anthropic or Bedrock
    logger.info(f"Calling Claude API with {len(tools)} tools")
    if tools:
        logger.info(f"Tool names: {[t['name'] for t in tools]}")
    if budget_tokens:
        logger.info(f"Using thinking with budget of {budget_tokens} tokens")
        
    response = await create_messages(**params)

    if customer_logger:
        customer_logger.log_system_event("anthropic_response", {
            "response": json.dumps(response, default=str)
        })


    # Extract thinking blocks from response.content if present
    total_new_thinking_blocks = 0
    if hasattr(response, "content"):
        for item in response.content:
            if hasattr(item, "type") and item.type in ["thinking", "redacted_thinking"]:
                thinking_blocks.append(item)
                total_new_thinking_blocks += 1
    logger.info(f"Added {total_new_thinking_blocks} thinking blocks from response to the tool use messages.")

    # Process tool calls if any
    response = await _process_tool_calls(response, messages, params, model, thinking_blocks, customer_logger)

    if customer_logger:
        customer_logger.log_system_event("anthropic_response", {
            "response": json.dumps(response, default=str)
        })
    
    total_new_thinking_blocks = 0
    if hasattr(response, "content"):
        for item in response.content:
            if hasattr(item, "type") and item.type in ["thinking", "redacted_thinking"]:
                thinking_blocks.append(item)
                total_new_thinking_blocks += 1
    logger.info(f"Added {total_new_thinking_blocks} thinking blocks from tool use response to the tool use messages.")


    # Format the response to match expected structure
    return _format_final_response(response, model, thinking_blocks)
