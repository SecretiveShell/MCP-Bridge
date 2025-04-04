#!/usr/bin/env python
"""Anthropic/Claude API handler for the agent worker"""

import json
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

from mcp_bridge.anthropic_clients.chatCompletion import anthropic_chat_completions
from mcp_bridge.anthropic_clients.utils import call_tool
from mcp_bridge.agent_worker.utils import (
    extract_tool_result_text,
    extract_tool_result_image,
    is_image_tool,
    is_task_complete
)
from lmos_openai_types import (
    ChatCompletionRequestMessage,
)


async def process_with_anthropic(
    messages: List[ChatCompletionRequestMessage],
    model: str,
    system_prompt: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage], bool]:
    """Process a single iteration with Anthropic API
    
    Args:
        messages: The current conversation history
        model: The model name to use
        system_prompt: Optional system prompt
        
    Returns:
        Tuple containing:
            - the updated anthropic_messages
            - the updated conversation history
            - a boolean indicating if task is complete
    """
    # Convert messages to Anthropic format
    anthropic_messages = _convert_messages_to_anthropic_format(messages)
    updated_messages = messages.copy()
    
    # Create system prompt with caching for Anthropic
    formatted_system_prompt = _format_system_prompt(system_prompt)
    
    # Call Anthropic API
    logger.info("Using Anthropic API")
    response = await anthropic_chat_completions(
        messages=anthropic_messages,
        model=model,
        system=formatted_system_prompt
    )
    
    if not response or "choices" not in response or not response["choices"]:
        logger.warning("No choices in Anthropic response")
        return anthropic_messages, updated_messages, False
    
    choice = response["choices"][0]
    message_data = choice["message"]
    finish_reason = choice["finish_reason"]
    
    # Get original content if available
    content_items = response.get("content", [])
    
    # Process any tool calls
    tool_calls = _extract_tool_calls(content_items)
    
    if tool_calls and finish_reason == "tool_use":
        logger.info("Anthropic model wants to use tools")
        print(f"\nAssistant wants to use tools:")
        
        # Process tool calls and update messages
        anthropic_messages, updated_messages = await _process_tool_calls(
            tool_calls, anthropic_messages, updated_messages
        )
        
        # Tool calls were processed, return to continue the loop
        return anthropic_messages, updated_messages, False
        
    # Handle regular text response
    return await _process_text_response(
        message_data, anthropic_messages, updated_messages
    )


def _convert_messages_to_anthropic_format(
    messages: List[ChatCompletionRequestMessage]
) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages to Anthropic format"""
    anthropic_messages = []
    
    for i, msg in enumerate(messages):
        try:
            # Get the role safely
            msg_role = None
            msg_content = None
            
            if hasattr(msg, "root"):
                # Access through root object
                if hasattr(msg.root, "role"):
                    msg_role = msg.root.role
                if hasattr(msg.root, "content"):
                    msg_content = msg.root.content
            elif hasattr(msg, "role"):
                # Direct access
                msg_role = msg.role
                if hasattr(msg, "content"):
                    msg_content = msg.content
            
            # Handle based on safely extracted values
            if msg_role == "system":
                # Skip system messages, will use system parameter instead
                continue
            elif msg_role == "user":
                # Handle user messages without prepending system prompt
                anthropic_messages.append({
                    "role": "user",
                    "content": msg_content or ""
                })
            else:
                # Convert other message formats
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    # Handle tool calls in assistant messages
                    tool_call_content = []
                    for tool_call in msg.tool_calls:
                        if tool_call.type == "function":
                            function = tool_call.function
                            tool_call_content.append({
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": function.name,
                                "input": json.loads(function.arguments)
                            })
                    if tool_call_content:
                        anthropic_messages.append({
                            "role": "assistant",
                            "content": tool_call_content
                        })
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    # Handle tool response messages with proper format for Anthropic
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg_content or ""
                            }
                        ]
                    })
                elif msg_role:
                    # Handle regular text messages
                    anthropic_messages.append({
                        "role": msg_role,
                        "content": msg_content or ""
                    })
        except Exception as e:
            logger.warning(f"Error processing message {i}: {e}")
            # Fall back to adding a simple user message with error info
            if i > 0:  # Skip appending for first messages to avoid duplicating system messages
                anthropic_messages.append({
                    "role": "user",
                    "content": f"Error processing previous message: {str(e)}. Please continue."
                })
    
    return anthropic_messages


def _format_system_prompt(system_prompt: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Format the system prompt for Anthropic"""
    if system_prompt:
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    return None


def _extract_tool_calls(content_items: List[Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from content items"""
    tool_calls = []
    for item in content_items:
        if isinstance(item, dict) and item.get("type") == "tool_use":
            tool_calls.append(item)
        elif hasattr(item, "type") and getattr(item, "type") == "tool_use":
            tool_calls.append(item)
    return tool_calls


async def _process_tool_calls(
    tool_calls: List[Dict[str, Any]],
    anthropic_messages: List[Dict[str, Any]],
    updated_messages: List[ChatCompletionRequestMessage]
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage]]:
    """Process all tool calls and update messages"""
    # Import ClientManager inside the function only when needed
    from mcp_bridge.mcp_clients.McpClientManager import ClientManager
    
    # Process each tool call
    for tool_call in tool_calls:
        try:
            # Get tool properties safely
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "")
                tool_input = tool_call.get("input", {})
                tool_id = tool_call.get("id", "")
            else:
                tool_name = getattr(tool_call, "name", "")
                tool_input = getattr(tool_call, "input", {})
                tool_id = getattr(tool_call, "id", "")
            
            # Sanitize tool name
            tool_name = _sanitize_tool_name(tool_name)
            
            logger.info(f"Calling tool: {tool_name}")
            print(f"\nTool Call: {tool_name}")
            print(f"Arguments: {json.dumps(tool_input, indent=2)}")
            
            # Handle the tool call and update messages
            anthropic_messages, updated_messages = await _handle_tool_call(
                tool_name, tool_input, tool_id, anthropic_messages, updated_messages
            )
            
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            # Add an error message instead of a tool call
            error_message = ChatCompletionRequestMessage(
                role="user",
                content=f"There was an error processing the tool call: {str(e)}. Please try a different approach."
            )
            updated_messages.append(error_message)
            anthropic_messages.append({
                "role": "user",
                "content": f"There was an error processing the tool call: {str(e)}. Please try a different approach."
            })
    
    return anthropic_messages, updated_messages


def _sanitize_tool_name(tool_name: str) -> str:
    """Sanitize tool name to remove invalid characters"""
    if "." in tool_name or not all(c.isalnum() or c in "_-" for c in tool_name):
        logger.warning(f"Sanitizing tool name: {tool_name}")
        if "." in tool_name:
            tool_name = tool_name.split(".")[-1]
        tool_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in tool_name)
    return tool_name


async def _handle_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_id: str,
    anthropic_messages: List[Dict[str, Any]],
    updated_messages: List[ChatCompletionRequestMessage]
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage]]:
    """Handle a single tool call and update messages"""
    # Call the tool using the utility
    result = await call_tool(tool_name, tool_input)
    
    if result is None:
        return await _handle_failed_tool_call(
            tool_name, tool_input, tool_id, anthropic_messages, updated_messages
        )
    
    # Check if it's an image tool
    is_tool_image = await is_image_tool(tool_name)
    
    result_text = await extract_tool_result_text(result)
    print(f"Tool Result: {result_text}")
    
    # Add assistant message with tool call
    tool_call_msg = ChatCompletionRequestMessage(
        role="assistant",
        content=None,
        tool_calls=[{
            "id": tool_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_input)
            }
        }]
    )
    updated_messages.append(tool_call_msg)
    
    # Add tool result message
    tool_result_message = ChatCompletionRequestMessage(
        role="tool",
        content=result_text,
        tool_call_id=tool_id
    )
    updated_messages.append(tool_result_message)
    
    # Also update anthropic_messages for continuity
    anthropic_messages.append({
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
        ]
    })
    
    # Process tool result based on type (image or text)
    return await _process_tool_result(
        tool_name, tool_id, result, result_text, is_tool_image, 
        anthropic_messages, updated_messages
    )


async def _handle_failed_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_id: str,
    anthropic_messages: List[Dict[str, Any]],
    updated_messages: List[ChatCompletionRequestMessage]
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage]]:
    """Handle a tool call that failed"""
    error_message = f"Tool {tool_name} not found or call failed"
    logger.error(error_message)
    
    # Create a placeholder tool call
    tool_call_msg = ChatCompletionRequestMessage(
        role="assistant",
        content=None,
        tool_calls=[{
            "id": tool_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_input)
            }
        }]
    )
    updated_messages.append(tool_call_msg)
    
    # Add error message as tool result
    tool_result_message = ChatCompletionRequestMessage(
        role="tool",
        content=error_message,
        tool_call_id=tool_id
    )
    updated_messages.append(tool_result_message)
    
    # Update anthropic_messages too
    anthropic_messages.append({
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
        ]
    })
    anthropic_messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": error_message
            }
        ]
    })
    
    return anthropic_messages, updated_messages


async def _process_tool_result(
    tool_name: str,
    tool_id: str,
    result: Any,
    result_text: str,
    is_tool_image: bool,
    anthropic_messages: List[Dict[str, Any]],
    updated_messages: List[ChatCompletionRequestMessage]
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage]]:
    """Process tool result based on type (image or text)"""
    # Check if this is an image tool and extract image if available
    if is_tool_image:
        image_content = await extract_tool_result_image(result)
        if image_content:
            # If we have image data, add it to the user message with tool result
            anthropic_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text or "Here is the screenshot:"
                    },
                    image_content
                ]
            })
            logger.info(f"Added image from {tool_name} to message")
        else:
            # Fall back to text-only if no image was extracted
            anthropic_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text
                    }
                ]
            })
            logger.warning(f"No image found for {tool_name}, using text-only response")
    else:
        # Regular text-only tool result
        anthropic_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text
                }
            ]
        })
    
    return anthropic_messages, updated_messages


async def _process_text_response(
    message_data: Dict[str, Any],
    anthropic_messages: List[Dict[str, Any]],
    updated_messages: List[ChatCompletionRequestMessage]
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage], bool]:
    """Process a regular text response from the model"""
    message_content = message_data["content"]
    
    # Add assistant message to conversation
    assistant_message = ChatCompletionRequestMessage(
        role="assistant",
        content=message_content
    )
    updated_messages.append(assistant_message)
    print(f"\nAssistant: {message_content}")
    
    # Check for task completion
    is_complete = is_task_complete(message_content)
    if is_complete:
        logger.info("Task completed successfully.")
    
    return anthropic_messages, updated_messages, is_complete 