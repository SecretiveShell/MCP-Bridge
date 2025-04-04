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
    anthropic_messages = []
    updated_messages = messages.copy()
    
    # Create system prompt with caching for Anthropic
    formatted_system_prompt = None
    if system_prompt:
        formatted_system_prompt = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    
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
    
    # Call Anthropic API
    logger.info("Using Anthropic API")
    response = await anthropic_chat_completions(
        messages=anthropic_messages,
        model=model,
        system=formatted_system_prompt
    )
    
    if response and "choices" in response and len(response["choices"]) > 0:
        choice = response["choices"][0]
        message_data = choice["message"]
        finish_reason = choice["finish_reason"]
        
        # Get original content if available
        content_items = response.get("content", [])
        
        # Process any tool calls
        tool_calls = []
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                tool_calls.append(item)
            elif hasattr(item, "type") and getattr(item, "type") == "tool_use":
                tool_calls.append(item)
        
        if tool_calls and finish_reason == "tool_use":
            logger.info("Anthropic model wants to use tools")
            print(f"\nAssistant wants to use tools:")
            
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
                    if "." in tool_name or not all(c.isalnum() or c in "_-" for c in tool_name):
                        logger.warning(f"Sanitizing tool name: {tool_name}")
                        if "." in tool_name:
                            tool_name = tool_name.split(".")[-1]
                        tool_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in tool_name)
                    
                    logger.info(f"Calling tool: {tool_name}")
                    print(f"\nTool Call: {tool_name}")
                    print(f"Arguments: {json.dumps(tool_input, indent=2)}")
                    
                    # Call the tool using the utility
                    result = await call_tool(tool_name, tool_input)
                    
                    if result is None:
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
                        continue
                    
                    # Check if it's an image tool, but don't do anything special yet
                    # The actual image extraction will happen later
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
            
            # Tool calls were processed, return to continue the loop
            return anthropic_messages, updated_messages, False
            
        # Handle regular text response
        message_content = message_data["content"]
        
        # Add assistant message to conversation
        assistant_message = ChatCompletionRequestMessage(
            role="assistant",
            content=message_content
        )
        updated_messages.append(assistant_message)
        print(f"\nAssistant: {message_content}")
        
        # Check for task completion
        if is_task_complete(message_content):
            logger.info("Task completed successfully.")
            return anthropic_messages, updated_messages, True
    else:
        logger.warning("No choices in Anthropic response")
    
    return anthropic_messages, updated_messages, False 