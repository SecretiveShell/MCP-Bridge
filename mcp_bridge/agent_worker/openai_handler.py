#!/usr/bin/env python
"""OpenAI API handler for the agent worker"""

import json
from typing import Dict, Any, List, Tuple
from loguru import logger

from mcp_bridge.openai_clients.chatCompletion import chat_completions
from mcp_bridge.agent_worker.utils import (
    extract_tool_result_text,
    extract_tool_result_image,
    is_image_tool,
    is_task_complete
)
from lmos_openai_types import (
    CreateChatCompletionRequest,
    ChatCompletionRequestMessage,
)


async def process_with_openai(
    messages: List[ChatCompletionRequestMessage],
    model: str
) -> Tuple[List[ChatCompletionRequestMessage], bool]:
    """Process a single iteration with OpenAI API
    
    Args:
        messages: The current conversation history
        model: The model name to use
        
    Returns:
        Tuple containing:
            - the updated conversation history
            - a boolean indicating if task is complete
    """
    updated_messages = messages.copy()
    
    # Process with MCP OpenAI-compatible API
    request = CreateChatCompletionRequest(
        model=model,
        messages=messages
    )
    
    response = await chat_completions(request)
    
    if response and response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        message = choice.message
        
        # Add assistant message to conversation
        assistant_message = ChatCompletionRequestMessage(
            role="assistant",
            content=message.content
        )
        
        # Process tool calls if present
        if message.tool_calls:
            assistant_message = ChatCompletionRequestMessage(
                role="assistant",
                content=None,
                tool_calls=message.tool_calls
            )
            updated_messages.append(assistant_message)
            
            # Print the message content if available
            print(f"\nAssistant wants to use tools:")
            
            for tool_call in message.tool_calls:
                if tool_call.type == "function":
                    function = tool_call.function
                    tool_name = function.name
                    
                    # Sanitize tool name to ensure it only contains allowed characters
                    if "." in tool_name or not all(c.isalnum() or c in "_-" for c in tool_name):
                        logger.warning(f"Sanitizing tool name: {tool_name}")
                        # For tools with periods (like mcp_mcp_my_apple_remembers_my_apple_recall_memory),
                        # use the last part after the final period
                        if "." in tool_name:
                            tool_name = tool_name.split(".")[-1]
                        # Replace any remaining invalid characters with underscores
                        tool_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in tool_name)
                    
                    tool_args = json.loads(function.arguments)
                    tool_id = tool_call.id
                    
                    logger.info(f"Calling tool: {tool_name}")
                    print(f"\nTool Call: {tool_name}")
                    print(f"Arguments: {json.dumps(tool_args, indent=2)}")
                    
                    # Import ClientManager here to avoid circular imports
                    from mcp_bridge.mcp_clients.McpClientManager import ClientManager
                    session = await ClientManager.get_client_from_tool(tool_name)
                    
                    if session:
                        try:
                            # Call the tool
                            result = await session.function_call(tool_name, tool_args)
                            result_text = await extract_tool_result_text(result)
                            print(f"Tool Result: {result_text}")
                            
                            # Add tool result message
                            tool_result_message = ChatCompletionRequestMessage(
                                role="tool",
                                content=result_text,
                                tool_call_id=tool_id
                            )
                            updated_messages.append(tool_result_message)
                            
                            # If this is an image tool, store the image data in a way that can be
                            # retrieved later by the anthropic formatter if needed
                            if await is_image_tool(tool_name):
                                image_content = await extract_tool_result_image(result)
                                if image_content:
                                    # Store the image data in the message's metadata for later use
                                    if not hasattr(tool_result_message, "metadata"):
                                        tool_result_message.metadata = {}
                                    tool_result_message.metadata["image_content"] = image_content
                                    logger.info(f"Stored image from {tool_name} in message metadata")
                        except Exception as e:
                            error_message = f"Error executing tool {tool_name}: {str(e)}"
                            logger.error(error_message)
                            
                            # Add error message to conversation
                            tool_result_message = ChatCompletionRequestMessage(
                                role="tool",
                                content=error_message,
                                tool_call_id=tool_id
                            )
                            updated_messages.append(tool_result_message)
                    else:
                        error_message = f"Tool {tool_name} not found or not available"
                        logger.error(error_message)
                        
                        # Add error message to conversation
                        tool_result_message = ChatCompletionRequestMessage(
                            role="tool",
                            content=error_message,
                            tool_call_id=tool_id
                        )
                        updated_messages.append(tool_result_message)
        else:
            if message.content:
                updated_messages.append(assistant_message)
                print(f"\nAssistant: {message.content}")
                
                # Check for task completion
                if is_task_complete(message.content):
                    logger.info("Task completed successfully.")
                    return updated_messages, True  # Task is complete
    else:
        logger.warning("No choices in response")
    
    return updated_messages, False  # Task is not complete 