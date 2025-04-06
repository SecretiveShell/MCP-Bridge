#!/usr/bin/env python
"""OpenAI API handler for the agent worker"""

import json
from typing import Dict, List, Tuple, Optional

from loguru import logger

from lmos_openai_types import (
    CreateChatCompletionRequest,
    ChatCompletionRequestMessage,
    ChatCompletionResponseMessage
)

from mcp_bridge.openai_clients.chatCompletion import chat_completions
from mcp_bridge.agent_worker.utils import (
    extract_tool_result_text,
    extract_tool_result_image,
    is_image_tool,
    is_task_complete
)


async def process_with_openai(
    messages: List[ChatCompletionRequestMessage],
    model: str,
    customer_logger: Optional["CustomerMessageLogger"] = None
) -> Tuple[List[ChatCompletionRequestMessage], bool]:
    """Process a single iteration with OpenAI API
    
    Args:
        messages: The current conversation history
        model: The model name to use
        customer_logger: Optional customer message logger
        
    Returns:
        Tuple containing:
            - the updated conversation history
            - a boolean indicating if task is complete
    """
    updated_messages = messages.copy()
    
    # Process with MCP OpenAI-compatible API
    if customer_logger:
        customer_logger.log_system_event("api_call", {
            "provider": "openai",
            "model": model
        })
        
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
            if customer_logger:
                customer_logger.log_system_event("tool_use", {
                    "tool_calls": len(message.tool_calls)
                })
                
            assistant_message = ChatCompletionRequestMessage(
                role="assistant",
                content=None,
                tool_calls=message.tool_calls
            )
            updated_messages.append(assistant_message)
            
            # Print the message content if available
            print(f"\nAssistant wants to use tools:")
            
            for tool_call in message.tool_calls:
                tool_id = tool_call.id
                tool_type = tool_call.type
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                
                # Attempt to parse JSON arguments
                try:
                    tool_args_dict = json.loads(tool_args)
                except Exception:
                    tool_args_dict = {"raw_args": tool_args}
                
                if customer_logger:
                    customer_logger.log_tool_call(tool_name, tool_args_dict, tool_id)
                    
                # Call the tool
                logger.info(f"Calling tool: {tool_name}")
                print(f"Tool: {tool_name}")
                print(f"Arguments: {tool_args}")
                
                # Import here to avoid circular imports
                from mcp_bridge.openai_clients.utils import call_tool
                
                tool_result = await call_tool(tool_name, tool_args)
                
                if tool_result:
                    # Extract content from result
                    tool_result_text = await extract_tool_result_text(tool_result)
                    
                    # Check if this is an image tool
                    has_image = await is_image_tool(tool_name) and await extract_tool_result_image(tool_result) is not None
                    
                    if customer_logger:
                        customer_logger.log_tool_result(tool_id, tool_result_text, has_image)
                        
                    print(f"Tool result: {tool_result_text}")
                    
                    # Add tool result to conversation
                    tool_result_message = ChatCompletionRequestMessage(
                        role="tool",
                        content=tool_result_text,
                        tool_call_id=tool_id
                    )
                    updated_messages.append(tool_result_message)
                else:
                    if customer_logger:
                        error_message = f"Error executing tool {tool_name}"
                        customer_logger.log_tool_result(tool_id, error_message)
                        
                    try:
                        from mcp_bridge.mcp_clients.McpClientManager import ClientManager
                        clients = ClientManager.get_clients()
                        client_names = [name for name, client in clients]
                        logger.error(f"Tool {tool_name} not found. Available clients: {client_names}")
                    except Exception as e:
                        logger.error(f"Error accessing ClientManager: {str(e)}")
                        
                    try:
                        error_message = f"Error executing tool {tool_name}: Tool not found or unavailable"
                        logger.error(error_message)
                        
                        # Add error message to conversation
                        tool_result_message = ChatCompletionRequestMessage(
                            role="tool",
                            content=error_message,
                            tool_call_id=tool_id
                        )
                        updated_messages.append(tool_result_message)
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
            if message.content:
                if customer_logger:
                    customer_logger.log_message("assistant", message.content)
                    
                updated_messages.append(assistant_message)
                print(f"\nAssistant: {message.content}")
                
                # Check for task completion
                if is_task_complete(message.content):
                    logger.info("Task completed successfully.")
                    if customer_logger:
                        customer_logger.log_system_event("task_complete", {
                            "final_message": message.content[:100] + "..." if len(message.content) > 100 else message.content
                        })
                    return updated_messages, True  # Task is complete
    else:
        logger.warning("No choices in response")
        if customer_logger:
            customer_logger.log_system_event("error", {
                "message": "No choices in OpenAI API response"
            })
    
    return updated_messages, False  # Task is not complete 