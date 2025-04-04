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
    system_prompt: Optional[str] = None,
    thinking_blocks: Optional[List[Dict[str, Any]]] = None,
    customer_logger: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage], List[Dict[str, Any]], bool]:
    """Process a single iteration with Anthropic API
    
    Args:
        messages: The current conversation history
        model: The model name to use
        system_prompt: Optional system prompt
        thinking_blocks: Previous thinking blocks for context
        customer_logger: Optional customer message logger
        
    Returns:
        Tuple containing:
            - the updated anthropic_messages
            - the updated conversation history
            - the updated thinking blocks
            - a boolean indicating if task is complete
    """
    # Convert messages to Anthropic format
    anthropic_messages = _convert_messages_to_anthropic_format(messages)
    updated_messages = messages.copy()
    
    # Create system prompt with caching for Anthropic
    formatted_system_prompt = _format_system_prompt(system_prompt)
    
    # Call Anthropic API
    logger.info("Using Anthropic API")
    if customer_logger:
        customer_logger.log_system_event("api_call", {
            "provider": "anthropic",
            "model": model
        })
        
    response = await anthropic_chat_completions(
        messages=anthropic_messages,
        model=model,
        system=formatted_system_prompt,
        max_tokens=3000,
        budget_tokens=2048,
        thinking_blocks=thinking_blocks
    )
    
    # Process different types of responses (tool calls, text, etc.)
    if 'choices' in response:
        # Extract thinking blocks if available
        new_thinking_blocks = response.get('thinking_blocks', [])
        
        # Process based on stop reason
        finish_reason = response['choices'][0]['finish_reason']
        
        if finish_reason == 'tool_use' and 'content' in response:
            # Handle tool calls
            logger.info("Processing tool calls from response")
            if customer_logger:
                customer_logger.log_system_event("tool_use", {
                    "content_items": len(response['content'])
                })
                
            # Process the tool calls and get updated messages
            anthropic_messages, updated_messages, new_thinking_blocks, is_complete = await _process_tool_calls_response(
                response, 
                anthropic_messages, 
                updated_messages, 
                thinking_blocks,
                customer_logger
            )
            return anthropic_messages, updated_messages, new_thinking_blocks, is_complete
        else:
            # Handle regular text response
            logger.info("Processing text response")
            if customer_logger:
                message_content = response['choices'][0]['message']['content']
                customer_logger.log_message("assistant", message_content)
                
            return await _process_text_response(
                response['choices'][0]['message'], 
                anthropic_messages, 
                updated_messages, 
                new_thinking_blocks
            )
    else:
        # Handle error response
        logger.error("Invalid response format")
        if customer_logger:
            customer_logger.log_system_event("error", {
                "message": "Invalid response format from Anthropic API",
            })
        # Return original messages, no completion
        return anthropic_messages, updated_messages, [], False


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


async def _process_tool_calls_response(
    response: Dict[str, Any],
    anthropic_messages: List[Dict[str, Any]],
    updated_messages: List[ChatCompletionRequestMessage],
    thinking_blocks: List[Dict[str, Any]],
    customer_logger: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage], List[Dict[str, Any]], bool]:
    """Process a response that contains tool calls"""
    content_items = response.get('content', [])
    
    # Find tool use blocks in the content
    for item in content_items:
        if hasattr(item, 'type') and item.type == 'tool_use':
            tool_name = item.name
            tool_input = item.input
            tool_id = item.id
            
            logger.info(f"Processing tool call: {tool_name}")
            
            if customer_logger:
                customer_logger.log_tool_call(tool_name, tool_input, tool_id)
            
            # Create assistant message with tool call
            assistant_message = ChatCompletionRequestMessage(
                role="assistant",
                content=f"I'll use the {tool_name} tool to help with this."
            )
            updated_messages.append(assistant_message)
            
            # Call the tool
            tool_result = await call_tool(tool_name, tool_input)
            
            if tool_result is None:
                error_message = f"Error: Tool {tool_name} call failed."
                logger.error(error_message)
                
                if customer_logger:
                    customer_logger.log_tool_result(tool_id, error_message)
                
                # Add error message as tool result
                tool_result_message = ChatCompletionRequestMessage(
                    role="tool",
                    content=error_message,
                    tool_call_id=tool_id
                )
                updated_messages.append(tool_result_message)
                continue
            
            # Process tool result to handle text and images
            result_text = extract_tool_result_text(tool_result)
            
            # Check if this is an image-generating tool
            has_image = is_image_tool(tool_name) and extract_tool_result_image(tool_result) is not None
            
            if customer_logger:
                customer_logger.log_tool_result(tool_id, result_text, has_image)
            
            # Add tool result to messages
            tool_result_message = ChatCompletionRequestMessage(
                role="tool",
                content=result_text,
                tool_call_id=tool_id
            )
            updated_messages.append(tool_result_message)
    
    # Make a follow-up API call with the tool results
    follow_up_response = await _make_follow_up_api_call(
        anthropic_messages, 
        updated_messages, 
        thinking_blocks,
        response,
        customer_logger
    )
    
    # Process the follow-up response
    if 'choices' in follow_up_response:
        # Extract thinking blocks
        new_thinking_blocks = follow_up_response.get('thinking_blocks', [])
        
        # Check for more tool calls
        finish_reason = follow_up_response['choices'][0]['finish_reason']
        if finish_reason == 'tool_use' and 'content' in follow_up_response:
            # Recursively process more tool calls
            return await _process_tool_calls_response(
                follow_up_response, 
                anthropic_messages, 
                updated_messages, 
                new_thinking_blocks,
                customer_logger
            )
        else:
            # Process as a text response
            return await _process_text_response(
                follow_up_response['choices'][0]['message'], 
                anthropic_messages, 
                updated_messages, 
                new_thinking_blocks
            )
    else:
        # Handle error
        logger.error("Invalid follow-up response format")
        if customer_logger:
            customer_logger.log_system_event("error", {
                "message": "Invalid follow-up response format from Anthropic API",
            })
        return anthropic_messages, updated_messages, thinking_blocks, False


async def _make_follow_up_api_call(
    anthropic_messages: List[Dict[str, Any]],
    updated_messages: List[ChatCompletionRequestMessage],
    thinking_blocks: List[Dict[str, Any]],
    previous_response: Dict[str, Any],
    customer_logger: Optional[Any] = None
) -> Dict[str, Any]:
    """Make a follow-up API call after processing tool results"""
    # Convert updated messages back to Anthropic format
    follow_up_messages = _convert_messages_to_anthropic_format(updated_messages)
    
    # Get the model from the previous response
    model = previous_response.get('model', 'claude-3-7-sonnet-20250219')
    
    if customer_logger:
        customer_logger.log_system_event("follow_up_api_call", {
            "provider": "anthropic",
            "model": model
        })
    
    # Call Anthropic API with the updated messages including tool results
    return await anthropic_chat_completions(
        messages=follow_up_messages,
        model=model,
        max_tokens=3000,
        budget_tokens=2048,
        thinking_blocks=thinking_blocks
    )


async def _process_text_response(
    message_data: Dict[str, Any],
    anthropic_messages: List[Dict[str, Any]],
    updated_messages: List[ChatCompletionRequestMessage],
    thinking_blocks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[ChatCompletionRequestMessage], List[Dict[str, Any]], bool]:
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
    
    return anthropic_messages, updated_messages, thinking_blocks, is_complete 