#!/usr/bin/env python
"""Agent worker module that provides standalone command-line agent execution"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from mcp_bridge.openai_clients.chatCompletion import chat_completions
from mcp_bridge.anthropic_clients.chatCompletion import anthropic_chat_completions
from mcp_bridge.agent_worker.utils import (
    extract_tool_result_text,
    extract_tool_result_image,
    is_image_tool,
    is_anthropic_model,
    is_task_complete
)
from lmos_openai_types import (
    CreateChatCompletionRequest,
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
)

class AgentWorker:
    """A standalone worker that processes tasks using MCP clients and LLM completions"""
    
    def __init__(
        self, 
        task: str, 
        model: str = "anthropic.claude-3-haiku-20240307-v1:0", 
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
    ):
        self.task = task
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant that completes tasks using available tools. Use the tools provided to you to help complete the user's task."
        self.messages: List[ChatCompletionRequestMessage] = []
        self.max_iterations = max_iterations
        
    async def initialize(self):
        """Initialize the MCP clients"""
        logger.info("Initializing MCP clients...")
        # Import here to avoid circular imports
        from mcp_bridge.mcp_clients.McpClientManager import ClientManager
        # Start the ClientManager to load all available MCP clients
        await ClientManager.initialize()
        
        # Wait a moment for clients to start up
        logger.info("Waiting for MCP clients to initialize...")
        await asyncio.sleep(2)
        
        # Check that at least one client is ready
        max_attempts = 3
        for attempt in range(max_attempts):
            clients = ClientManager.get_clients()
            ready_clients = [name for name, client in clients if client.session is not None]
            
            if ready_clients:
                logger.info(f"MCP clients ready: {', '.join(ready_clients)}")
                break
                
            logger.warning(f"No MCP clients ready yet, waiting (attempt {attempt+1}/{max_attempts})...")
            await asyncio.sleep(2)
        
        # Initialize the conversation with system and user messages
        self.messages = [
            ChatCompletionRequestSystemMessage(
                role="system",
                content=self.system_prompt
            ),
            ChatCompletionRequestUserMessage(
                role="user",
                content=self.task
            )
        ]
    
    async def shutdown(self):
        """Shutdown all MCP clients"""
        logger.info("Shutting down MCP clients...")
        # Import here to avoid circular imports
        import os
        os._exit(0)
    
    async def _process_with_anthropic(self) -> Tuple[List[Dict[str, Any]], bool]:
        """Process a single iteration with Anthropic API
        
        Returns:
            Tuple containing the updated anthropic_messages and a boolean indicating if task is complete
        """
        # Convert messages to Anthropic format
        anthropic_messages = []
        
        # Create system prompt with caching for Anthropic
        system_prompt = None
        if self.system_prompt:
            system_prompt = [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        
        for i, msg in enumerate(self.messages):
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
            model=self.model,
            system=system_prompt
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
                
                # Import ClientManager here
                from mcp_bridge.mcp_clients.McpClientManager import ClientManager
                # Import the call_tool utility
                from mcp_bridge.anthropic_clients.utils import call_tool
                
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
                            self.messages.append(tool_call_msg)
                            
                            # Add error message as tool result
                            tool_result_message = ChatCompletionRequestMessage(
                                role="tool",
                                content=error_message,
                                tool_call_id=tool_id
                            )
                            self.messages.append(tool_result_message)
                            
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
                        self.messages.append(tool_call_msg)
                        
                        # Add tool result message
                        tool_result_message = ChatCompletionRequestMessage(
                            role="tool",
                            content=result_text,
                            tool_call_id=tool_id
                        )
                        self.messages.append(tool_result_message)
                        
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
                        self.messages.append(error_message)
                        anthropic_messages.append({
                            "role": "user",
                            "content": f"There was an error processing the tool call: {str(e)}. Please try a different approach."
                        })
                
                # Tool calls were processed, return to continue the loop
                return anthropic_messages, False
                
            # Handle regular text response
            message_content = message_data["content"]
            
            # Add assistant message to conversation
            assistant_message = ChatCompletionRequestMessage(
                role="assistant",
                content=message_content
            )
            self.messages.append(assistant_message)
            print(f"\nAssistant: {message_content}")
            
            # Check for task completion
            if is_task_complete(message_content):
                logger.info("Task completed successfully.")
                return anthropic_messages, True
        else:
            logger.warning("No choices in Anthropic response")
        
        return anthropic_messages, False
            
    async def _process_with_openai(self) -> bool:
        """Process a single iteration with OpenAI API
        
        Returns:
            Boolean indicating if task is complete
        """
        # Process with MCP OpenAI-compatible API
        request = CreateChatCompletionRequest(
            model=self.model,
            messages=self.messages
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
                self.messages.append(assistant_message)
                
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
                                self.messages.append(tool_result_message)
                                
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
                                self.messages.append(tool_result_message)
                        else:
                            error_message = f"Tool {tool_name} not found or not available"
                            logger.error(error_message)
                            
                            # Add error message to conversation
                            tool_result_message = ChatCompletionRequestMessage(
                                role="tool",
                                content=error_message,
                                tool_call_id=tool_id
                            )
                            self.messages.append(tool_result_message)
            else:
                if message.content:
                    self.messages.append(assistant_message)
                    print(f"\nAssistant: {message.content}")
                    
                    # Check for task completion
                    if is_task_complete(message.content):
                        logger.info("Task completed successfully.")
                        return True  # Task is complete
        else:
            logger.warning("No choices in response")
        
        return False  # Task is not complete
            
    async def run_agent_loop(self):
        """Run the agent loop to process the task until completion"""
        try:
            await self.initialize()
            logger.info("Starting agent loop...")
            
            # Keep running until the task is complete
            for iteration in range(self.max_iterations):
                logger.info(f"Agent iteration {iteration+1}/{self.max_iterations}")
                
                try:
                    # Process with either Anthropic or OpenAI API based on model name
                    task_complete = False
                    if is_anthropic_model(self.model):
                        # Use Anthropic processing
                        _, task_complete = await self._process_with_anthropic()
                    else:
                        # Use OpenAI processing
                        task_complete = await self._process_with_openai()
                        
                    # If task is complete, return the messages
                    if task_complete:
                        return self.messages
                    
                except Exception as e:
                    logger.exception(f"API error: {str(e)}")
                    # Add a user message to the conversation explaining the error
                    error_message = ChatCompletionRequestMessage(
                        role="user",
                        content=f"There was an error with the previous request: {str(e)}. Please try a different approach."
                    )
                    self.messages.append(error_message)
            
            # If we reached max iterations without completion
            if iteration >= self.max_iterations - 1:
                logger.warning(f"Reached maximum iterations ({self.max_iterations}) without task completion")
                
            # Return final messages for inspection
            return self.messages
                
        except Exception as e:
            logger.exception(f"Error in agent loop: {str(e)}")
            raise

