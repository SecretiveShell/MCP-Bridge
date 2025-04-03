#!/usr/bin/env python
"""Agent worker module that provides standalone command-line agent execution"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional
from loguru import logger

from mcp_bridge.openai_clients.chatCompletion import chat_completions
from mcp_bridge.anthropic_clients.chatCompletion import anthropic_chat_completions
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
    
    def is_task_complete(self, content: str) -> bool:
        """Check if the task is complete based on the assistant's response"""
        completion_indicators = [
            "i've completed the task",
            "task is complete", 
            "the task has been completed",
            "i have completed", 
            "task has been finished",
            "the task is now finished"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in completion_indicators)
    
    async def extract_tool_result_text(self, result) -> str:
        """Extract text content from tool result"""
        if not result or not hasattr(result, "content"):
            return "No result returned from tool"
            
        text_parts = []
        for part in result.content:
            if hasattr(part, "text"):
                text_parts.append(part.text)
        
        return " ".join(text_parts) if text_parts else "No text content in result"
    
    async def extract_tool_result_image(self, result) -> Optional[Dict[str, Any]]:
        """Extract image content from tool result if available"""
        if not result or not hasattr(result, "content"):
            return None
        
        # Try multiple approaches to find and extract the image
        
        # Approach 1: Look for ImageContent objects with data attribute
        for part in result.content:
            part_type_name = type(part).__name__
            
            # Check if this is an ImageContent object (by type name)
            if part_type_name == "ImageContent" or hasattr(part, "type") and getattr(part, "type") == "image":
                # Check for data attribute which contains the base64 image
                if hasattr(part, "data"):
                    try:
                        # Get the image data and MIME type
                        image_data = part.data
                        mime_type = "image/png"  # Default
                        if hasattr(part, "mimeType"):
                            mime_type = part.mimeType
                        
                        # Save the image locally for inspection
                        self.save_image_locally(image_data, mime_type)
                        
                        return {
                            "type": "image",
                            "source": {
                                "type": "base64", 
                                "media_type": mime_type,
                                "data": image_data
                            }
                        }
                    except Exception as e:
                        logger.error(f"Error processing image data from ImageContent: {e}")
        
        # Approach 2: Look for direct image attribute (legacy approach)
        for part in result.content:
            if hasattr(part, "image") and part.image:
                try:
                    image_data = part.image
                    self.save_image_locally(image_data)
                    
                    return {
                        "type": "image",
                        "source": {
                            "type": "base64", 
                            "media_type": "image/png",
                            "data": image_data
                        }
                    }
                except Exception as e:
                    logger.error(f"Error processing image data from attribute: {e}")
        
        # Approach 3: Check if content parts are dictionaries with image-related keys
        for part in result.content:
            if isinstance(part, dict):
                for key in ["image", "data"]:
                    if key in part and isinstance(part[key], str) and len(part[key]) > 1000:
                        try:
                            image_data = part[key]
                            self.save_image_locally(image_data)
                            
                            return {
                                "type": "image",
                                "source": {
                                    "type": "base64", 
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            }
                        except Exception as e:
                            logger.error(f"Error processing image data from dictionary: {e}")
                
        return None
    
    def save_image_locally(self, base64_data: str, mime_type: str = "image/png") -> None:
        """Save base64 image data to a local file for inspection"""
        import base64
        import os
        from datetime import datetime
        
        # Create a screenshots directory if it doesn't exist
        screenshots_dir = "screenshots"
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Determine file extension from MIME type
        extension = "png"  # Default
        if mime_type == "image/jpeg":
            extension = "jpg"
        elif mime_type == "image/gif":
            extension = "gif"
        elif mime_type == "image/webp":
            extension = "webp"
        
        # Generate a filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.{extension}")
        
        try:
            # Decode the base64 data
            image_data = base64.b64decode(base64_data)
            
            # Write the image to a file
            with open(filename, "wb") as f:
                f.write(image_data)
            
            logger.info(f"Saved screenshot: {filename}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")
    
    async def is_image_tool(self, tool_name: str) -> bool:
        """Check if a tool is expected to return image data by examining its description or response type"""
        # Known image tools - fallback if we can't determine from specs
        known_image_tools = [
            "remote_macos_get_screen", 
            "mcp_remote_macos_get_screen",
            "get_screen"
        ]
        
        # Check if tool is known to return images
        if tool_name.lower() in [t.lower() for t in known_image_tools]:
            return True
            
        # Try to get the tool definition from MCP
        try:
            from mcp_bridge.mcp_clients.McpClientManager import ClientManager
            
            # Get the client that has this tool
            client = await ClientManager.get_client_from_tool(tool_name)
            if client:
                # Get the tool definition
                tools_result = await client.list_tools()
                if tools_result and hasattr(tools_result, "tools"):
                    for tool in tools_result.tools:
                        if tool.name == tool_name:
                            # Check if the tool description mentions images or screenshots
                            if hasattr(tool, "description") and tool.description:
                                desc_lower = tool.description.lower()
                                if any(term in desc_lower for term in ["image", "screenshot", "screen capture", "photo"]):
                                    logger.debug(f"Tool {tool_name} identified as image tool from description")
                                    return True
                            
                            # Check if the tool's response schema includes image types
                            if hasattr(tool, "outputSchema") and tool.outputSchema:
                                schema_str = str(tool.outputSchema).lower()
                                if any(term in schema_str for term in ["image", "imagecontent", "screenshot", "base64"]):
                                    logger.debug(f"Tool {tool_name} identified as image tool from output schema")
                                    return True
        except Exception as e:
            logger.warning(f"Error checking if {tool_name} is an image tool: {e}")
        
        # Fallback to known tools list if we couldn't determine from specs
        return False
    
    def is_anthropic_model(self) -> bool:
        """Check if the model is an Anthropic model"""
        return self.model.startswith("claude-") or "claude" in self.model
    
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
                    if self.is_anthropic_model():
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
                                        is_image_tool = await self.is_image_tool(tool_name)
                                        
                                        result_text = await self.extract_tool_result_text(result)
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
                                        if is_image_tool:
                                            image_content = await self.extract_tool_result_image(result)
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
                                
                                # Continue to next iteration with updated messages
                                continue
                                
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
                            if self.is_task_complete(message_content):
                                logger.info("Task completed successfully.")
                                return self.messages
                        else:
                            logger.warning("No choices in Anthropic response")
                            
                    else:
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
                                                result_text = await self.extract_tool_result_text(result)
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
                                                if await self.is_image_tool(tool_name):
                                                    image_content = await self.extract_tool_result_image(result)
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
                                    if self.is_task_complete(message.content):
                                        logger.info("Task completed successfully.")
                                        return self.messages  # Return immediately when task is complete
                        else:
                            logger.warning("No choices in response")
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


def load_config_from_file(config_file: str = "agent_worker_task.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        config_path = os.path.join(os.getcwd(), config_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Ensure required fields are present
        if 'task' not in config:
            raise ValueError("Missing required field 'task' in config file")
            
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise


async def run_cli():
    """Run the CLI interface for the agent worker"""
    worker = None
    try:
        # Load configuration from file
        config = load_config_from_file()
        
        # Set default values if not in config
        task = config['task']
        model = config.get('model', "anthropic.claude-3-haiku-20240307-v1:0")
        system_prompt = config.get('system_prompt')
        verbose = config.get('verbose', False)
        max_iterations = config.get('max_iterations', 10)
        
        # Configure logging
        log_level = "DEBUG" if verbose else "INFO"
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        
        logger.info(f"Starting MCP-Bridge Agent Worker with task: {task}")
        
        # Create and run the agent worker
        worker = AgentWorker(
            task=task,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
        )
        
        messages = await worker.run_agent_loop()
        logger.info("Agent worker completed")
        return 0
    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {str(e)}")
        return 1
    except KeyboardInterrupt:
        logger.info("Agent worker interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Error running agent worker: {e}")
        return 1
    finally:
        # Ensure clients are shut down even if there's an exception
        if worker:
            await worker.shutdown()
            
        # Force exit any remaining tasks 
        remaining_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if remaining_tasks:
            logger.info(f"Cancelling {len(remaining_tasks)} remaining tasks")
            for task in remaining_tasks:
                task.cancel()
            
            # Wait briefly for tasks to be cancelled
            await asyncio.wait(remaining_tasks, timeout=1.0)


def main():
    """Entry point for the CLI"""
    try:
        exit_code = asyncio.run(run_cli())
        # Force program to exit immediately
        os._exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Agent worker interrupted by user")
        os._exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        os._exit(1)


if __name__ == "__main__":
    main() 