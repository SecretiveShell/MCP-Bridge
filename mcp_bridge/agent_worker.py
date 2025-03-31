#!/usr/bin/env python
"""Agent worker module that provides standalone command-line agent execution"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, List, Optional
from loguru import logger

from mcp_bridge.openai_clients.chatCompletion import chat_completions
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
        from mcp_bridge.mcp_clients.McpClientManager import ClientManager
        # Shutdown all client sessions
        await ClientManager.shutdown()
        logger.info("All MCP clients shut down")
    
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
        if not result or not result.content:
            return "No result returned from tool"
            
        text_parts = []
        for part in result.content:
            if hasattr(part, "text"):
                text_parts.append(part.text)
        
        return " ".join(text_parts) if text_parts else "No text content in result"
    
    async def run_agent_loop(self):
        """Run the agent loop to process the task until completion"""
        try:
            await self.initialize()
            logger.info("Starting agent loop...")
            
            # Keep running until the task is complete
            for iteration in range(self.max_iterations):
                logger.info(f"Agent iteration {iteration+1}/{self.max_iterations}")
                
                # Process with MCP OpenAI-compatible API
                request = CreateChatCompletionRequest(
                    model=self.model,
                    messages=self.messages
                )
                
                try:
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
        
        try:
            messages = await worker.run_agent_loop()
            logger.info("Agent worker completed")
            return 0
        finally:
            # Ensure clients are shut down even if there's an exception
            await worker.shutdown()
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {str(e)}")
        return 1
    except KeyboardInterrupt:
        logger.info("Agent worker interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Error running agent worker: {e}")
        return 1


def main():
    """Entry point for the CLI"""
    try:
        exit_code = asyncio.run(run_cli())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Agent worker interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 