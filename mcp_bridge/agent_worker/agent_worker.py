#!/usr/bin/env python
"""Agent worker module that provides standalone command-line agent execution"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from mcp_bridge.agent_worker.utils import is_anthropic_model
from mcp_bridge.agent_worker.anthropic_handler import process_with_anthropic
from mcp_bridge.agent_worker.openai_handler import process_with_openai
from lmos_openai_types import (
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
        self.thinking_blocks: List[Dict[str, Any]] = []
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
                        _, updated_messages, thinking_blocks, task_complete = await process_with_anthropic(
                            messages=self.messages,
                            model=self.model,
                            system_prompt=self.system_prompt,
                            thinking_blocks=self.thinking_blocks
                        )
                        self.messages = updated_messages
                        
                        # Check for duplicate thinking blocks before adding
                        # ThinkingBlock from Anthropic has a signature property
                        existing_signatures = {block.signature for block in self.thinking_blocks if block.signature}
                        unique_blocks = [block for block in thinking_blocks if block.signature not in existing_signatures]
                        self.thinking_blocks.extend(unique_blocks)
                    else:
                        # Use OpenAI processing
                        updated_messages, task_complete = await process_with_openai(
                            messages=self.messages,
                            model=self.model
                        )
                        self.messages = updated_messages
                        
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

