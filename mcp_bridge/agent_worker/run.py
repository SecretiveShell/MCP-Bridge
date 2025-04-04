#!/usr/bin/env python
"""Agent worker module that provides standalone command-line agent execution"""

import asyncio
import json
import sys
import os
import uuid
from typing import Dict, Any
from loguru import logger
from agent_worker import AgentWorker

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
        
        # Create logs directory for customer logs
        logs_dir = config.get('logs_dir', "logs/customer")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate a session ID for this run
        session_id = config.get('session_id', str(uuid.uuid4()))
        
        logger.info(f"Starting MCP-Bridge Agent Worker with task: {task}")
        logger.info(f"Using session ID: {session_id}")
        logger.info(f"Customer logs will be stored in: {logs_dir}")
        
        # Create and run the agent worker
        worker = AgentWorker(
            task=task,
            model=model,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            session_id=session_id,
        )
        
        messages = await worker.run_agent_loop()
        logger.info("Agent worker completed")
        logger.info(f"Check customer logs in {logs_dir}")
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
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Agent worker interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 