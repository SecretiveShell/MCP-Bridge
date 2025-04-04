#!/usr/bin/env python
"""Customer-facing message logs for MCP-Bridge Agent Worker"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from loguru import logger
import uuid

class CustomerMessageLogger:
    """Logger for customer-facing message flows in agent worker"""
    
    def __init__(self, log_dir: str = "logs/customer", session_id: Optional[str] = None):
        """Initialize the customer message logger
        
        Args:
            log_dir: Directory to store log files
            session_id: Optional session ID, will be generated if not provided
        """
        self.log_dir = log_dir
        self.session_id = session_id or str(uuid.uuid4())
        self.log_file = None
        self.log_path = None
        self.messages = []
        self.thinking_blocks = []
        self.system_events = []
        
    def initialize(self) -> str:
        """Initialize the log file and directory
        
        Returns:
            Path to the log file
        """
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_session_{timestamp}_{self.session_id}.json"
        self.log_path = os.path.join(self.log_dir, filename)
        
        # Create initial log structure
        log_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "messages": [],
            "thinking_blocks": [],
            "system_events": []
        }
        
        # Write initial log structure to file
        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Initialized customer message log: {self.log_path}")
        return self.log_path
    
    def log_message(self, role: str, content: str, message_type: str = "message") -> None:
        """Log a message in the conversation
        
        Args:
            role: Role of the message sender (user, assistant, system)
            content: Content of the message
            message_type: Type of message (message, tool_call, tool_result)
        """
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "type": message_type
        }
        
        self.messages.append(message)
        self._write_to_log()
        
    def log_tool_call(self, tool_name: str, tool_input: Any, tool_id: str) -> None:
        """Log a tool call by the assistant
        
        Args:
            tool_name: Name of the tool being called
            tool_input: Input to the tool
            tool_id: ID of the tool call
        """
        tool_call = {
            "timestamp": datetime.now().isoformat(),
            "role": "assistant",
            "type": "tool_call",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_id": tool_id
        }
        
        self.messages.append(tool_call)
        self._write_to_log()
        
    def log_tool_result(self, tool_id: str, result: str, has_image: bool = False) -> None:
        """Log a tool result
        
        Args:
            tool_id: ID of the tool call
            result: Result from the tool
            has_image: Whether the result includes an image
        """
        tool_result = {
            "timestamp": datetime.now().isoformat(),
            "role": "tool",
            "type": "tool_result",
            "tool_id": tool_id,
            "content": result,
            "has_image": has_image
        }
        
        self.messages.append(tool_result)
        self._write_to_log()
        
    def log_thinking(self, thinking_content: str, signature: Optional[str] = None) -> None:
        """Log a thinking block from the AI
        
        Args:
            thinking_content: Content of the thinking block
            signature: Optional signature for the thinking block
        """
        thinking = {
            "timestamp": datetime.now().isoformat(),
            "content": thinking_content,
            "signature": signature
        }
        
        self.thinking_blocks.append(thinking)
        self._write_to_log()
        
    def log_system_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a system event
        
        Args:
            event_type: Type of system event (error, info, warning)
            details: Details of the event
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        
        self.system_events.append(event)
        self._write_to_log()
        
    def _write_to_log(self) -> None:
        """Write current state to the log file"""
        if not self.log_path:
            logger.warning("Cannot write to log, log file not initialized")
            return
            
        try:
            log_data = {
                "session_id": self.session_id,
                "start_time": self.messages[0]["timestamp"] if self.messages else datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "messages": self.messages,
                "thinking_blocks": self.thinking_blocks,
                "system_events": self.system_events
            }
            
            with open(self.log_path, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to customer log: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the message flow
        
        Returns:
            Summary dictionary with message counts and timing information
        """
        if not self.messages:
            return {"message_count": 0}
            
        start_time = datetime.fromisoformat(self.messages[0]["timestamp"])
        end_time = datetime.fromisoformat(self.messages[-1]["timestamp"])
        duration = (end_time - start_time).total_seconds()
        
        # Count message types
        message_types = {}
        for message in self.messages:
            message_type = message.get("type", "message")
            if message_type not in message_types:
                message_types[message_type] = 0
            message_types[message_type] += 1
            
        return {
            "message_count": len(self.messages),
            "thinking_block_count": len(self.thinking_blocks),
            "system_event_count": len(self.system_events),
            "duration_seconds": duration,
            "message_types": message_types
        }

# Singleton instance for global access
_logger_instance = None

def get_logger(initialize: bool = False, log_dir: str = "logs/customer", session_id: Optional[str] = None) -> CustomerMessageLogger:
    """Get or create the customer message logger instance
    
    Args:
        initialize: Whether to initialize a new logger (if True, replaces existing instance)
        log_dir: Directory to store log files (if initializing)
        session_id: Optional session ID (if initializing)
        
    Returns:
        CustomerMessageLogger instance
    """
    global _logger_instance
    
    if _logger_instance is None or initialize:
        _logger_instance = CustomerMessageLogger(log_dir=log_dir, session_id=session_id)
        if initialize:
            _logger_instance.initialize()
    
    return _logger_instance 