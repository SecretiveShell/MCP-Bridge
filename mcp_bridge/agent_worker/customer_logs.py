#!/usr/bin/env python
"""Customer-facing message logs for MCP-Bridge Agent Worker"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, TypedDict
from loguru import logger


class ThinkingBlockDict(TypedDict, total=False):
    """Type for thinking block dictionary"""
    timestamp: str
    content: str
    signature: Optional[str]


class MessageDict(TypedDict, total=False):
    """Type for message dictionary"""
    timestamp: str
    role: str
    content: str
    type: str
    tool_name: Optional[str]
    tool_input: object
    tool_id: Optional[str]
    has_image: Optional[bool]


class SystemEventDict(TypedDict, total=False):
    """Type for system event dictionary"""
    timestamp: str
    type: str
    details: Dict[str, object]


class StreamEntryDict(TypedDict):
    """Type for stream entry dictionary"""
    entry_type: str
    timestamp: str
    data: Dict[str, object]


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
        self.stream_path: Optional[str] = None
        self.messages: List[MessageDict] = []
        self.thinking_blocks: List[ThinkingBlockDict] = []
        self.system_events: List[SystemEventDict] = []
        
    def initialize(self) -> str:
        """Initialize the log file and directory
        
        Returns:
            Path to the log file
        """
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a streaming log file for real-time reading
        self.stream_path = os.path.join(self.log_dir, f"session_{timestamp}_{self.session_id}.jsonl")
        with open(self.stream_path, "w") as f:
            # Write the session metadata as the first line
            f.write(json.dumps({
                "type": "metadata",
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat()
            }) + "\n")
            
        logger.info(f"Initialized streaming log: {self.stream_path}")
        return self.stream_path
    
    def log_message(self, role: str, content: str, message_type: str = "message") -> None:
        """Log a message in the conversation
        
        Args:
            role: Role of the message sender (user, assistant, system)
            content: Content of the message
            message_type: Type of message (message, tool_call, tool_result)
        """
        message: MessageDict = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "type": message_type
        }
        
        self.messages.append(message)
        self._append_to_stream("message", message)
        
    def log_tool_call(self, tool_name: str, tool_input: object, tool_id: str) -> None:
        """Log a tool call by the assistant
        
        Args:
            tool_name: Name of the tool being called
            tool_input: Input to the tool
            tool_id: ID of the tool call
        """
        tool_call: MessageDict = {
            "timestamp": datetime.now().isoformat(),
            "role": "assistant",
            "type": "tool_call",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_id": tool_id
        }
        
        self.messages.append(tool_call)
        self._append_to_stream("message", tool_call)
        
    def log_tool_result(self, tool_id: str, result: str, has_image: bool = False) -> None:
        """Log a tool result
        
        Args:
            tool_id: ID of the tool call
            result: Result from the tool
            has_image: Whether the result includes an image
        """
        tool_result: MessageDict = {
            "timestamp": datetime.now().isoformat(),
            "role": "tool",
            "type": "tool_result",
            "tool_id": tool_id,
            "content": result,
            "has_image": has_image
        }
        
        self.messages.append(tool_result)
        self._append_to_stream("message", tool_result)
        
    def log_thinking(self, thinking_content: str, signature: Optional[str] = None) -> None:
        """Log a thinking block from the AI
        
        Args:
            thinking_content: Content of the thinking block
            signature: Optional signature for the thinking block
        """
        thinking: ThinkingBlockDict = {
            "timestamp": datetime.now().isoformat(),
            "content": thinking_content,
            "signature": signature
        }
        
        self.thinking_blocks.append(thinking)
        self._append_to_stream("thinking", thinking)
        
    def log_system_event(self, event_type: str, details: Dict[str, object]) -> None:
        """Log a system event
        
        Args:
            event_type: Type of system event (error, info, warning)
            details: Details of the event
        """
        event: SystemEventDict = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details
        }
        
        self.system_events.append(event)
        self._append_to_stream("system_event", event)
        
    def _append_to_stream(self, entry_type: str, data: Dict[str, object]) -> None:
        """Append an entry to the streaming log file
        
        Args:
            entry_type: Type of entry (message, thinking, system_event)
            data: Data to append
        """
        if not self.stream_path:
            logger.warning("Cannot write to stream, log file not initialized")
            return
            
        try:
            stream_entry: StreamEntryDict = {
                "entry_type": entry_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            with open(self.stream_path, "a") as f:
                f.write(json.dumps(stream_entry) + "\n")
                f.flush()  # Force write to disk
        except Exception as e:
            logger.error(f"Error appending to stream log: {e}")
    
    def write_final_log(self) -> None:
        """Write a final summary entry to the log at the end of the session"""
        if not self.stream_path:
            logger.warning("Cannot write final entry, log file not initialized")
            return
            
        try:
            # Add a final summary entry to the stream
            summary = self.get_summary()
            summary_entry: StreamEntryDict = {
                "entry_type": "summary",
                "timestamp": datetime.now().isoformat(),
                "data": summary
            }
            
            with open(self.stream_path, "a") as f:
                f.write(json.dumps(summary_entry) + "\n")
                f.flush()
                
            logger.info(f"Finalized customer message log: {self.stream_path}")
        except Exception as e:
            logger.error(f"Error writing final summary: {e}")
    
    def get_summary(self) -> Dict[str, object]:
        """Get a summary of the message flow
        
        Returns:
            Summary dictionary with message counts and timing information
        """
        if not self.messages:
            return {"message_count": 0}
            
        start_time = datetime.fromisoformat(self.messages[0]["timestamp"])
        end_time = datetime.fromisoformat(self.messages[-1]["timestamp"])
        
        # Calculate message counts by role
        role_counts: Dict[str, int] = {}
        for msg in self.messages:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            
        # Calculate tool usage
        tool_calls = [msg for msg in self.messages if msg.get("type") == "tool_call"]
        tool_results = [msg for msg in self.messages if msg.get("type") == "tool_result"]
        
        # Count tools by name
        tool_counts: Dict[str, int] = {}
        for tool in tool_calls:
            tool_name = tool.get("tool_name", "unknown")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "message_count": len(self.messages),
            "thinking_count": len(self.thinking_blocks),
            "event_count": len(self.system_events),
            "role_counts": role_counts,
            "tool_call_count": len(tool_calls),
            "tool_result_count": len(tool_results),
            "tool_counts": tool_counts
        }

# Singleton instance for global access
_logger_instance: Optional[CustomerMessageLogger] = None

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