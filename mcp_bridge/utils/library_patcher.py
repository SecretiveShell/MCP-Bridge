# mcp_bridge/utils/library_patcher.py

import json
from loguru import logger
import mcp.types
from mcp.types import JSONRPCMessage
from mcp.shared.message import SessionMessage

def patch_jsonrpc_message():
    """
    Patch the JSONRPCMessage class to add the 'message' property.
    """
    # Check if the class is already patched
    if hasattr(JSONRPCMessage, '_original_getattr'):
        logger.info("JSONRPCMessage already patched")
        return
        
    # Store the original __getattr__ method
    original_getattr = JSONRPCMessage.__getattr__
    
    # Define the patched method
    def patched_getattr(self, name):
        # If trying to access 'message', return self
        if name == 'message':
            return self
            
        # Otherwise, use the original method
        return original_getattr(self, name)
    
    # Apply the patch
    JSONRPCMessage._original_getattr = original_getattr
    JSONRPCMessage.__getattr__ = patched_getattr
    
    logger.info("Successfully patched JSONRPCMessage.__getattr__")
    
    return original_getattr

def patch_session_message():
    """
    Patch the SessionMessage class to add missing serialization methods.
    """
    # Skip if already patched
    if hasattr(SessionMessage, 'model_dump_json'):
        logger.info("SessionMessage already patched")
        return
    
    # Add model_dump_json method to SessionMessage
    def model_dump_json(self, **kwargs):
        """Convert the message to JSON string, compatible with Pydantic v2."""
        # If message has its own model_dump_json method, use it
        if hasattr(self.message, 'model_dump_json'):
            return self.message.model_dump_json(**kwargs)
        
        # If message has dict method (Pydantic v1), use it
        if hasattr(self.message, 'dict'):
            # Convert to dict then to JSON
            message_dict = self.message.dict(**kwargs)
            return json.dumps(message_dict)
        
        # Fallback: try to convert to dict directly
        try:
            return json.dumps(self.message.__dict__)
        except:
            # Last resort, just stringify
            return json.dumps(str(self.message))
    
    # Add the method to the class
    SessionMessage.model_dump_json = model_dump_json
    
    logger.info("Successfully patched SessionMessage with model_dump_json")
    
    return True

def apply_patches():
    """Apply all necessary patches to the MCP library."""
    original_getattr = patch_jsonrpc_message()
    session_message_patched = patch_session_message()
    
    logger.info("All MCP library patches applied successfully")
    return {
        "original_getattr": original_getattr,
        "session_message_patched": session_message_patched
    }