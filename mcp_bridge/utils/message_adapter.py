# mcp_bridge/utils/message_adapter.py

from typing import Any

class MessageWrapper:
    """
    A wrapper class for JSONRPCMessage objects that provides the expected structure.
    Only used in client-side code, not for patching the library.
    """
    def __init__(self, original_message):
        self._original = original_message
        
    @property
    def message(self):
        """Return self to provide the message.message.root structure"""
        return self
        
    @property
    def root(self):
        """Forward root access to the original message"""
        if hasattr(self._original, 'root'):
            return self._original.root
        return None
        
    def __getattr__(self, name):
        """Forward all other attribute access to the original message"""
        return getattr(self._original, name)

def wrap_message(message: Any) -> Any:
    """
    Wrap a message object to provide the expected message.message.root structure.
    """
    if message is None or isinstance(message, Exception):
        return message
        
    # No need to wrap if it's already a MessageWrapper
    if isinstance(message, MessageWrapper):
        return message
        
    # Wrap the message
    return MessageWrapper(message)

# Add alias for backward compatibility
adapt_jsonrpc_message = wrap_message

# Dummy function for backward compatibility
def patch_base_session():
    """
    Dummy function for backward compatibility.
    The actual patching is done in library_patcher.py
    """
    return None