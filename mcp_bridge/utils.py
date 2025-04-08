"""Common utility functions for MCP Bridge"""

import os
from typing import Optional

def force_exit(exit_code: int = 0) -> None:
    """Force exit the program with the specified exit code
    
    This is a hard exit that bypasses normal Python cleanup
    and should only be used when necessary.
    """
    os._exit(exit_code) 