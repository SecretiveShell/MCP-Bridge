# mcp_bridge/utils/__init__.py

# Import and apply patches
from .library_patcher import apply_patches

# Apply patches at module import time
original_functions = apply_patches()