import json
import asyncio
from typing import Dict, List, Optional, Union, Protocol, runtime_checkable
from loguru import logger


@runtime_checkable
class ToolResult(Protocol):
    """Protocol for tool result objects"""
    content: List[object]


async def extract_tool_result_text(result: Optional[ToolResult]) -> str:
    """Extract text content from tool result"""
    if not result or not hasattr(result, "content"):
        return "No result returned from tool"
        
    text_parts = []
    for part in result.content:
        if hasattr(part, "text"):
            text_parts.append(part.text)
    
    return " ".join(text_parts) if text_parts else "No text content in result"


async def extract_tool_result_image(result: Optional[ToolResult]) -> Optional[Dict[str, object]]:
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


async def is_image_tool(tool_name: str) -> bool:
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


def is_anthropic_model(model: str) -> bool:
    """Check if the model is an Anthropic model"""
    return model.startswith("claude-") or "claude" in model


def is_task_complete(content: str) -> bool:
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