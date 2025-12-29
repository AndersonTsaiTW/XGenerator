"""
Authentication utilities for API key verification
"""
from fastapi import Header, HTTPException
from pathlib import Path
import json
from typing import Dict

from app.config import USERS_DIR


def mask_api_key(api_key: str) -> str:
    """
    Mask API key for display
    Example: sk_live_abc123def456ghi789 -> sk_live_abc1...i789
    """
    if not api_key or len(api_key) < 16:
        return "***"
    
    prefix = api_key[:12]  # "sk_live_abc1"
    suffix = api_key[-4:]   # "i789"
    return f"{prefix}...{suffix}"


async def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")) -> Dict:
    """
    Verify API key and return user data
    
    Args:
        x_api_key: API key from X-API-Key header
        
    Returns:
        User data dict
        
    Raises:
        HTTPException: 401 if API key is invalid
    """
    # Search all users for matching api_key
    for user_file in USERS_DIR.glob("*.json"):
        try:
            with open(user_file, "r") as f:
                user_data = json.load(f)
            
            if user_data.get("api_key") == x_api_key:
                return user_data
        except (json.JSONDecodeError, IOError):
            continue
    
    raise HTTPException(
        status_code=401,
        detail={
            "error": "invalid_api_key",
            "message": "Invalid or missing API key",
            "details": None
        }
    )


async def verify_ownership(resource_user_id: str, current_user: Dict) -> None:
    """
    Verify that current user owns the resource
    
    Args:
        resource_user_id: User ID of the resource owner
        current_user: Current authenticated user data
        
    Raises:
        HTTPException: 403 if user doesn't own the resource
    """
    if resource_user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "forbidden",
                "message": "You don't have permission to access this resource",
                "details": {"resource_owner": resource_user_id}
            }
        )
