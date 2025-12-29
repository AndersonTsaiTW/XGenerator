"""
User management endpoints
"""
from fastapi import APIRouter, HTTPException, Request
import secrets
import json
from datetime import datetime
from pathlib import Path

from app.config import USERS_DIR
from app.models.schemas import UserCreate, UserCreateResponse, UserResponse
from app.utils.file_utils import generate_id
from app.utils.auth import mask_api_key
from app.utils.rate_limit import limiter

router = APIRouter(prefix="/users", tags=["users"])


def check_username_exists(username: str) -> bool:
    """Check if username already exists"""
    for user_file in USERS_DIR.glob("*.json"):
        try:
            with open(user_file, "r") as f:
                user_data = json.load(f)
            if user_data.get("username") == username:
                return True
        except (json.JSONDecodeError, IOError):
            continue
    return False


@router.post("", response_model=UserCreateResponse, status_code=201)
@limiter.limit("12/hour")  # Prevent mass user creation
async def create_user(request: Request, user_request: UserCreate):
    """
    Create a new user and generate an API key.
    
    **Rate Limit**: 12 requests per hour per IP
    
    **Important**: The API key is shown only once upon creation. Save it securely!
    
    - **username**: 3-50 characters, alphanumeric + underscore/hyphen only
    - **email**: Optional email address
    
    Returns the user ID and full API key (masked in subsequent requests).
    """
    # Check username uniqueness
    if check_username_exists(user_request.username):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "username_exists",
                "message": f"Username '{request.username}' is already taken",
                "details": None
            }
        )
    
    # Generate user ID and API key
    user_id = generate_id()
    api_key = f"sk_live_{secrets.token_urlsafe(32)}"
    created_at = datetime.utcnow().isoformat() + "Z"
    
    # Create user data
    user_data = {
        "user_id": user_id,
        "username": user_request.username,
        "email": user_request.email,
        "tier": "premium",  # Default tier for all new users
        "api_key": api_key,
        "created_at": created_at
    }
    
    # Save to file
    user_file = USERS_DIR / f"{user_id}.json"
    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=2)
    
    # Return response with full API key
    return UserCreateResponse(**user_data)


@router.get("", response_model=list[UserResponse])
@limiter.limit("120/minute")
async def list_users(request: Request):
    """
    List all users.
    
    **Rate Limit**: 120 requests per minute per IP
    
    API keys are masked for security (shown as preview only).
    """
    users = []
    
    for user_file in sorted(USERS_DIR.glob("*.json")):
        try:
            with open(user_file, "r") as f:
                user_data = json.load(f)
            
            # Ensure tier exists (backwards compatibility)
            if 'tier' not in user_data:
                user_data['tier'] = 'premium'
            
            # Mask API key
            user_data["api_key_preview"] = mask_api_key(user_data.get("api_key", ""))
            del user_data["api_key"]  # Remove full key
            
            users.append(UserResponse(**user_data))
        except (json.JSONDecodeError, IOError, KeyError):
            continue
    
    return users


@router.get("/{user_id}", response_model=UserResponse)
@limiter.limit("120/minute")
async def get_user(request: Request, user_id: str):
    """
    Get user details by user ID.
    
    **Rate Limit**: 120 requests per minute per IP
    
    API key is masked for security.
    """
    user_file = USERS_DIR / f"{user_id}.json"
    
    if not user_file.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "user_not_found",
                "message": f"User {user_id} not found",
                "details": None
            }
        )
    
    try:
        with open(user_file, "r") as f:
            user_data = json.load(f)
        
        # Ensure tier exists (backwards compatibility)
        if 'tier' not in user_data:
            user_data['tier'] = 'premium'
        
        # Mask API key
        user_data["api_key_preview"] = mask_api_key(user_data.get("api_key", ""))
        del user_data["api_key"]
        
        return UserResponse(**user_data)
    except (json.JSONDecodeError, IOError, KeyError) as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "read_failed",
                "message": f"Failed to read user data: {str(e)}",
                "details": None
            }
        )
