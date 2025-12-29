"""
Rate limiting configuration for XGGenerator API
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import os

# Initialize limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],  # No global limits, set per-endpoint
    strategy="fixed-window"
)

# Check if in testing mode
TESTING = os.getenv("TESTING", "false").lower() in ("true", "1", "yes")


def get_api_key_from_request(request: Request) -> str:
    """Extract API key from request header for rate limiting"""
    return request.headers.get("X-API-Key", get_remote_address(request))


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom handler for rate limit exceeded errors"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please slow down.",
            "details": {
                "retry_after": str(exc.detail)
            }
        }
    )
