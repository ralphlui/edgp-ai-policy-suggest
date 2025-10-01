"""
Custom exception handlers for standardized API responses
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


class SchemaGenerationError(Exception):
    """Exception raised when schema generation fails"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class StandardResponse(BaseModel):
    """Standardized API response format"""
    data: Optional[Any] = None
    success: bool = False
    message: str = ""
    totalRecord: int = 0
    status: int = 200

def create_standard_response(
    status_code: int, 
    message: str, 
    data: Optional[Any] = None, 
    success: bool = False,
    total_record: int = 0
) -> JSONResponse:
    """Create standardized API response"""
    response_data = StandardResponse(
        data=data,
        success=success,
        message=message,
        totalRecord=total_record,
        status=status_code
    )
    return JSONResponse(
        status_code=status_code,
        content=response_data.model_dump()
    )

async def authentication_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle authentication exceptions with standardized format"""
    
    # Map common authentication error messages
    message_mapping = {
        "Bearer token required": "Authentication required",
        "Invalid authentication scheme. Expected 'Bearer'": "Invalid authentication scheme",
        "Bearer token missing": "Authentication token missing",
        "Token missing user email": "Invalid token format",
        "Token has expired": "JWT token is expired",
        "JWT token is expired": "JWT token is expired",
        "Invalid token signature": "Invalid token signature",
        "Invalid JWT token": "Invalid JWT token",
        "User not authorized": "User not authorized",
        "User not found": "User not found",
        "Insufficient permissions": "Insufficient permissions"
    }
    
    # Get mapped message or use original
    mapped_message = message_mapping.get(exc.detail, exc.detail)
    
    logger.warning(f"Authentication error: {mapped_message} (Original: {exc.detail})")
    
    return create_standard_response(
        status_code=exc.status_code,
        message=mapped_message,
        data=None,
        success=False,
        total_record=0
    )

async def general_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle general HTTP exceptions with standardized format"""
    
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return create_standard_response(
        status_code=exc.status_code,
        message=exc.detail,
        data=None,
        success=False,
        total_record=0
    )

async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle validation exceptions"""
    
    logger.error(f"Validation error: {str(exc)}")
    
    return create_standard_response(
        status_code=422,
        message=f"Validation error: {str(exc)}",
        data=None,
        success=False,
        total_record=0
    )

async def internal_server_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle internal server errors"""
    
    logger.error(f"Internal server error: {str(exc)}")
    
    return create_standard_response(
        status_code=500,
        message="Internal server error",
        data=None,
        success=False,
        total_record=0
    )