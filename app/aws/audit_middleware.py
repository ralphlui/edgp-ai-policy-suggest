#!/usr/bin/env python3
"""
Audit Middleware for FastAPI
Automatically logs all API requests for audit trail
"""

import time
import json
import logging
import jwt
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.aws.audit_models import AuditLogDTO, AuditContext, ActivityType, ResponseStatus, endpoint_activity_mapping, get_activity_info
from app.aws.audit_service import send_audit_log_async, log_audit_locally
import asyncio
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic audit logging
    Captures all API requests and responses for audit trail
    """
    
    def __init__(self, app, 
                 excluded_paths: Optional[list] = None,
                 log_request_body: bool = True,
                 log_response_body: bool = False,
                 max_body_size: int = 10000):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health", 
            "/metrics", 
            "/docs", 
            "/openapi.json",
            "/favicon.ico"
        ]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip audit logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Capture request details
        audit_context = await self._build_audit_context(request, request_id)
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create audit log
            await self._create_audit_log(
                request, 
                response, 
                audit_context, 
                processing_time,
                success=True
            )
            
            return response
            
        except Exception as e:
            # Log failed requests
            processing_time = time.time() - start_time
            
            error_response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id}
            )
            
            await self._create_audit_log(
                request, 
                error_response, 
                audit_context, 
                processing_time,
                success=False,
                error_message=str(e)
            )
            
            raise e
    
    async def _build_audit_context(self, request: Request, request_id: str) -> AuditContext:
        """Build audit context from request"""
        
        # Extract user information from authentication
        user_id = await self._extract_user_id(request)
        user_name = await self._extract_user_name(request)
        
        # Get client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        return AuditContext(
            request_id=request_id,
            user_id=user_id,
            user_name=user_name,
            client_ip=client_ip,
            user_agent=user_agent,
            timestamp=datetime.utcnow()
        )
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID (sub claim) from JWT Bearer token"""
        try:
            # Check for Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            # Extract token
            token = auth_header.replace("Bearer ", "").strip()
            
            # Decode JWT token (without verification for audit purposes)
            # In production, you should verify the token with proper secret/key
            try:
                # Decode without verification to extract claims for audit
                payload = jwt.decode(token, options={"verify_signature": False})
                
                # Extract user ID from 'sub' claim (standard JWT claim)
                user_id = payload.get("sub")
                if user_id:
                    logger.debug(f"Extracted user ID from JWT: {user_id}")
                    return str(user_id)
                
                # Fallback to other possible user ID fields
                user_id = payload.get("user_id") or payload.get("userId")
                if user_id:
                    logger.debug(f"Extracted user ID from alternative field: {user_id}")
                    return str(user_id)
                
                logger.debug("No user ID found in JWT token")
                return None
                
            except jwt.DecodeError:
                logger.debug("Invalid JWT token format")
                return None
            except Exception as e:
                logger.debug(f"Error decoding JWT token: {e}")
                return None
            
        except Exception as e:
            logger.debug(f"Could not extract user ID: {e}")
            return None
    
    async def _extract_user_name(self, request: Request) -> Optional[str]:
        """Extract username from JWT Bearer token"""
        try:
            # Check for Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            # Extract token
            token = auth_header.replace("Bearer ", "").strip()
            
            # Decode JWT token (without verification for audit purposes)
            try:
                # Decode without verification to extract claims for audit
                payload = jwt.decode(token, options={"verify_signature": False})
                
                # Extract username from standard claims
                username = payload.get("username")
                if username:
                    logger.debug(f"Extracted username from JWT: {username}")
                    return str(username)
                
                # Fallback to other possible username fields
                username = (payload.get("preferred_username") or 
                           payload.get("name") or 
                           payload.get("email") or
                           payload.get("userName"))
                
                if username:
                    logger.debug(f"Extracted username from alternative field: {username}")
                    return str(username)
                
                logger.debug("No username found in JWT token")
                return None
                
            except jwt.DecodeError:
                logger.debug("Invalid JWT token format")
                return None
            except Exception as e:
                logger.debug(f"Error decoding JWT token: {e}")
                return None
            
        except Exception as e:
            logger.debug(f"Could not extract user name: {e}")
            return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address considering proxies"""
        # Check for forwarded headers (common in load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        forwarded = request.headers.get("X-Forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def _create_audit_log(self, 
                               request: Request, 
                               response: Response, 
                               audit_context: AuditContext,
                               processing_time: float,
                               success: bool = True,
                               error_message: Optional[str] = None):
        """Create and send audit log"""
        
        try:
            # Determine activity type and description based on endpoint
            endpoint_key = f"{request.method} {request.url.path}"
            activity_type, activity_description = get_activity_info(endpoint_key)
            
            # Determine response status
            if error_message:
                response_status = ResponseStatus.ERROR
            elif response.status_code >= 500:
                response_status = ResponseStatus.ERROR
            elif response.status_code >= 400:
                response_status = ResponseStatus.FAILED
            else:
                response_status = ResponseStatus.SUCCESS
            
            # Capture request body if enabled
            request_body = ""
            if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
                request_body = await self._safe_get_request_body(request)
            
            # Capture response body if enabled
            response_body = ""
            if self.log_response_body and hasattr(response, 'body'):
                response_body = self._safe_get_response_body(response)
            
            # Build remarks with request details
            remarks = self._build_remarks(
                request, 
                response, 
                processing_time, 
                request_body, 
                response_body,
                error_message
            )
            
            # Build complete endpoint for requestActionEndpoint
            endpoint = request.url.path
            
            # Create audit DTO with new format
            audit_dto = AuditLogDTO(
                statusCode=str(response.status_code),
                userId=audit_context.user_id or "0000",
                username=audit_context.user_name or "public user",
                activityType=activity_type,
                activityDescription=activity_description,
                requestActionEndpoint=endpoint,
                responseStatus=response_status,
                requestType=request.method.upper(),
                remarks=remarks
            )
            
            # Send audit log asynchronously
            try:
                success_sent = await send_audit_log_async(audit_dto)
                if not success_sent:
                    # Fallback to local logging
                    log_audit_locally(audit_dto)
            except Exception as e:
                logger.error(f"Failed to send audit log: {e}")
                # Fallback to local logging
                log_audit_locally(audit_dto)
                
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
    
    async def _safe_get_request_body(self, request: Request) -> str:
        """Safely extract request body with size limits"""
        try:
            if hasattr(request, '_body'):
                # Body already read
                body = request._body
            else:
                # Read body
                body = await request.body()
            
            if len(body) > self.max_body_size:
                return f"[Request body too large: {len(body)} bytes, truncated...]"
            
            # Try to decode as JSON for better formatting
            try:
                body_str = body.decode('utf-8')
                json_body = json.loads(body_str)
                return json.dumps(json_body, indent=2)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return body.decode('utf-8', errors='replace')[:self.max_body_size]
                
        except Exception as e:
            logger.debug(f"Could not capture request body: {e}")
            return "[Request body not available]"
    
    def _safe_get_response_body(self, response: Response) -> str:
        """Safely extract response body with size limits"""
        try:
            if hasattr(response, 'body') and response.body:
                body = response.body
                if isinstance(body, bytes):
                    if len(body) > self.max_body_size:
                        return f"[Response body too large: {len(body)} bytes, truncated...]"
                    
                    # Try to decode as JSON
                    try:
                        body_str = body.decode('utf-8')
                        json_body = json.loads(body_str)
                        return json.dumps(json_body, indent=2)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return body.decode('utf-8', errors='replace')[:self.max_body_size]
                else:
                    return str(body)[:self.max_body_size]
            return "[No response body]"
        except Exception as e:
            logger.debug(f"Could not capture response body: {e}")
            return "[Response body not available]"
    
    def _build_remarks(self, 
                      request: Request, 
                      response: Response, 
                      processing_time: float,
                      request_body: str,
                      response_body: str,
                      error_message: Optional[str] = None) -> str:
        """Build simple remarks for audit log"""
        
        # Determine user context for remarks
        user_context = "Public user"
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.replace("Bearer ", "").strip()
                payload = jwt.decode(token, options={"verify_signature": False})
                username = payload.get("username", "authenticated user")
                user_context = f"{username}"
            except:
                user_context = "authenticated user"
        
        # Build simple remarks based on the operation
        if error_message:
            return f"{user_context} encountered error: {error_message}"
        elif request.method == "GET":
            return f"{user_context} retrieving"
        elif request.method == "POST":
            return f"{user_context} creating/submitting"
        elif request.method == "PUT":
            return f"{user_context} updating"
        elif request.method == "DELETE":
            return f"{user_context} deleting"
        else:
            return f"{user_context} performing {request.method} operation"

# Helper function to add audit middleware to FastAPI app
def add_audit_middleware(app, **kwargs):
    """Add audit logging middleware to FastAPI application"""
    app.add_middleware(AuditLoggingMiddleware, **kwargs)
    logger.info("Audit logging middleware added to FastAPI application")