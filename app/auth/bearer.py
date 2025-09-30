"""
JWT Bearer Token Authentication for FastAPI
Provides secure API access using JWT tokens with RSA public key verification.
Inte            # Decode and verify the JWT token
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "iat", "userEmail", "scope", "sub"]  # Require 'sub' field for user ID
                }
            )authentication microservice for user validation.
"""

import jwt
import httpx
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime
from pydantic import BaseModel
from app.core.config import settings

logger = logging.getLogger(__name__)

# Security scheme for FastAPI documentation
security = HTTPBearer()

class StandardResponse(BaseModel):
    """Standardized API response format"""
    data: Optional[Any] = None
    success: bool = False
    message: str = ""
    totalRecord: int = 0
    status: int = 200

def create_error_response(status_code: int, message: str, data: Optional[Any] = None) -> JSONResponse:
    """Create standardized error response"""
    response_data = StandardResponse(
        data=data,
        success=False,
        message=message,
        totalRecord=0,
        status=status_code
    )
    return JSONResponse(
        status_code=status_code,
        content=response_data.model_dump()
    )

def create_auth_error_response(message: str) -> JSONResponse:
    """Create standardized authentication error response"""
    return create_error_response(401, message)

class StandardResponse(BaseModel):
    """Standardized API response format"""
    data: Optional[Any] = None
    success: bool = False
    message: str = ""
    totalRecord: int = 0
    status: int = 200

def create_error_response(status_code: int, message: str, data: Optional[Any] = None) -> JSONResponse:
    """Create standardized error response"""
    response_data = StandardResponse(
        data=data,
        success=False,
        message=message,
        totalRecord=0,
        status=status_code
    )
    return JSONResponse(
        status_code=status_code,
        content=response_data.model_dump()
    )

def create_auth_error_response(message: str) -> JSONResponse:
    """Create standardized authentication error response"""
    return create_error_response(401, message)

class JWTTokenValidator:
    """
    JWT Token validator with RSA public key verification and auth microservice integration
    """
    
    def __init__(self):
        self.public_key = None
        self.algorithm = settings.jwt_algorithm
        self.auth_service_url = settings.admin_api_url
        self._load_public_key()
    
    def _load_public_key(self):
        """Load RSA public key from configuration"""
        try:
            if settings.jwt_public_key:
                # Handle both PEM format and base64 encoded keys
                public_key_str = settings.jwt_public_key.strip()
                
                # If it doesn't start with -----BEGIN, assume it's base64 encoded
                if not public_key_str.startswith('-----BEGIN'):
                    import base64
                    try:
                        decoded_key = base64.b64decode(public_key_str).decode('utf-8')
                        public_key_str = decoded_key
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 public key: {e}")
                
                self.public_key = public_key_str
                logger.info("✅ JWT public key loaded successfully")
            else:
                logger.warning("⚠️ No JWT public key configured - token validation will fail")
                
        except Exception as e:
            logger.error(f"❌ Failed to load JWT public key: {e}")
            self.public_key = None
    
    def decode_token(self, token: str) -> Dict:
        """
        Decode and verify JWT token using RSA public key
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid or verification fails
        """
        try:
            if not self.public_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="JWT public key not configured"
                )
            
            # Decode and verify the token
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "iat", "userEmail", "scope"]
                }
            )
            
            logger.debug(f"✅ JWT token decoded successfully for user: {payload.get('userEmail', 'unknown')}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("⚠️ JWT token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT token is expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidSignatureError:
            logger.warning("⚠️ JWT token has invalid signature")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token signature",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.MissingRequiredClaimError as e:
            logger.warning(f"⚠️ JWT token missing required claim: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"JWT token missing required claim: {e}",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"⚠️ Invalid JWT token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid JWT token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error decoding JWT token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token validation error"
            )
    
    async def validate_user_with_auth_service(self, token_payload: Dict, original_token: str) -> Dict:
        """
        Validate user with authentication microservice using user ID from token
        
        Args:
            token_payload: Decoded JWT token payload containing user information
            original_token: Original JWT token string to pass to auth service
            
        Returns:
            User validation response from auth service
            
        Raises:
            HTTPException: If user validation fails
        """
        try:
            # Extract user information from token
            user_email = token_payload.get('userEmail')
            user_id = token_payload.get('sub')  # User ID from 'sub' field
            
            if not user_email:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing user email",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing user ID (sub)",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.debug(f"🔍 Validating user {user_email} (ID: {user_id}) with auth service...")
                
                # Prepare headers exactly as expected by the auth service
                headers = {
                    "X-User-Id": user_id,  # Pass user ID from token's 'sub' field
                    "Authorization": f"Bearer {original_token}"  # Pass original token
                }
                
                # Call the auth service profile endpoint (GET request with headers only)
                response = await client.get(
                    f"{self.auth_service_url}/users/profile",
                    headers=headers
                )
                
                if response.status_code == 200:
                    user_data = response.json()
                    logger.info(f"✅ User {user_email} (ID: {user_id}) validated successfully")
                    return user_data
                elif response.status_code == 401:
                    logger.warning(f"⚠️ User {user_email} (ID: {user_id}) is not authorized")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User not authorized",
                        headers={"WWW-Authenticate": "Bearer"}
                    )
                elif response.status_code == 404:
                    logger.warning(f"⚠️ User {user_email} (ID: {user_id}) not found")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User not found",
                        headers={"WWW-Authenticate": "Bearer"}
                    )
                else:
                    logger.error(f"❌ Auth service returned {response.status_code}: {response.text}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Authentication service error"
                    )
                    
        except httpx.TimeoutException:
            logger.error("❌ Auth service request timed out")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service timeout"
            )
        except httpx.RequestError as e:
            logger.error(f"❌ Auth service request failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error validating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User validation error"
            )
    
    def check_scope_permissions(self, token_payload: Dict, required_scopes: List[str]) -> bool:
        """
        Check if token has required scopes for the operation
        
        Args:
            token_payload: Decoded JWT token payload
            required_scopes: List of required scopes
            
        Returns:
            True if user has required permissions
        """
        try:
            token_scope = token_payload.get('scope', '')
            
            # Parse scope string - handle both space-separated and comma-separated formats
            # e.g., "view:org view:mdm manage:policy" or "manage:policy,manage:mdm"
            if ',' in token_scope:
                # Comma-separated format
                user_scopes = [scope.strip() for scope in token_scope.split(',') if scope.strip()]
            else:
                # Space-separated format (default)
                user_scopes = [scope.strip() for scope in token_scope.split() if scope.strip()]
            
            # Only allow if user has manage:policy scope
            if 'manage:policy' in user_scopes:
                logger.debug(f"✅ User has manage:policy scope - access granted. All scopes: {user_scopes}")
                return True
            
            logger.warning(f"⚠️ User lacks manage:policy scope. Has: {user_scopes}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking scope permissions: {e}")
            return False

# Global token validator instance
token_validator = JWTTokenValidator()

class UserInfo:
    """User information extracted from JWT token"""
    def __init__(self, email: str, user_id: str, scopes: List[str], token_payload: Dict):
        self.email = email
        self.user_id = user_id  # User ID from 'sub' field
        self.scopes = scopes
        self.payload = token_payload
        self.iat = token_payload.get('iat')
        self.exp = token_payload.get('exp')
        self.org_id = token_payload.get('orgId')  # Organization ID if present
        self.user_name = token_payload.get('userName')  # User name if present

async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    required_scopes: List[str] = None
) -> UserInfo:
    """
    FastAPI dependency to verify JWT bearer token and validate user
    
    Args:
        credentials: HTTP authorization credentials from request
        required_scopes: List of required scopes (default: ["manage:policy", "manage:mdm"])
    
    Returns:
        UserInfo object with user details
    
    Raises:
        HTTPException: If token is missing, invalid, or user unauthorized
    """
    try:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Expected 'Bearer'",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = credentials.credentials
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token missing",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Decode and verify the JWT token
        token_payload = token_validator.decode_token(token)
        
        # Extract user information
        user_email = token_payload.get('userEmail')
        user_id = token_payload.get('sub')
        
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user email",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user ID (sub)",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Parse user scopes - handle both space-separated and comma-separated formats
        token_scope = token_payload.get('scope', '')
        if ',' in token_scope:
            # Comma-separated format
            user_scopes = [scope.strip() for scope in token_scope.split(',') if scope.strip()]
        else:
            # Space-separated format (default)
            user_scopes = [scope.strip() for scope in token_scope.split() if scope.strip()]
        
        # Check scope permissions if required_scopes specified
        if required_scopes is None:
            required_scopes = ["manage:policy"]  # Only require manage:policy scope
        
        if required_scopes and not token_validator.check_scope_permissions(token_payload, required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: manage:policy scope",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate user with authentication microservice
        await token_validator.validate_user_with_auth_service(token_payload, token)
        
        logger.info(f"✅ Authentication successful for user: {user_email} (ID: {user_id})")
        
        return UserInfo(
            email=user_email,
            user_id=user_id,
            scopes=user_scopes,
            token_payload=token_payload
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in JWT token verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

# Convenience dependencies for specific scopes
async def verify_policy_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInfo:
    """Verify token with manage:policy scope only"""
    return await verify_jwt_token(credentials, required_scopes=["manage:policy"])

async def verify_mdm_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInfo:
    """Verify token with manage:policy scope only"""
    return await verify_jwt_token(credentials, required_scopes=["manage:policy"])

async def verify_any_scope_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInfo:
    """Verify token with manage:policy scope only"""
    return await verify_jwt_token(credentials, required_scopes=["manage:policy"])

# Optional dependency for endpoints that can work with or without authentication
async def optional_jwt_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[UserInfo]:
    """
    Optional JWT token verification for endpoints that can work with or without auth
    
    Returns:
        UserInfo object if valid token provided, None if no token provided
    """
    try:
        if not credentials or not credentials.credentials:
            return None
        
        return await verify_jwt_token(credentials, required_scopes=[])
        
    except HTTPException:
        # Return None for optional auth instead of raising exception
        return None
    except Exception as e:
        logger.error(f"❌ Error in optional JWT token verification: {e}")
        return None

def get_token_validator() -> JWTTokenValidator:
    """Get the global token validator instance"""
    return token_validator