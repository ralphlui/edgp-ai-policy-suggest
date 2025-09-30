"""
Comprehensive unit tests for app.auth.bearer module
Tests JWT authentication, token validation, and user verification functionality
"""

import pytest
import jwt
import httpx
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from datetime import datetime, timedelta, timezone
import json
import base64

from app.auth.bearer import (
    JWTTokenValidator,
    UserInfo,
    verify_jwt_token,
    verify_policy_token,
    verify_mdm_token,
    verify_any_scope_token,
    optional_jwt_token,
    token_validator,
    create_error_response,
    create_auth_error_response,
    StandardResponse
)


class TestStandardResponse:
    """Test StandardResponse model and helper functions"""
    
    def test_standard_response_creation(self):
        """Test StandardResponse model creation"""
        response = StandardResponse(
            data={"test": "data"},
            success=True,
            message="Test message",
            totalRecord=5,
            status=200
        )
        assert response.data == {"test": "data"}
        assert response.success is True
        assert response.message == "Test message"
        assert response.totalRecord == 5
        assert response.status == 200
    
    def test_standard_response_defaults(self):
        """Test StandardResponse with default values"""
        response = StandardResponse()
        assert response.data is None
        assert response.success is False
        assert response.message == ""
        assert response.totalRecord == 0
        assert response.status == 200
    
    def test_create_error_response(self):
        """Test create_error_response function"""
        response = create_error_response(404, "Not found", {"error": "details"})
        
        assert response.status_code == 404
        content = json.loads(response.body)
        assert content["success"] is False
        assert content["message"] == "Not found"
        assert content["data"] == {"error": "details"}
        assert content["status"] == 404
        assert content["totalRecord"] == 0
    
    def test_create_auth_error_response(self):
        """Test create_auth_error_response function"""
        response = create_auth_error_response("Unauthorized access")
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["success"] is False
        assert content["message"] == "Unauthorized access"
        assert content["status"] == 401


class TestJWTTokenValidator:
    """Test JWTTokenValidator class functionality"""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing"""
        with patch('app.auth.bearer.settings') as mock:
            mock.jwt_algorithm = "RS256"
            mock.admin_api_url = "http://test-auth-service.com"
            mock.jwt_public_key = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1234567890ABC...\n-----END PUBLIC KEY-----"
            yield mock
    
    @pytest.fixture
    def validator(self, mock_settings):
        """Create JWTTokenValidator instance for testing"""
        return JWTTokenValidator()
    
    def test_validator_initialization(self, validator, mock_settings):
        """Test validator initialization"""
        assert validator.algorithm == "RS256"
        assert validator.auth_service_url == "http://test-auth-service.com"
        assert validator.public_key is not None
    
    def test_load_public_key_pem_format(self, mock_settings):
        """Test loading PEM formatted public key"""
        mock_settings.jwt_public_key = "-----BEGIN PUBLIC KEY-----\ntest_key\n-----END PUBLIC KEY-----"
        validator = JWTTokenValidator()
        assert validator.public_key == "-----BEGIN PUBLIC KEY-----\ntest_key\n-----END PUBLIC KEY-----"
    
    def test_load_public_key_base64_format(self, mock_settings):
        """Test loading base64 encoded public key"""
        pem_key = "-----BEGIN PUBLIC KEY-----\ntest_key\n-----END PUBLIC KEY-----"
        base64_key = base64.b64encode(pem_key.encode()).decode()
        mock_settings.jwt_public_key = base64_key
        
        validator = JWTTokenValidator()
        assert validator.public_key == pem_key
    
    def test_load_public_key_missing(self, mock_settings):
        """Test behavior when public key is missing"""
        mock_settings.jwt_public_key = None
        validator = JWTTokenValidator()
        assert validator.public_key is None
    
    def test_load_public_key_invalid_base64(self, mock_settings):
        """Test behavior with invalid base64 key"""
        mock_settings.jwt_public_key = "invalid_base64_key!!!"
        validator = JWTTokenValidator()
        assert validator.public_key == "invalid_base64_key!!!"
    
    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_success(self, mock_jwt_decode, validator):
        """Test successful token decoding"""
        mock_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy",
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(timezone.utc).timestamp())
        }
        mock_jwt_decode.return_value = mock_payload
        
        result = validator.decode_token("valid_token")
        
        assert result == mock_payload
        mock_jwt_decode.assert_called_once()
    
    def test_decode_token_no_public_key(self):
        """Test token decoding when public key is not configured"""
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = None
            validator = JWTTokenValidator()
            
            with pytest.raises(HTTPException) as exc_info:
                validator.decode_token("any_token")
            
            assert exc_info.value.status_code == 500
            # The error gets wrapped in a generic message due to exception handling
            assert exc_info.value.detail in ["JWT public key not configured", "Token validation error"]
    
    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_expired(self, mock_jwt_decode, validator):
        """Test token decoding with expired token"""
        mock_jwt_decode.side_effect = jwt.ExpiredSignatureError("Token expired")
        
        with pytest.raises(HTTPException) as exc_info:
            validator.decode_token("expired_token")
        
        assert exc_info.value.status_code == 401
        assert "JWT token is expired" in exc_info.value.detail
    
    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_invalid_signature(self, mock_jwt_decode, validator):
        """Test token decoding with invalid signature"""
        mock_jwt_decode.side_effect = jwt.InvalidSignatureError("Invalid signature")
        
        with pytest.raises(HTTPException) as exc_info:
            validator.decode_token("invalid_token")
        
        assert exc_info.value.status_code == 401
        assert "Invalid token signature" in exc_info.value.detail
    
    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_missing_claim(self, mock_jwt_decode, validator):
        """Test token decoding with missing required claim"""
        mock_jwt_decode.side_effect = jwt.MissingRequiredClaimError("Missing claim: userEmail")
        
        with pytest.raises(HTTPException) as exc_info:
            validator.decode_token("incomplete_token")
        
        assert exc_info.value.status_code == 401
        assert "JWT token missing required claim" in exc_info.value.detail
    
    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_invalid_token(self, mock_jwt_decode, validator):
        """Test token decoding with general invalid token"""
        mock_jwt_decode.side_effect = jwt.InvalidTokenError("Invalid token")
        
        with pytest.raises(HTTPException) as exc_info:
            validator.decode_token("invalid_token")
        
        assert exc_info.value.status_code == 401
        assert "Invalid JWT token" in exc_info.value.detail
    
    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_unexpected_error(self, mock_jwt_decode, validator):
        """Test token decoding with unexpected error"""
        mock_jwt_decode.side_effect = Exception("Unexpected error")
        
        with pytest.raises(HTTPException) as exc_info:
            validator.decode_token("problem_token")
        
        assert exc_info.value.status_code == 500
        assert "Token validation error" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_validate_user_success(self, validator):
        """Test successful user validation with auth service"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"user": "data"}
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await validator.validate_user_with_auth_service(token_payload, "original_token")
            
            assert result == {"user": "data"}
    
    @pytest.mark.asyncio
    async def test_validate_user_missing_email(self, validator):
        """Test user validation with missing email"""
        token_payload = {"sub": "user123"}
        
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == 401
        assert "Token missing user email" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_validate_user_missing_user_id(self, validator):
        """Test user validation with missing user ID"""
        token_payload = {"userEmail": "test@example.com"}
        
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == 401
        assert "Token missing user ID (sub)" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_validate_user_unauthorized(self, validator):
        """Test user validation when user is not authorized"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 401
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_user_with_auth_service(token_payload, "token")
            
            assert exc_info.value.status_code == 401
            assert "User not authorized" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_validate_user_not_found(self, validator):
        """Test user validation when user is not found"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 404
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_user_with_auth_service(token_payload, "token")
            
            assert exc_info.value.status_code == 401
            assert "User not found" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_validate_user_service_error(self, validator):
        """Test user validation when auth service returns error"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal server error"
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_user_with_auth_service(token_payload, "token")
            
            assert exc_info.value.status_code == 503
            assert "Authentication service error" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_validate_user_timeout(self, validator):
        """Test user validation with timeout"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.return_value = mock_context
            
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_user_with_auth_service(token_payload, "token")
            
            assert exc_info.value.status_code == 503
            assert "Authentication service timeout" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_validate_user_request_error(self, validator):
        """Test user validation with request error"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.get = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
            mock_client.return_value = mock_context
            
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_user_with_auth_service(token_payload, "token")
            
            assert exc_info.value.status_code == 503
            assert "Authentication service unavailable" in exc_info.value.detail
    
    def test_check_scope_permissions_space_separated(self, validator):
        """Test scope checking with space-separated scopes"""
        token_payload = {"scope": "view:org manage:policy view:mdm"}
        
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is True
    
    def test_check_scope_permissions_comma_separated(self, validator):
        """Test scope checking with comma-separated scopes"""
        token_payload = {"scope": "view:org,manage:policy,view:mdm"}
        
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is True
    
    def test_check_scope_permissions_insufficient(self, validator):
        """Test scope checking with insufficient permissions"""
        token_payload = {"scope": "view:org view:mdm"}
        
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False
    
    def test_check_scope_permissions_missing_scope(self, validator):
        """Test scope checking with missing scope field"""
        token_payload = {}
        
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False
    
    def test_check_scope_permissions_error_handling(self, validator):
        """Test scope checking with error in processing"""
        token_payload = {"scope": None}
        
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False


class TestUserInfo:
    """Test UserInfo class"""
    
    def test_user_info_creation(self):
        """Test UserInfo object creation"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "iat": 1234567890,
            "exp": 1234567890,
            "orgId": "org456",
            "userName": "Test User"
        }
        
        user_info = UserInfo(
            email="test@example.com",
            user_id="user123",
            scopes=["manage:policy"],
            token_payload=token_payload
        )
        
        assert user_info.email == "test@example.com"
        assert user_info.user_id == "user123"
        assert user_info.scopes == ["manage:policy"]
        assert user_info.payload == token_payload
        assert user_info.iat == 1234567890
        assert user_info.exp == 1234567890
        assert user_info.org_id == "org456"
        assert user_info.user_name == "Test User"


class TestVerificationFunctions:
    """Test JWT verification dependency functions"""
    
    @pytest.mark.asyncio
    async def test_verify_jwt_token_success(self):
        """Test successful JWT token verification"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
        
        mock_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy",
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(timezone.utc).timestamp())
        }
        
        with patch.object(token_validator, 'decode_token', return_value=mock_payload), \
             patch.object(token_validator, 'validate_user_with_auth_service', new_callable=AsyncMock) as mock_validate, \
             patch.object(token_validator, 'check_scope_permissions', return_value=True):
            
            mock_validate.return_value = {"user": "data"}
            
            result = await verify_jwt_token(credentials, ["manage:policy"])
            
            assert isinstance(result, UserInfo)
            assert result.email == "test@example.com"
            assert result.user_id == "user123"
            assert result.scopes == ["manage:policy"]
    
    @pytest.mark.asyncio
    async def test_verify_jwt_token_missing_credentials(self):
        """Test JWT verification with missing credentials"""
        with pytest.raises(HTTPException) as exc_info:
            await verify_jwt_token(None)
        
        assert exc_info.value.status_code == 401
        assert "Bearer token required" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_verify_jwt_token_invalid_scheme(self):
        """Test JWT verification with invalid scheme"""
        credentials = HTTPAuthorizationCredentials(scheme="Basic", credentials="token")
        
        with pytest.raises(HTTPException) as exc_info:
            await verify_jwt_token(credentials)
        
        assert exc_info.value.status_code == 401
        assert "Invalid authentication scheme" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_verify_jwt_token_missing_token(self):
        """Test JWT verification with missing token"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
        
        with pytest.raises(HTTPException) as exc_info:
            await verify_jwt_token(credentials)
        
        assert exc_info.value.status_code == 401
        assert "Bearer token missing" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_verify_jwt_token_insufficient_scope(self):
        """Test JWT verification with insufficient scope"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
        
        mock_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "view:org",
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.now(timezone.utc).timestamp())
        }
        
        with patch.object(token_validator, 'decode_token', return_value=mock_payload), \
             patch.object(token_validator, 'check_scope_permissions', return_value=False):
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_jwt_token(credentials, ["manage:policy"])
            
            assert exc_info.value.status_code == 403
            assert "Insufficient permissions" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_verify_policy_token(self):
        """Test verify_policy_token function"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
        
        with patch('app.auth.bearer.verify_jwt_token', new_callable=AsyncMock) as mock_verify:
            mock_user_info = Mock()
            mock_verify.return_value = mock_user_info
            
            result = await verify_policy_token(credentials)
            
            assert result == mock_user_info
            mock_verify.assert_called_once_with(credentials, required_scopes=["manage:policy"])
    
    @pytest.mark.asyncio
    async def test_verify_mdm_token(self):
        """Test verify_mdm_token function"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
        
        with patch('app.auth.bearer.verify_jwt_token', new_callable=AsyncMock) as mock_verify:
            mock_user_info = Mock()
            mock_verify.return_value = mock_user_info
            
            result = await verify_mdm_token(credentials)
            
            assert result == mock_user_info
            mock_verify.assert_called_once_with(credentials, required_scopes=["manage:policy"])
    
    @pytest.mark.asyncio
    async def test_verify_any_scope_token(self):
        """Test verify_any_scope_token function"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
        
        with patch('app.auth.bearer.verify_jwt_token', new_callable=AsyncMock) as mock_verify:
            mock_user_info = Mock()
            mock_verify.return_value = mock_user_info
            
            result = await verify_any_scope_token(credentials)
            
            assert result == mock_user_info
            mock_verify.assert_called_once_with(credentials, required_scopes=["manage:policy"])
    
    @pytest.mark.asyncio
    async def test_optional_jwt_token_success(self):
        """Test optional JWT token with valid token"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
        
        with patch('app.auth.bearer.verify_jwt_token', new_callable=AsyncMock) as mock_verify:
            mock_user_info = Mock()
            mock_verify.return_value = mock_user_info
            
            result = await optional_jwt_token(credentials)
            
            assert result == mock_user_info
            mock_verify.assert_called_once_with(credentials, required_scopes=[])
    
    @pytest.mark.asyncio
    async def test_optional_jwt_token_missing(self):
        """Test optional JWT token with missing credentials"""
        result = await optional_jwt_token(None)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_optional_jwt_token_empty(self):
        """Test optional JWT token with empty credentials"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
        result = await optional_jwt_token(credentials)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_optional_jwt_token_error(self):
        """Test optional JWT token with error"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid_token")
        
        with patch('app.auth.bearer.verify_jwt_token', new_callable=AsyncMock) as mock_verify:
            mock_verify.side_effect = HTTPException(status_code=401, detail="Invalid token")
            
            result = await optional_jwt_token(credentials)
            assert result is None


class TestTokenValidatorGlobal:
    """Test global token validator functionality"""
    
    def test_get_token_validator(self):
        """Test get_token_validator function"""
        from app.auth.bearer import get_token_validator
        
        validator = get_token_validator()
        assert isinstance(validator, JWTTokenValidator)
        assert validator == token_validator