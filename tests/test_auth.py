"""
Consolidated Comprehensive JWT Authentication Test Suite
Combines test_auth.py and test_auth_enhanced.py into a single optimized test file

This unified test suite provides:
- Core JWT authentication functionality testing
- Enhanced edge cases and error condition testing  
- Performance and concurrency testing
- Integration testing with FastAPI endpoints
- Shared fixtures for optimal test performance
- Manual test functions for debugging

Run with: pytest tests/test_auth_consolidated.py -v
"""

import os
import sys
import json
import base64
import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from concurrent.futures import ThreadPoolExecutor

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Setup test environment first
os.environ["ENVIRONMENT"] = "test"
os.environ["TESTING"] = "true"
os.environ["DISABLE_EXTERNAL_CALLS"] = "true"

# Now import the modules
try:
    import pytest
    import jwt
    import httpx
    from fastapi import HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials
    from fastapi.testclient import TestClient
    
    # Import auth components
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
        StandardResponse,
        get_token_validator,
        get_user_info,
        security
    )
    from app.main import app
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports not available: {e}")
    IMPORTS_AVAILABLE = False


# ==========================================================================
# SHARED PYTEST FIXTURES FOR PERFORMANCE OPTIMIZATION
# ==========================================================================

@pytest.fixture(scope="module")
def test_client():
    """Module-level TestClient to avoid repeated app initialization"""
    if IMPORTS_AVAILABLE:
        return TestClient(app)
    return None

@pytest.fixture(scope="module")
def mock_settings():
    """Module-level mock settings for consistent test environment"""
    with patch('app.auth.bearer.settings') as mock:
        mock.jwt_algorithm = "RS256"
        mock.admin_api_url = "http://test-auth-service.com"
        mock.jwt_public_key = "-----BEGIN PUBLIC KEY-----\ntest_key\n-----END PUBLIC KEY-----"
        yield mock

@pytest.fixture
def base_validator(mock_settings):
    """Create JWTTokenValidator instance for testing with shared settings"""
    return JWTTokenValidator()

@pytest.fixture
def sample_token_payload():
    """Standard token payload for testing"""
    return {
        "userEmail": "test@example.com",
        "sub": "user123",
        "scope": "manage:policy view:org",
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "userName": "Test User",
        "orgId": "test-org"
    }

@pytest.fixture
def valid_credentials():
    """Standard credentials for testing"""
    if IMPORTS_AVAILABLE:
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
    return None


# ==========================================================================
# CORE FUNCTIONALITY TESTS
# ==========================================================================

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
    
    def test_create_error_response(self):
        """Test create_error_response function"""
        response = create_error_response(404, "Not found", {"error": "details"})
        
        assert response.status_code == 404
        content = json.loads(response.body)
        assert content["success"] is False
        assert content["message"] == "Not found"
        assert content["data"] == {"error": "details"}

    def test_create_auth_error_response(self):
        """Test create_auth_error_response function"""
        response = create_auth_error_response("Token expired")
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["success"] is False
        assert content["message"] == "Token expired"
        assert content["status"] == 401

    def test_standard_response_with_defaults(self):
        """Test StandardResponse with default values"""
        response = StandardResponse(message="Test")
        assert response.success is False
        assert response.data is None
        assert response.totalRecord == 0
        assert response.status == 200


class TestJWTTokenValidator:
    """Test JWTTokenValidator class core functionality"""
    
    def test_validator_initialization(self, base_validator, mock_settings):
        """Test validator initialization"""
        assert base_validator.algorithm == "RS256"
        assert base_validator.auth_service_url == "http://test-auth-service.com"
        assert base_validator.public_key is not None
    
    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_success(self, mock_jwt_decode, base_validator, sample_token_payload):
        """Test successful token decoding"""
        mock_jwt_decode.return_value = sample_token_payload
        
        result = base_validator.decode_token("valid_token")
        
        assert result == sample_token_payload
        mock_jwt_decode.assert_called_once()
    
    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_expired(self, mock_jwt_decode, base_validator):
        """Test token decoding with expired token"""
        mock_jwt_decode.side_effect = jwt.ExpiredSignatureError("Token expired")
        
        with pytest.raises(HTTPException) as exc_info:
            base_validator.decode_token("expired_token")
        
        assert exc_info.value.status_code == 401
        assert "JWT token is expired" in exc_info.value.detail

    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_invalid_signature(self, mock_jwt_decode, base_validator):
        """Test handling of invalid signature error"""
        mock_jwt_decode.side_effect = jwt.InvalidSignatureError("Invalid signature")
        
        with pytest.raises(HTTPException) as exc_info:
            base_validator.decode_token("invalid_signature_token")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token signature" in str(exc_info.value.detail)

    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_missing_claim(self, mock_jwt_decode, base_validator):
        """Test handling of missing required claim error"""
        mock_jwt_decode.side_effect = jwt.MissingRequiredClaimError("Missing 'exp' claim")
        
        with pytest.raises(HTTPException) as exc_info:
            base_validator.decode_token("token_missing_claim")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "JWT token missing required claim" in str(exc_info.value.detail)


class TestScopeValidation:
    """Test scope-based authorization"""
    
    def test_check_scope_permissions_space_separated(self, base_validator):
        """Test scope checking with space-separated scopes"""
        token_payload = {"scope": "view:org manage:policy view:mdm"}
        
        result = base_validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is True
    
    def test_check_scope_permissions_insufficient(self, base_validator):
        """Test scope checking with insufficient permissions"""
        token_payload = {"scope": "view:org view:mdm"}
        
        result = base_validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False
    
    @pytest.mark.parametrize("scope_format,required,expected", [
        ("view:org manage:policy view:mdm", ["manage:policy"], True),
        ("view:org,manage:policy,view:mdm", ["manage:policy"], True),
        ("view:org view:mdm", ["manage:policy"], False),
        ("", ["manage:policy"], False),
        ("   ", False, False),  # Whitespace only
        ("manage:policy, view:org, view:mdm", ["manage:policy"], True),  # Comma with spaces
    ])
    def test_scope_validation_scenarios(self, scope_format, required, expected, base_validator):
        """Test various scope validation scenarios"""
        token_payload = {"scope": scope_format} if scope_format else {}
        
        result = base_validator.check_scope_permissions(token_payload, required)
        assert result == expected

    def test_check_scope_permissions_empty_scope(self, base_validator):
        """Test scope checking with empty scope string"""
        token_payload = {"scope": ""}
        result = base_validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False

    def test_check_scope_permissions_missing_scope(self, base_validator):
        """Test scope checking with missing scope field"""
        token_payload = {}
        result = base_validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False

    def test_check_scope_permissions_exception_handling(self, base_validator):
        """Test scope checking exception handling"""
        # Mock a scenario where payload access causes an exception
        token_payload = Mock()
        token_payload.get.side_effect = Exception("Unexpected error")
        
        result = base_validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False


class TestUserInfo:
    """Test UserInfo class and user information extraction"""
    
    def test_user_info_creation(self, sample_token_payload):
        """Test UserInfo object creation"""
        user_info = UserInfo(
            email="test@example.com",
            user_id="user123",
            scopes=["manage:policy"],
            token_payload=sample_token_payload
        )
        
        assert user_info.email == "test@example.com"
        assert user_info.user_id == "user123"
        assert user_info.scopes == ["manage:policy"]
        assert user_info.payload == sample_token_payload

    def test_user_info_with_all_optional_fields(self, sample_token_payload):
        """Test UserInfo creation with all optional fields"""
        user_info = UserInfo(
            email="test@example.com",
            user_id="user123",
            scopes=["manage:policy", "view:org"],
            token_payload=sample_token_payload
        )
        
        assert user_info.iat == sample_token_payload["iat"]
        assert user_info.exp == sample_token_payload["exp"]
        assert user_info.org_id == sample_token_payload["orgId"]
        assert user_info.user_name == sample_token_payload["userName"]

    def test_user_info_with_missing_optional_fields(self):
        """Test UserInfo creation with missing optional fields"""
        minimal_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        
        user_info = UserInfo(
            email="test@example.com",
            user_id="user123",
            scopes=["manage:policy"],
            token_payload=minimal_payload
        )
        
        assert user_info.iat is None
        assert user_info.exp is None
        assert user_info.org_id is None
        assert user_info.user_name is None


# ==========================================================================
# VERIFICATION FUNCTIONS TESTS
# ==========================================================================

class TestVerificationFunctions:
    """Test JWT verification dependency functions"""
    
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
    async def test_verify_jwt_token_empty_token(self):
        """Test verification with empty token string"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
        
        with pytest.raises(HTTPException) as exc_info:
            await verify_jwt_token(credentials)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Bearer token missing" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_verify_jwt_token_whitespace_token(self):
        """Test verification with whitespace-only token"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="   ")
        
        # Should pass the empty check but fail in JWT decoding
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode:
            mock_decode.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid JWT token"
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_jwt_token(credentials)
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_verify_jwt_token_with_custom_scopes(self, valid_credentials):
        """Test JWT token verification with custom required scopes"""
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode, \
             patch('app.auth.bearer.token_validator.validate_user_with_auth_service') as mock_validate:
            
            mock_decode.return_value = {
                "userEmail": "test@example.com",
                "sub": "user123",
                "scope": "view:org manage:policy"
            }
            mock_validate.return_value = {"status": "valid"}
            
            # Test with custom scopes that user has
            result = await verify_jwt_token(valid_credentials, required_scopes=["manage:policy"])
            assert result.email == "test@example.com"
            
            # Test with custom scopes that user doesn't have
            with pytest.raises(HTTPException) as exc_info:
                await verify_jwt_token(valid_credentials, required_scopes=["admin:system"])
            
            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_verify_policy_token(self, valid_credentials):
        """Test verify_policy_token function"""
        with patch('app.auth.bearer.verify_jwt_token', new_callable=AsyncMock) as mock_verify:
            mock_user_info = Mock()
            mock_verify.return_value = mock_user_info
            
            result = await verify_policy_token(valid_credentials)
            
            assert result == mock_user_info
            mock_verify.assert_called_once_with(valid_credentials, required_scopes=["manage:policy"])

    @pytest.mark.asyncio
    async def test_optional_jwt_token_with_none_credentials(self):
        """Test optional JWT token with None credentials"""
        result = await optional_jwt_token(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_optional_jwt_token_with_empty_credentials(self):
        """Test optional JWT token with empty credentials object"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
        result = await optional_jwt_token(credentials)
        assert result is None

    @pytest.mark.asyncio
    async def test_optional_jwt_token_with_invalid_token(self):
        """Test optional JWT token with invalid token (should return None, not raise)"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")
        
        with patch('app.auth.bearer.verify_jwt_token') as mock_verify:
            mock_verify.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
            result = await optional_jwt_token(credentials)
            assert result is None


# ==========================================================================
# AUTH SERVICE INTEGRATION TESTS
# ==========================================================================

class TestAuthServiceIntegration:
    """Test integration with external auth service"""
    
    @pytest.mark.asyncio
    async def test_validate_user_success(self, base_validator):
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
            
            result = await base_validator.validate_user_with_auth_service(token_payload, "original_token")
            
            assert result == {"user": "data"}
    
    @pytest.mark.asyncio
    async def test_validate_user_missing_email(self, base_validator):
        """Test user validation with missing email"""
        token_payload = {"sub": "user123"}
        
        with pytest.raises(HTTPException) as exc_info:
            await base_validator.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == 401
        assert "Token missing user email" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_validate_user_missing_user_id(self, base_validator):
        """Test user validation with missing user ID (sub field)"""
        token_payload = {"userEmail": "test@example.com", "scope": "manage:policy"}
        
        with pytest.raises(HTTPException) as exc_info:
            await base_validator.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Token missing user ID (sub)" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_validate_user_auth_service_404(self, mock_get, base_validator):
        """Test user validation when auth service returns 404"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        token_payload = {
            "userEmail": "notfound@example.com",
            "sub": "user404",
            "scope": "manage:policy"
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await base_validator.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "User not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_validate_user_timeout(self, mock_get, base_validator):
        """Test user validation with auth service timeout"""
        mock_get.side_effect = httpx.TimeoutException("Request timeout")
        
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await base_validator.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Authentication service timeout" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_auth_service_integration_headers_validation(self, base_validator):
        """Test that correct headers are sent to auth service"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        original_token = "bearer-token-123"
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"profile": "data"}
            mock_get.return_value = mock_response
            
            await base_validator.validate_user_with_auth_service(token_payload, original_token)
            
            # Verify correct headers were sent
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            headers = call_args.kwargs['headers']
            
            assert headers["X-User-Id"] == "user123"
            assert headers["Authorization"] == f"Bearer {original_token}"


# ==========================================================================
# ENHANCED EDGE CASES AND ERROR CONDITIONS
# ==========================================================================

class TestEnhancedValidatorEdgeCases:
    """Enhanced tests for JWTTokenValidator edge cases"""

    @pytest.fixture
    def validator_no_key(self):
        """Validator with no public key configured"""
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = None
            mock_settings.jwt_algorithm = "RS256"
            mock_settings.admin_api_url = "http://test-auth-service"
            return JWTTokenValidator()

    @pytest.fixture
    def validator_base64_key(self):
        """Validator with base64 encoded public key"""
        with patch('app.auth.bearer.settings') as mock_settings:
            # Simulate base64 encoded key (without PEM headers)
            mock_settings.jwt_public_key = "LS0tLS1CRUdJTiBQVUJMSUMgS0VZLS0tLS0KTUlJQklqQU5CZ2txaGtpRzl3MEJBUUVGQUFPQ0FROEFNSUlCQ2dLQ0FRRUF0dz"
            mock_settings.jwt_algorithm = "RS256"
            mock_settings.admin_api_url = "http://test-auth-service"
            return JWTTokenValidator()

    def test_public_key_loading_base64_format(self, validator_base64_key):
        """Test loading base64 encoded public key"""
        assert validator_base64_key.public_key is not None
        assert validator_base64_key.public_key.startswith("-----BEGIN PUBLIC KEY-----")
        assert validator_base64_key.public_key.endswith("-----END PUBLIC KEY-----")

    def test_decode_token_no_public_key(self, validator_no_key):
        """Test token decoding when no public key is configured"""
        with pytest.raises(HTTPException) as exc_info:
            validator_no_key.decode_token("dummy.token.here")
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Token validation error" in str(exc_info.value.detail)


class TestConcurrencyAndPerformance:
    """Tests for concurrent access and performance scenarios"""

    @pytest.mark.asyncio
    async def test_concurrent_token_validation(self, valid_credentials):
        """Test concurrent token validation to ensure thread safety"""
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode, \
             patch('app.auth.bearer.token_validator.validate_user_with_auth_service') as mock_validate:
            
            mock_decode.return_value = {
                "userEmail": "test@example.com",
                "sub": "user123",
                "scope": "manage:policy"
            }
            mock_validate.return_value = {"status": "valid"}
            
            # Run multiple concurrent validations with reduced count for performance
            tasks = [verify_jwt_token(valid_credentials) for _ in range(3)]
            results = await asyncio.gather(*tasks)
            
            # All should succeed and return the same user
            assert len(results) == 3
            assert all(result.email == "test@example.com" for result in results)

    @pytest.mark.asyncio
    async def test_token_validator_singleton_behavior(self):
        """Test that token validator behaves correctly as singleton"""
        validator1 = get_token_validator()
        validator2 = get_token_validator()
        
        # Should be the same instance
        assert validator1 is validator2

    def test_token_validator_singleton_thread_safety(self):
        """Test that token validator singleton is thread-safe"""
        validators = []
        
        def get_validator():
            validators.append(get_token_validator())
        
        # Create multiple threads that get the validator
        threads = [threading.Thread(target=get_validator) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All should be the same instance
        assert len(validators) == 3
        assert all(v is validators[0] for v in validators)


# ==========================================================================
# FASTAPI INTEGRATION TESTS
# ==========================================================================

class TestJWTAuthenticationIntegration:
    """Test JWT token authentication integration with FastAPI endpoints"""
    
    def test_missing_token(self, test_client):
        """Test authentication failure with missing token"""
        response = test_client.post("/api/aips/suggest-rules", 
                                   json={"domain": "test_domain"})
        assert response.status_code in [401, 403]
        data = response.json()
        assert data["success"] is False
    
    def test_invalid_token_format(self, test_client):
        """Test authentication failure with invalid token format"""
        headers = {"Authorization": "InvalidToken"}
        response = test_client.post("/api/aips/suggest-rules", 
                                   json={"domain": "test_domain"}, 
                                   headers=headers)
        assert response.status_code in [401, 403]


class TestLiveAPIEndpoints:
    """Test live API endpoints for JWT authentication functionality"""
    
    @pytest.mark.asyncio
    async def test_api_endpoint_no_auth(self):
        """Test API endpoint without authorization (should return 401/403)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8092/api/aips/suggest-rules",
                    json={},
                    timeout=2.0  # Reduced timeout for faster tests
                )
                assert response.status_code in [401, 403]
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Server not available for live testing")


# ==========================================================================
# CONVENIENCE FUNCTION TESTS
# ==========================================================================

class TestGetUserInfoFunction:
    """Tests for the get_user_info compatibility function"""

    @pytest.mark.asyncio
    async def test_get_user_info_success(self):
        """Test successful user info retrieval"""
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode, \
             patch('app.auth.bearer.token_validator.validate_user_with_auth_service') as mock_validate:
            
            mock_decode.return_value = {
                "userEmail": "test@example.com",
                "sub": "user123",
                "scope": "manage:policy view:org"
            }
            mock_validate.return_value = {"profile": "data"}
            
            result = await get_user_info("test-token")
            
            assert result["email"] == "test@example.com"
            assert result["user_id"] == "user123"
            assert result["scopes"] == ["manage:policy", "view:org"]
            assert result["user_data"] == {"profile": "data"}

    @pytest.mark.asyncio
    async def test_convenience_auth_functions(self, valid_credentials):
        """Test convenience authentication functions"""
        with patch('app.auth.bearer.verify_jwt_token') as mock_verify:
            mock_user = UserInfo(
                email="test@example.com",
                user_id="user123",
                scopes=["manage:policy"],
                token_payload={"userEmail": "test@example.com", "sub": "user123"}
            )
            mock_verify.return_value = mock_user
            
            # Test verify_policy_token
            result = await verify_policy_token(valid_credentials)
            assert result.email == "test@example.com"
            mock_verify.assert_called_with(valid_credentials, required_scopes=["manage:policy"])
            
            # Test verify_mdm_token  
            result = await verify_mdm_token(valid_credentials)
            assert result.email == "test@example.com"
            
            # Test verify_any_scope_token
            result = await verify_any_scope_token(valid_credentials)
            assert result.email == "test@example.com"


# ==========================================================================
# MANUAL TEST FUNCTIONS (For debugging without pytest)
# ==========================================================================

def manual_test_user_info_creation():
    """Manual test for UserInfo object creation"""
    print("Testing UserInfo creation...")
    
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
    print("✅ UserInfo creation test passed")


def manual_test_standard_response():
    """Manual test for StandardResponse"""
    print("Testing StandardResponse...")
    
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
    print("✅ StandardResponse test passed")


# ==========================================================================
# UTILITY FUNCTIONS
# ==========================================================================

def create_mock_token_payload(email="test@example.com", user_id="user123", scope="manage:policy"):
    """Create mock token payload for testing"""
    return {
        "userEmail": email,
        "sub": user_id,
        "scope": scope,
        "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "userName": "Test User",
        "orgId": "test-org"
    }


async def run_manual_tests():
    """Run all manual tests for debugging"""
    print("🧪 Running consolidated JWT authentication tests...")
    print("=" * 60)
    
    try:
        manual_test_standard_response()
        manual_test_user_info_creation()
        
        print("=" * 60)
        print("✅ All manual tests completed successfully!")
        print("🔍 This consolidated file combines:")
        print("   - Core JWT authentication functionality")
        print("   - Enhanced edge cases and error conditions")
        print("   - Performance and concurrency testing")
        print("   - FastAPI integration testing")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if IMPORTS_AVAILABLE:
        print("✅ All imports available - can run with pytest")
        print("Run with: pytest tests/test_auth_consolidated.py -v")
    else:
        print("⚠️ Some imports not available - running manual tests only")
    
    # Run manual tests
    import asyncio
    asyncio.run(run_manual_tests())