"""
Consolidated comprehensive tests for JWT authentication and authorization
Combined from test_auth_bearer.py, test_jwt_auth.py, and test_jwt_fix.py

This file consolidates all JWT authentication tests into a single comprehensive suite:
- Unit tests for JWTTokenValidator class
- Integration tests for auth service communication
- Scope validation and authorization tests
- FastAPI endpoint authentication tests
- Live API testing capabilities
- Parametrized tests for comprehensive coverage

Run with: python tests/run_tests.py --pattern "test_auth"
"""

import os
import sys
import json
import base64
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Setup test environment first
os.environ["ENVIRONMENT"] = "test"
os.environ["TESTING"] = "true"
os.environ["USE_AWS_SECRETS"] = "false"
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
        get_token_validator
    )
    from app.main import app
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports not available: {e}")
    IMPORTS_AVAILABLE = False


# ==========================================================================
# MANUAL TEST FUNCTIONS (Can run without pytest)
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


def manual_test_jwt_validator_initialization():
    """Manual test for JWT validator initialization"""
    print("Testing JWT validator initialization...")
    
    with patch('app.auth.bearer.settings') as mock_settings:
        mock_settings.jwt_algorithm = "RS256"
        mock_settings.admin_api_url = "http://test-auth-service.com"
        mock_settings.jwt_public_key = "test_key"
        
        validator = JWTTokenValidator()
        assert validator.algorithm == "RS256"
        assert validator.auth_service_url == "http://test-auth-service.com"
        assert validator.public_key is not None
        print("✅ JWT validator initialization test passed")


def manual_test_scope_validation():
    """Manual test for scope validation"""
    print("Testing scope validation...")
    
    validator = JWTTokenValidator()
    
    # Test space-separated scopes
    token_payload = {"scope": "view:org manage:policy view:mdm"}
    result = validator.check_scope_permissions(token_payload, ["manage:policy"])
    assert result is True
    
    # Test insufficient permissions
    token_payload = {"scope": "view:org view:mdm"}
    result = validator.check_scope_permissions(token_payload, ["manage:policy"])
    assert result is False
    
    print("✅ Scope validation test passed")


def manual_test_token_validator_global():
    """Manual test for global token validator"""
    print("Testing global token validator...")
    
    validator = get_token_validator()
    assert isinstance(validator, JWTTokenValidator)
    assert validator == token_validator
    print("✅ Global token validator test passed")


async def manual_test_auth_service_integration():
    """Manual test for auth service integration"""
    print("Testing auth service integration...")
    
    with patch('app.auth.bearer.settings') as mock_settings:
        mock_settings.jwt_algorithm = "RS256"
        mock_settings.admin_api_url = "http://test-auth-service.com"
        mock_settings.jwt_public_key = "test_key"
        validator = JWTTokenValidator()
    
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
        print("✅ Auth service integration test passed")


async def manual_test_live_api_endpoints():
    """Manual test for live API endpoints"""
    print("Testing live API endpoints...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8092/api/aips/suggest-rules",
                json={},
                timeout=5.0
            )
            if response.status_code in [401, 403]:
                print("✅ Live API endpoint test passed (proper auth rejection)")
            else:
                print(f"⚠️ Live API endpoint returned unexpected status: {response.status_code}")
    except (httpx.ConnectError, httpx.TimeoutException):
        print("ℹ️ Live API endpoint test skipped (server not available)")


def manual_test_create_error_response():
    """Manual test for error response creation"""
    print("Testing error response creation...")
    
    response = create_error_response(404, "Not found", {"error": "details"})
    
    assert response.status_code == 404
    content = json.loads(response.body)
    assert content["success"] is False
    assert content["message"] == "Not found"
    assert content["data"] == {"error": "details"}
    print("✅ Error response creation test passed")


# ==========================================================================
# PYTEST TEST CLASSES (Run with pytest if available)
# ==========================================================================

if IMPORTS_AVAILABLE:
    
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


    class TestJWTTokenValidator:
        """Test JWTTokenValidator class functionality"""
        
        @pytest.fixture
        def mock_settings(self):
            """Mock settings for testing"""
            with patch('app.auth.bearer.settings') as mock:
                mock.jwt_algorithm = "RS256"
                mock.admin_api_url = "http://test-auth-service.com"
                mock.jwt_public_key = "-----BEGIN PUBLIC KEY-----\ntest_key\n-----END PUBLIC KEY-----"
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
        
        @patch('app.auth.bearer.jwt.decode')
        def test_decode_token_expired(self, mock_jwt_decode, validator):
            """Test token decoding with expired token"""
            mock_jwt_decode.side_effect = jwt.ExpiredSignatureError("Token expired")
            
            with pytest.raises(HTTPException) as exc_info:
                validator.decode_token("expired_token")
            
            assert exc_info.value.status_code == 401
            assert "JWT token is expired" in exc_info.value.detail


    class TestScopeValidation:
        """Test scope-based authorization"""
        
        @pytest.fixture
        def validator(self):
            """Create validator for scope tests"""
            with patch('app.auth.bearer.settings'):
                return JWTTokenValidator()
        
        def test_check_scope_permissions_space_separated(self, validator):
            """Test scope checking with space-separated scopes"""
            token_payload = {"scope": "view:org manage:policy view:mdm"}
            
            result = validator.check_scope_permissions(token_payload, ["manage:policy"])
            assert result is True
        
        def test_check_scope_permissions_insufficient(self, validator):
            """Test scope checking with insufficient permissions"""
            token_payload = {"scope": "view:org view:mdm"}
            
            result = validator.check_scope_permissions(token_payload, ["manage:policy"])
            assert result is False
        
        @pytest.mark.parametrize("scope_format,required,expected", [
            ("view:org manage:policy view:mdm", ["manage:policy"], True),
            ("view:org,manage:policy,view:mdm", ["manage:policy"], True),
            ("view:org view:mdm", ["manage:policy"], False),
            ("", ["manage:policy"], False),
        ])
        def test_scope_validation_scenarios(self, scope_format, required, expected):
            """Test various scope validation scenarios"""
            validator = JWTTokenValidator()
            token_payload = {"scope": scope_format} if scope_format else {}
            
            result = validator.check_scope_permissions(token_payload, required)
            assert result == expected


    class TestUserInfo:
        """Test UserInfo class and user information extraction"""
        
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
        async def test_verify_policy_token(self):
            """Test verify_policy_token function"""
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
            
            with patch('app.auth.bearer.verify_jwt_token', new_callable=AsyncMock) as mock_verify:
                mock_user_info = Mock()
                mock_verify.return_value = mock_user_info
                
                result = await verify_policy_token(credentials)
                
                assert result == mock_user_info
                mock_verify.assert_called_once_with(credentials, required_scopes=["manage:policy"])


    class TestAuthServiceIntegration:
        """Test integration with external auth service"""
        
        @pytest.fixture
        def validator(self):
            """Create validator for auth service tests"""
            with patch('app.auth.bearer.settings') as mock_settings:
                mock_settings.jwt_algorithm = "RS256"
                mock_settings.admin_api_url = "http://test-auth-service.com"
                mock_settings.jwt_public_key = "test_key"
                return JWTTokenValidator()
        
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


    class TestJWTAuthenticationIntegration:
        """Test JWT token authentication integration with FastAPI endpoints"""
        
        @pytest.fixture
        def test_client(self):
            """Create test FastAPI client"""
            return TestClient(app)
        
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
                        timeout=5.0
                    )
                    assert response.status_code in [401, 403]
            except (httpx.ConnectError, httpx.TimeoutException):
                pytest.skip("Server not available for live testing")


    class TestTokenValidatorGlobal:
        """Test global token validator functionality"""
        
        def test_get_token_validator(self):
            """Test get_token_validator function"""
            validator = get_token_validator()
            assert isinstance(validator, JWTTokenValidator)
            assert validator == token_validator


# ==========================================================================
# UTILITY FUNCTIONS FOR TESTS
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


def create_mock_credentials(token="valid_token"):
    """Create mock HTTPAuthorizationCredentials for testing"""
    if IMPORTS_AVAILABLE:
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    return None


# ==========================================================================
# MAIN EXECUTION FOR MANUAL TESTING
# ==========================================================================

async def run_manual_tests():
    """Run all manual tests"""
    print("🧪 Running consolidated JWT authentication tests...")
    print("=" * 60)
    
    try:
        manual_test_standard_response()
        manual_test_user_info_creation()
        manual_test_jwt_validator_initialization()
        manual_test_scope_validation()
        manual_test_token_validator_global()
        manual_test_create_error_response()
        await manual_test_auth_service_integration()
        await manual_test_live_api_endpoints()
        
        print("=" * 60)
        print("✅ All manual tests completed successfully!")
        print("🔍 This consolidated file combines functionality from:")
        print("   - test_auth_bearer.py (comprehensive JWT unit tests)")
        print("   - test_jwt_auth.py (integration and scope tests)")
        print("   - test_jwt_fix.py (live endpoint validation)")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if IMPORTS_AVAILABLE:
        print("✅ All imports available - can run with pytest")
        print("Run with: python tests/run_tests.py --pattern 'test_auth'")
        print("Or manually: python tests/test_auth_consolidated.py")
    else:
        print("⚠️ Some imports not available - running manual tests only")
    
    # Run manual tests
    import asyncio
    asyncio.run(run_manual_tests())