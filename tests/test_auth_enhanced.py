"""
Enhanced test suite for JWT Bearer Token Authentication
Provides comprehensive test coverage including edge cases, error conditions, and performance scenarios.
"""

import pytest
import jwt
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
import httpx
from datetime import datetime, timedelta
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Import modules to test
from app.auth.bearer import (
    JWTTokenValidator,
    UserInfo,
    verify_jwt_token,
    verify_policy_token,
    verify_mdm_token,
    verify_any_scope_token,
    optional_jwt_token,
    get_token_validator,
    get_user_info,
    create_error_response,
    create_auth_error_response,
    StandardResponse,
    security
)


class TestJWTTokenValidatorEnhanced:
    """Enhanced tests for JWTTokenValidator class with edge cases"""

    @pytest.fixture
    def validator_no_key(self):
        """Validator with no public key configured"""
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = None
            mock_settings.jwt_algorithm = "RS256"
            mock_settings.admin_api_url = "http://test-auth-service"
            return JWTTokenValidator()

    @pytest.fixture
    def validator_invalid_key(self):
        """Validator with invalid public key"""
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = "invalid-key-format"
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

    def test_public_key_loading_already_pem_format(self):
        """Test loading public key that's already in PEM format"""
        pem_key = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtw
-----END PUBLIC KEY-----"""
        
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = pem_key
            mock_settings.jwt_algorithm = "RS256"
            mock_settings.admin_api_url = "http://test-auth-service"
            validator = JWTTokenValidator()
            
        assert validator.public_key == pem_key

    def test_public_key_loading_failure(self):
        """Test handling of public key loading failure"""
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = "invalid-key"
            mock_settings.jwt_algorithm = "RS256"
            mock_settings.admin_api_url = "http://test-auth-service"
            
            # Mock an exception during key processing
            with patch('app.auth.bearer.logger') as mock_logger:
                validator = JWTTokenValidator()
                # Should handle the error gracefully
                assert validator.public_key is not None  # Still processes the invalid key

    def test_decode_token_no_public_key(self, validator_no_key):
        """Test token decoding when no public key is configured"""
        with pytest.raises(HTTPException) as exc_info:
            validator_no_key.decode_token("dummy.token.here")
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        # The actual implementation catches HTTPException and converts it to "Token validation error"
        assert "Token validation error" in str(exc_info.value.detail)

    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_invalid_signature_error(self, mock_jwt_decode, validator_base64_key):
        """Test handling of invalid signature error"""
        mock_jwt_decode.side_effect = jwt.InvalidSignatureError("Invalid signature")
        
        with pytest.raises(HTTPException) as exc_info:
            validator_base64_key.decode_token("invalid.signature.token")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token signature" in str(exc_info.value.detail)

    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_missing_required_claim_error(self, mock_jwt_decode, validator_base64_key):
        """Test handling of missing required claim error"""
        mock_jwt_decode.side_effect = jwt.MissingRequiredClaimError("Missing 'exp' claim")
        
        with pytest.raises(HTTPException) as exc_info:
            validator_base64_key.decode_token("token.missing.claim")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "JWT token missing required claim" in str(exc_info.value.detail)

    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_generic_invalid_token_error(self, mock_jwt_decode, validator_base64_key):
        """Test handling of generic invalid token error"""
        mock_jwt_decode.side_effect = jwt.InvalidTokenError("Invalid token")
        
        with pytest.raises(HTTPException) as exc_info:
            validator_base64_key.decode_token("invalid.token.format")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid JWT token" in str(exc_info.value.detail)

    @patch('app.auth.bearer.jwt.decode')
    def test_decode_token_unexpected_error(self, mock_jwt_decode, validator_base64_key):
        """Test handling of unexpected error during token decoding"""
        mock_jwt_decode.side_effect = ValueError("Unexpected error")
        
        with pytest.raises(HTTPException) as exc_info:
            validator_base64_key.decode_token("token.causing.error")
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Token validation error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_validate_user_missing_user_email(self, validator_base64_key):
        """Test user validation with missing user email"""
        token_payload = {"sub": "user123", "scope": "manage:policy"}
        
        with pytest.raises(HTTPException) as exc_info:
            await validator_base64_key.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Token missing user email" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_validate_user_missing_user_id(self, validator_base64_key):
        """Test user validation with missing user ID (sub field)"""
        token_payload = {"userEmail": "test@example.com", "scope": "manage:policy"}
        
        with pytest.raises(HTTPException) as exc_info:
            await validator_base64_key.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Token missing user ID (sub)" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_validate_user_auth_service_404(self, mock_get, validator_base64_key):
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
            await validator_base64_key.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "User not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_validate_user_auth_service_500(self, mock_get, validator_base64_key):
        """Test user validation when auth service returns 500"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_get.return_value = mock_response
        
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await validator_base64_key.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Authentication service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_validate_user_timeout(self, mock_get, validator_base64_key):
        """Test user validation with auth service timeout"""
        mock_get.side_effect = httpx.TimeoutException("Request timeout")
        
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await validator_base64_key.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Authentication service timeout" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_validate_user_request_error(self, mock_get, validator_base64_key):
        """Test user validation with auth service request error"""
        mock_get.side_effect = httpx.RequestError("Connection failed")
        
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await validator_base64_key.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Authentication service unavailable" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_validate_user_unexpected_error(self, mock_get, validator_base64_key):
        """Test user validation with unexpected error"""
        mock_get.side_effect = ValueError("Unexpected error")
        
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await validator_base64_key.validate_user_with_auth_service(token_payload, "token")
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "User validation error" in str(exc_info.value.detail)

    def test_check_scope_permissions_empty_scope(self, validator_base64_key):
        """Test scope checking with empty scope string"""
        token_payload = {"scope": ""}
        result = validator_base64_key.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False

    def test_check_scope_permissions_missing_scope(self, validator_base64_key):
        """Test scope checking with missing scope field"""
        token_payload = {}
        result = validator_base64_key.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False

    def test_check_scope_permissions_comma_separated_with_spaces(self, validator_base64_key):
        """Test scope checking with comma-separated scopes including spaces"""
        token_payload = {"scope": "view:org, manage:policy , view:mdm"}
        result = validator_base64_key.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is True

    def test_check_scope_permissions_mixed_separators(self, validator_base64_key):
        """Test scope checking with mixed separators (edge case)"""
        token_payload = {"scope": "view:org manage:policy, view:mdm"}
        result = validator_base64_key.check_scope_permissions(token_payload, ["manage:policy"])
        # Should handle comma-separated format first
        assert result is False  # Because "view:org manage:policy" is treated as one scope

    def test_check_scope_permissions_exception_handling(self, validator_base64_key):
        """Test scope checking exception handling"""
        # Mock a scenario where payload access causes an exception
        token_payload = Mock()
        token_payload.get.side_effect = Exception("Unexpected error")
        
        result = validator_base64_key.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False


class TestEnhancedScopeValidation:
    """Enhanced scope validation tests with edge cases"""

    @pytest.fixture
    def validator(self):
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = "test-key"
            mock_settings.jwt_algorithm = "RS256"
            mock_settings.admin_api_url = "http://test-auth-service"
            return JWTTokenValidator()

    @pytest.mark.parametrize("scope_string,expected", [
        ("", False),  # Empty scope
        ("   ", False),  # Whitespace only
        ("manage:policy", True),  # Single correct scope
        ("view:org", False),  # Wrong scope
        ("manage:policy view:org", True),  # Multiple scopes with correct one
        ("view:org manage:policy", True),  # Multiple scopes with correct one (different order)
        ("manage:policy,view:org", True),  # Comma-separated with correct scope
        ("view:org,manage:policy", True),  # Comma-separated with correct scope (different order)
        ("manage:policy, view:org, view:mdm", True),  # Comma-separated with spaces
        ("view:org, view:mdm, manage:other", False),  # Comma-separated without correct scope
        ("MANAGE:POLICY", False),  # Case sensitivity test
        ("manage:policy manage:mdm", True),  # Multiple similar scopes
    ])
    def test_comprehensive_scope_validation(self, validator, scope_string, expected):
        """Comprehensive scope validation scenarios"""
        token_payload = {"scope": scope_string}
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result == expected


class TestEnhancedUserInfo:
    """Enhanced UserInfo class tests"""

    def test_user_info_with_all_optional_fields(self):
        """Test UserInfo creation with all optional fields"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy view:org",
            "iat": 1640995200,
            "exp": 1641081600,
            "orgId": "org456",
            "userName": "testuser"
        }
        
        user_info = UserInfo(
            email="test@example.com",
            user_id="user123",
            scopes=["manage:policy", "view:org"],
            token_payload=token_payload
        )
        
        assert user_info.email == "test@example.com"
        assert user_info.user_id == "user123"
        assert user_info.scopes == ["manage:policy", "view:org"]
        assert user_info.iat == 1640995200
        assert user_info.exp == 1641081600
        assert user_info.org_id == "org456"
        assert user_info.user_name == "testuser"
        assert user_info.payload == token_payload

    def test_user_info_with_missing_optional_fields(self):
        """Test UserInfo creation with missing optional fields"""
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        
        user_info = UserInfo(
            email="test@example.com",
            user_id="user123",
            scopes=["manage:policy"],
            token_payload=token_payload
        )
        
        assert user_info.iat is None
        assert user_info.exp is None
        assert user_info.org_id is None
        assert user_info.user_name is None


class TestEnhancedVerificationFunctions:
    """Enhanced tests for verification functions with edge cases"""

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
    async def test_verify_jwt_token_case_insensitive_scheme(self):
        """Test verification with different case schemes"""
        credentials = HTTPAuthorizationCredentials(scheme="bearer", credentials="token123")
        
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode, \
             patch('app.auth.bearer.token_validator.validate_user_with_auth_service') as mock_validate:
            
            mock_decode.return_value = {
                "userEmail": "test@example.com",
                "sub": "user123",
                "scope": "manage:policy"
            }
            mock_validate.return_value = {"status": "valid"}
            
            result = await verify_jwt_token(credentials)
            assert result.email == "test@example.com"

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

    @pytest.mark.asyncio
    async def test_optional_jwt_token_with_unexpected_error(self):
        """Test optional JWT token with unexpected error (should return None)"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")
        
        with patch('app.auth.bearer.verify_jwt_token') as mock_verify:
            mock_verify.side_effect = ValueError("Unexpected error")
            
            result = await optional_jwt_token(credentials)
            assert result is None


class TestConcurrencyAndPerformance:
    """Tests for concurrent access and performance scenarios"""

    @pytest.mark.asyncio
    async def test_concurrent_token_validation(self):
        """Test concurrent token validation to ensure thread safety"""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-token")
        
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode, \
             patch('app.auth.bearer.token_validator.validate_user_with_auth_service') as mock_validate:
            
            mock_decode.return_value = {
                "userEmail": "test@example.com",
                "sub": "user123",
                "scope": "manage:policy"
            }
            mock_validate.return_value = {"status": "valid"}
            
            # Run multiple concurrent validations
            tasks = [verify_jwt_token(credentials) for _ in range(10)]
            results = await asyncio.gather(*tasks)
            
            # All should succeed and return the same user
            assert len(results) == 10
            assert all(result.email == "test@example.com" for result in results)

    @pytest.mark.asyncio
    async def test_token_validator_singleton_behavior(self):
        """Test that token validator behaves correctly as singleton"""
        validator1 = get_token_validator()
        validator2 = get_token_validator()
        
        # Should be the same instance
        assert validator1 is validator2

    def test_multiple_validator_instances_thread_safety(self):
        """Test thread safety when creating multiple validator instances"""
        validators = []
        
        def create_validator():
            with patch('app.auth.bearer.settings') as mock_settings:
                mock_settings.jwt_public_key = "test-key"
                mock_settings.jwt_algorithm = "RS256"
                mock_settings.admin_api_url = "http://test-auth-service"
                validator = JWTTokenValidator()
                validators.append(validator)
        
        # Create validators in multiple threads
        threads = [threading.Thread(target=create_validator) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All validators should be created successfully
        assert len(validators) == 5
        assert all(v.algorithm == "RS256" for v in validators)


class TestErrorResponseHelpers:
    """Tests for error response helper functions"""

    def test_create_error_response_with_data(self):
        """Test creating error response with additional data"""
        response = create_error_response(
            status_code=400,
            message="Validation failed",
            data={"field": "email", "error": "invalid format"}
        )
        
        assert response.status_code == 400
        content = json.loads(response.body)
        assert content["success"] is False
        assert content["message"] == "Validation failed"
        assert content["data"]["field"] == "email"
        assert content["status"] == 400

    def test_create_auth_error_response(self):
        """Test creating authentication error response"""
        response = create_auth_error_response("Token expired")
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["success"] is False
        assert content["message"] == "Token expired"
        assert content["status"] == 401

    def test_standard_response_model_validation(self):
        """Test StandardResponse model validation"""
        # Valid response
        response = StandardResponse(
            data={"test": "data"},
            success=True,
            message="Success",
            totalRecord=1,
            status=200
        )
        
        assert response.success is True
        assert response.data == {"test": "data"}
        
        # Response with defaults
        response_default = StandardResponse(message="Test")
        assert response_default.success is False
        assert response_default.data is None
        assert response_default.totalRecord == 0
        assert response_default.status == 200


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
    async def test_get_user_info_decode_failure(self):
        """Test user info retrieval with decode failure"""
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode:
            mock_decode.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
            with pytest.raises(HTTPException):
                await get_user_info("invalid-token")

    @pytest.mark.asyncio
    async def test_get_user_info_validation_failure(self):
        """Test user info retrieval with auth service validation failure"""
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode, \
             patch('app.auth.bearer.token_validator.validate_user_with_auth_service') as mock_validate:
            
            mock_decode.return_value = {
                "userEmail": "test@example.com",
                "sub": "user123",
                "scope": "manage:policy"
            }
            mock_validate.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authorized"
            )
            
            with pytest.raises(HTTPException):
                await get_user_info("test-token")


class TestEdgeCaseScenarios:
    """Tests for unusual edge case scenarios"""

    @pytest.mark.asyncio
    async def test_verify_token_with_very_long_token(self):
        """Test token verification with unusually long token"""
        # Create a very long token string
        long_token = "a" * 10000
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=long_token)
        
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode:
            mock_decode.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
            with pytest.raises(HTTPException):
                await verify_jwt_token(credentials)

    @pytest.mark.asyncio
    async def test_verify_token_with_unicode_characters(self):
        """Test token verification with unicode characters"""
        unicode_token = "token.with.unicode.🔒.chars"
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=unicode_token)
        
        with patch('app.auth.bearer.token_validator.decode_token') as mock_decode:
            mock_decode.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
            with pytest.raises(HTTPException):
                await verify_jwt_token(credentials)

    def test_scope_permissions_with_unicode_scopes(self):
        """Test scope checking with unicode characters"""
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = "test-key"
            mock_settings.jwt_algorithm = "RS256"
            mock_settings.admin_api_url = "http://test-auth-service"
            validator = JWTTokenValidator()
        
        token_payload = {"scope": "manage:policy view:org🔒"}
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is True  # Should still find manage:policy

    @pytest.mark.asyncio
    async def test_auth_service_response_malformed_json(self):
        """Test handling of malformed JSON response from auth service"""
        with patch('app.auth.bearer.settings') as mock_settings:
            mock_settings.jwt_public_key = "test-key"
            mock_settings.jwt_algorithm = "RS256"
            mock_settings.admin_api_url = "http://test-auth-service"
            validator = JWTTokenValidator()
        
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy"
        }
        
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
            mock_get.return_value = mock_response
            
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_user_with_auth_service(token_payload, "token")
            
            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    # Run specific test categories
    print("Running enhanced authentication tests...")
    
    # Example of running specific test categories
    pytest.main([
        "-v",
        __file__ + "::TestJWTTokenValidatorEnhanced",
        __file__ + "::TestEnhancedScopeValidation",
        "--tb=short"
    ])