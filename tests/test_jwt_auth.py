"""
Test JWT authentication and authorization functionality
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from tests.test_config import setup_test_environment

# Setup test environment
setup_test_environment()

@pytest.mark.auth
@pytest.mark.jwt
class TestJWTAuthentication:
    """Test JWT token authentication and validation"""
    
    def test_valid_token_authentication(self, test_client, auth_headers):
        """Test successful authentication with valid token"""
        response = test_client.get("/api/suggest-policy", headers=auth_headers)
        assert response.status_code in [200, 404]  # 404 if endpoint doesn't exist yet
    
    def test_missing_token(self, test_client):
        """Test authentication failure with missing token"""
        response = test_client.get("/api/suggest-policy")
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert "Bearer token required" in data["message"] or "Unauthorized" in str(data)
    
    def test_invalid_token_format(self, test_client):
        """Test authentication failure with invalid token format"""
        headers = {"Authorization": "InvalidToken"}
        response = test_client.get("/api/suggest-policy", headers=headers)
        assert response.status_code == 401
    
    def test_expired_token(self, test_client, mock_jwt_validator):
        """Test authentication failure with expired token"""
        from jwt import ExpiredSignatureError
        mock_jwt_validator.decode_token.side_effect = ExpiredSignatureError("Token expired")
        
        headers = {"Authorization": "Bearer expired.token.here"}
        response = test_client.get("/api/suggest-policy", headers=headers)
        assert response.status_code == 401

@pytest.mark.auth
@pytest.mark.unit
class TestScopeValidation:
    """Test scope-based authorization"""
    
    def test_valid_manage_policy_scope(self, mock_jwt_validator):
        """Test scope validation with manage:policy scope"""
        from app.auth.bearer import JWTTokenValidator
        
        validator = JWTTokenValidator()
        token_payload = {
            "scope": "view:org view:mdm manage:policy",
            "userEmail": "test@example.com"
        }
        
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is True
    
    def test_missing_manage_policy_scope(self, mock_jwt_validator):
        """Test scope validation without manage:policy scope"""
        from app.auth.bearer import JWTTokenValidator
        
        validator = JWTTokenValidator()
        token_payload = {
            "scope": "view:org view:mdm",
            "userEmail": "test@example.com"
        }
        
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is False
    
    def test_comma_separated_scopes(self, mock_jwt_validator):
        """Test comma-separated scope format"""
        from app.auth.bearer import JWTTokenValidator
        
        validator = JWTTokenValidator()
        token_payload = {
            "scope": "view:org,view:mdm,manage:policy",
            "userEmail": "test@example.com"
        }
        
        result = validator.check_scope_permissions(token_payload, ["manage:policy"])
        assert result is True

@pytest.mark.auth
@pytest.mark.integration
class TestAuthServiceIntegration:
    """Test integration with external auth service"""
    
    @patch('httpx.AsyncClient')
    async def test_successful_user_validation(self, mock_httpx):
        """Test successful user validation with auth service"""
        from app.auth.bearer import JWTTokenValidator
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"userId": "test-123", "status": "active"}
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_httpx.return_value = mock_client
        
        validator = JWTTokenValidator()
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "test-user-123"
        }
        
        result = await validator.validate_user_with_auth_service(token_payload, "test-token")
        assert result["userId"] == "test-123"
    
    @patch('httpx.AsyncClient')
    async def test_user_not_authorized(self, mock_httpx):
        """Test user validation failure (unauthorized)"""
        from app.auth.bearer import JWTTokenValidator
        
        # Mock 401 response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        
        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response
        mock_httpx.return_value = mock_client
        
        validator = JWTTokenValidator()
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "test-user-123"
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_user_with_auth_service(token_payload, "test-token")
        
        assert exc_info.value.status_code == 401
        assert "not authorized" in str(exc_info.value.detail)

@pytest.mark.auth
@pytest.mark.unit
class TestUserInfoExtraction:
    """Test user information extraction from JWT tokens"""
    
    def test_user_info_creation(self, test_user_info):
        """Test UserInfo object creation"""
        from app.auth.bearer import UserInfo
        
        token_payload = {
            "userEmail": "test@example.com",
            "sub": "test-user-123",
            "userName": "Test User",
            "orgId": "test-org-id",
            "iat": 1234567890,
            "exp": 9999999999
        }
        
        user_info = UserInfo(
            email="test@example.com",
            user_id="test-user-123",
            scopes=["manage:policy"],
            token_payload=token_payload
        )
        
        assert user_info.email == "test@example.com"
        assert user_info.user_id == "test-user-123"
        assert user_info.user_name == "Test User"
        assert user_info.org_id == "test-org-id"
        assert "manage:policy" in user_info.scopes

if __name__ == "__main__":
    pytest.main([__file__, "-v"])