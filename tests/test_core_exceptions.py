"""
Comprehensive unit tests for app.exception.exceptions module
Tests custom exception handlers and standardized response formatting
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.exception.exceptions import (
    StandardResponse,
    create_standard_response,
    authentication_exception_handler,
    general_exception_handler,
    validation_exception_handler,
    internal_server_error_handler
)


class TestStandardResponse:
    """Test StandardResponse model"""
    
    def test_standard_response_creation(self):
        """Test StandardResponse model creation with all fields"""
        response = StandardResponse(
            data={"test": "data"},
            success=True,
            message="Test message",
            totalRecord=10,
            status=200
        )
        
        assert response.data == {"test": "data"}
        assert response.success is True
        assert response.message == "Test message"
        assert response.totalRecord == 10
        assert response.status == 200
    
    def test_standard_response_defaults(self):
        """Test StandardResponse with default values"""
        response = StandardResponse()
        
        assert response.data is None
        assert response.success is False
        assert response.message == ""
        assert response.totalRecord == 0
        assert response.status == 200
    
    def test_standard_response_partial(self):
        """Test StandardResponse with partial data"""
        response = StandardResponse(
            message="Error occurred",
            status=400
        )
        
        assert response.data is None
        assert response.success is False
        assert response.message == "Error occurred"
        assert response.totalRecord == 0
        assert response.status == 400
    
    def test_standard_response_serialization(self):
        """Test StandardResponse model serialization"""
        response = StandardResponse(
            data={"key": "value"},
            success=True,
            message="Success",
            totalRecord=5,
            status=200
        )
        
        serialized = response.model_dump()
        expected = {
            "data": {"key": "value"},
            "success": True,
            "message": "Success",
            "totalRecord": 5,
            "status": 200
        }
        
        assert serialized == expected


class TestCreateStandardResponse:
    """Test create_standard_response helper function"""
    
    def test_create_standard_response_success(self):
        """Test creating successful response"""
        response = create_standard_response(
            status_code=200,
            message="Operation successful",
            data={"result": "success"},
            success=True,
            total_record=1
        )
        
        assert response.status_code == 200
        content = json.loads(response.body)
        
        assert content["success"] is True
        assert content["message"] == "Operation successful"
        assert content["data"] == {"result": "success"}
        assert content["totalRecord"] == 1
        assert content["status"] == 200
    
    def test_create_standard_response_error(self):
        """Test creating error response"""
        response = create_standard_response(
            status_code=400,
            message="Bad request",
            data=None,
            success=False,
            total_record=0
        )
        
        assert response.status_code == 400
        content = json.loads(response.body)
        
        assert content["success"] is False
        assert content["message"] == "Bad request"
        assert content["data"] is None
        assert content["totalRecord"] == 0
        assert content["status"] == 400
    
    def test_create_standard_response_minimal(self):
        """Test creating response with minimal parameters"""
        response = create_standard_response(
            status_code=404,
            message="Not found"
        )
        
        assert response.status_code == 404
        content = json.loads(response.body)
        
        assert content["success"] is False
        assert content["message"] == "Not found"
        assert content["data"] is None
        assert content["totalRecord"] == 0
        assert content["status"] == 404
    
    def test_create_standard_response_with_complex_data(self):
        """Test creating response with complex data structure"""
        complex_data = {
            "users": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}],
            "metadata": {"total": 2, "page": 1}
        }
        
        response = create_standard_response(
            status_code=200,
            message="Users retrieved",
            data=complex_data,
            success=True,
            total_record=2
        )
        
        assert response.status_code == 200
        content = json.loads(response.body)
        
        assert content["data"] == complex_data
        assert content["totalRecord"] == 2


class TestAuthenticationExceptionHandler:
    """Test authentication exception handler"""
    
    @pytest.mark.asyncio
    async def test_authentication_handler_bearer_required(self):
        """Test authentication handler with bearer token required"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=401, detail="Bearer token required")
        
        with patch('app.exception.exceptions.logger') as mock_logger:
            response = await authentication_exception_handler(request, exc)
            
            assert response.status_code == 401
            content = json.loads(response.body)
            
            assert content["success"] is False
            assert content["message"] == "Authentication required"
            assert content["data"] is None
            assert content["status"] == 401
            
            mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authentication_handler_invalid_scheme(self):
        """Test authentication handler with invalid scheme"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=401, detail="Invalid authentication scheme. Expected 'Bearer'")
        
        response = await authentication_exception_handler(request, exc)
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["message"] == "Invalid authentication scheme"
    
    @pytest.mark.asyncio
    async def test_authentication_handler_token_missing(self):
        """Test authentication handler with missing token"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=401, detail="Bearer token missing")
        
        response = await authentication_exception_handler(request, exc)
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["message"] == "Authentication token missing"
    
    @pytest.mark.asyncio
    async def test_authentication_handler_expired_token(self):
        """Test authentication handler with expired token"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=401, detail="JWT token is expired")
        
        response = await authentication_exception_handler(request, exc)
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["message"] == "JWT token is expired"
    
    @pytest.mark.asyncio
    async def test_authentication_handler_invalid_signature(self):
        """Test authentication handler with invalid signature"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=401, detail="Invalid token signature")
        
        response = await authentication_exception_handler(request, exc)
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["message"] == "Invalid token signature"
    
    @pytest.mark.asyncio
    async def test_authentication_handler_user_not_authorized(self):
        """Test authentication handler with unauthorized user"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=401, detail="User not authorized")
        
        response = await authentication_exception_handler(request, exc)
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["message"] == "User not authorized"
    
    @pytest.mark.asyncio
    async def test_authentication_handler_insufficient_permissions(self):
        """Test authentication handler with insufficient permissions"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=403, detail="Insufficient permissions")
        
        response = await authentication_exception_handler(request, exc)
        
        assert response.status_code == 403
        content = json.loads(response.body)
        assert content["message"] == "Insufficient permissions"
    
    @pytest.mark.asyncio
    async def test_authentication_handler_unmapped_message(self):
        """Test authentication handler with unmapped error message"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=401, detail="Custom authentication error")
        
        response = await authentication_exception_handler(request, exc)
        
        assert response.status_code == 401
        content = json.loads(response.body)
        assert content["message"] == "Custom authentication error"
    
    @pytest.mark.asyncio
    async def test_authentication_handler_different_status_codes(self):
        """Test authentication handler with different status codes"""
        request = Mock(spec=Request)
        
        # Test 403 Forbidden
        exc_403 = HTTPException(status_code=403, detail="Forbidden access")
        response_403 = await authentication_exception_handler(request, exc_403)
        assert response_403.status_code == 403
        
        # Test 401 Unauthorized
        exc_401 = HTTPException(status_code=401, detail="Unauthorized access")
        response_401 = await authentication_exception_handler(request, exc_401)
        assert response_401.status_code == 401


class TestGeneralExceptionHandler:
    """Test general exception handler"""
    
    @pytest.mark.asyncio
    async def test_general_handler_not_found(self):
        """Test general exception handler with 404 error"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=404, detail="Resource not found")
        
        with patch('app.exception.exceptions.logger') as mock_logger:
            response = await general_exception_handler(request, exc)
            
            assert response.status_code == 404
            content = json.loads(response.body)
            
            assert content["success"] is False
            assert content["message"] == "Resource not found"
            assert content["data"] is None
            assert content["status"] == 404
            
            mock_logger.error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_general_handler_bad_request(self):
        """Test general exception handler with 400 error"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=400, detail="Bad request parameters")
        
        response = await general_exception_handler(request, exc)
        
        assert response.status_code == 400
        content = json.loads(response.body)
        assert content["message"] == "Bad request parameters"
    
    @pytest.mark.asyncio
    async def test_general_handler_service_unavailable(self):
        """Test general exception handler with 503 error"""
        request = Mock(spec=Request)
        exc = HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        response = await general_exception_handler(request, exc)
        
        assert response.status_code == 503
        content = json.loads(response.body)
        assert content["message"] == "Service temporarily unavailable"
    
    @pytest.mark.asyncio
    async def test_general_handler_with_headers(self):
        """Test general exception handler preserves response structure"""
        request = Mock(spec=Request)
        exc = HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )
        
        response = await general_exception_handler(request, exc)
        
        assert response.status_code == 429
        content = json.loads(response.body)
        assert content["message"] == "Rate limit exceeded"


class TestValidationExceptionHandler:
    """Test validation exception handler"""
    
    @pytest.mark.asyncio
    async def test_validation_handler_basic(self):
        """Test validation exception handler with basic error"""
        request = Mock(spec=Request)
        exc = ValidationError.from_exception_data("Test", [])
        
        with patch('app.exception.exceptions.logger') as mock_logger:
            response = await validation_exception_handler(request, exc)
            
            assert response.status_code == 422
            content = json.loads(response.body)
            
            assert content["success"] is False
            assert "Validation error:" in content["message"]
            assert content["data"] is None
            assert content["status"] == 422
            
            mock_logger.error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validation_handler_request_validation_error(self):
        """Test validation handler with RequestValidationError"""
        request = Mock(spec=Request)
        
        # Create a mock RequestValidationError
        exc = Exception("Field 'email' is required")
        
        response = await validation_exception_handler(request, exc)
        
        assert response.status_code == 422
        content = json.loads(response.body)
        assert "Field 'email' is required" in content["message"]
    
    @pytest.mark.asyncio
    async def test_validation_handler_complex_error(self):
        """Test validation handler with complex validation error"""
        request = Mock(spec=Request)
        exc = Exception("Multiple validation errors: field1 missing, field2 invalid format")
        
        response = await validation_exception_handler(request, exc)
        
        assert response.status_code == 422
        content = json.loads(response.body)
        assert "Multiple validation errors" in content["message"]
    
    @pytest.mark.asyncio
    async def test_validation_handler_empty_error(self):
        """Test validation handler with empty error message"""
        request = Mock(spec=Request)
        exc = Exception("")
        
        response = await validation_exception_handler(request, exc)
        
        assert response.status_code == 422
        content = json.loads(response.body)
        assert content["message"] == "Validation error: "


class TestInternalServerErrorHandler:
    """Test internal server error handler"""
    
    @pytest.mark.asyncio
    async def test_internal_error_handler_basic(self):
        """Test internal server error handler with basic error"""
        request = Mock(spec=Request)
        exc = Exception("Database connection failed")
        
        with patch('app.exception.exceptions.logger') as mock_logger:
            response = await internal_server_error_handler(request, exc)
            
            assert response.status_code == 500
            content = json.loads(response.body)
            
            assert content["success"] is False
            assert content["message"] == "Internal server error"
            assert content["data"] is None
            assert content["status"] == 500
            
            mock_logger.error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_internal_error_handler_runtime_error(self):
        """Test internal server error handler with RuntimeError"""
        request = Mock(spec=Request)
        exc = RuntimeError("Memory allocation failed")
        
        response = await internal_server_error_handler(request, exc)
        
        assert response.status_code == 500
        content = json.loads(response.body)
        assert content["message"] == "Internal server error"
    
    @pytest.mark.asyncio
    async def test_internal_error_handler_type_error(self):
        """Test internal server error handler with TypeError"""
        request = Mock(spec=Request)
        exc = TypeError("'NoneType' object is not subscriptable")
        
        response = await internal_server_error_handler(request, exc)
        
        assert response.status_code == 500
        content = json.loads(response.body)
        assert content["message"] == "Internal server error"
    
    @pytest.mark.asyncio
    async def test_internal_error_handler_key_error(self):
        """Test internal server error handler with KeyError"""
        request = Mock(spec=Request)
        exc = KeyError("'required_field'")
        
        response = await internal_server_error_handler(request, exc)
        
        assert response.status_code == 500
        content = json.loads(response.body)
        assert content["message"] == "Internal server error"
    
    @pytest.mark.asyncio
    async def test_internal_error_handler_custom_exception(self):
        """Test internal server error handler with custom exception"""
        class CustomException(Exception):
            pass
        
        request = Mock(spec=Request)
        exc = CustomException("Custom error occurred")
        
        response = await internal_server_error_handler(request, exc)
        
        assert response.status_code == 500
        content = json.loads(response.body)
        assert content["message"] == "Internal server error"


class TestExceptionHandlerIntegration:
    """Test exception handlers integration"""
    
    @pytest.mark.asyncio
    async def test_handler_response_consistency(self):
        """Test that all handlers return consistent response format"""
        request = Mock(spec=Request)
        
        # Test all handlers return JSONResponse with same structure
        handlers_and_exceptions = [
            (authentication_exception_handler, HTTPException(status_code=401, detail="Auth error")),
            (general_exception_handler, HTTPException(status_code=404, detail="Not found")),
            (validation_exception_handler, Exception("Validation failed")),
            (internal_server_error_handler, Exception("Server error"))
        ]
        
        for handler, exception in handlers_and_exceptions:
            response = await handler(request, exception)
            content = json.loads(response.body)
            
            # Check consistent structure
            required_fields = ["data", "success", "message", "totalRecord", "status"]
            for field in required_fields:
                assert field in content
            
            assert isinstance(content["success"], bool)
            assert isinstance(content["message"], str)
            assert isinstance(content["totalRecord"], int)
            assert isinstance(content["status"], int)
    
    @pytest.mark.asyncio
    async def test_handler_logging_behavior(self):
        """Test that handlers log appropriately"""
        request = Mock(spec=Request)
        
        with patch('app.exception.exceptions.logger') as mock_logger:
            # Authentication handler should use warning
            auth_exc = HTTPException(status_code=401, detail="Token expired")
            await authentication_exception_handler(request, auth_exc)
            mock_logger.warning.assert_called()
            
            # General handler should use error
            mock_logger.reset_mock()
            general_exc = HTTPException(status_code=500, detail="Server error")
            await general_exception_handler(request, general_exc)
            mock_logger.error.assert_called()
            
            # Validation handler should use error
            mock_logger.reset_mock()
            validation_exc = Exception("Validation error")
            await validation_exception_handler(request, validation_exc)
            mock_logger.error.assert_called()
            
            # Internal error handler should use error
            mock_logger.reset_mock()
            internal_exc = Exception("Internal error")
            await internal_server_error_handler(request, internal_exc)
            mock_logger.error.assert_called()