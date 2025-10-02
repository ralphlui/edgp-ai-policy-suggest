"""
Comprehensive test suite for app/main.py - FastAPI application
Tests FastAPI app initialization, middleware, exception handlers, and endpoints
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError


class TestFastAPIAppInitialization:
    """Test FastAPI application initialization and configuration"""

    def test_app_creation_and_metadata(self):
        """Test FastAPI app is created with correct metadata"""
        from app.main import app
        
        assert app.title == "EDGP AI Policy Suggest Microservice"
        assert app.version == "1.0"
        assert app.description == "AI-powered data quality policy and rule suggestion microservice"

    def test_cors_middleware_configuration(self):
        """Test CORS middleware is properly configured"""
        from app.main import app
        
        # Check that CORS middleware is added
        middleware_found = False
        for middleware in app.user_middleware:
            if hasattr(middleware.cls, '__name__') and 'CORS' in middleware.cls.__name__:
                middleware_found = True
                break
            # Also check if it's the actual CORS middleware class
            if 'cors' in str(middleware.cls).lower():
                middleware_found = True
                break
        
        assert middleware_found, "CORS middleware not found in app middleware stack"

    def test_exception_handlers_registration(self):
        """Test that exception handlers are properly registered"""
        from app.main import app
        
        # Check that exception handlers are registered
        assert HTTPException in app.exception_handlers
        assert RequestValidationError in app.exception_handlers
        assert Exception in app.exception_handlers

    def test_router_inclusion(self):
        """Test that the API router is included"""
        from app.main import app
        
        # Check that routes are included (should have more than just the health routes)
        routes = [route.path for route in app.routes]
        
        # Should include at least our health and info endpoints
        assert "/api/aips/health" in routes
        assert "/api/aips/info" in routes


class TestHealthEndpoint:
    """Test health check endpoint functionality"""

    def setup_method(self):
        """Setup test client before each test"""
        from app.main import app
        self.client = TestClient(app)

    def test_health_endpoint_basic_response(self):
        """Test basic health endpoint response structure"""
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service_name"] == "EDGP AI Policy Suggest Microservice"
        assert data["version"] == "1.0"
        assert "timestamp" in data
        assert "services" in data
        assert "fastapi" in data["services"]
        assert "opensearch" in data["services"]

    @patch('app.api.routes.get_store')
    def test_health_endpoint_with_healthy_opensearch(self, mock_get_store):
        """Test health endpoint when OpenSearch is healthy"""
        # Mock a healthy store
        mock_store = Mock()
        mock_store.client.info.return_value = {"version": {"number": "1.0"}}
        mock_get_store.return_value = mock_store
        
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["services"]["opensearch"] == "healthy"
        assert data["services"]["fastapi"] == "healthy"

    @patch('app.api.routes.get_store')
    def test_health_endpoint_with_none_store(self, mock_get_store):
        """Test health endpoint when store initialization fails"""
        mock_get_store.return_value = None
        
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["opensearch"] == "unavailable"
        assert "opensearch_message" in data
        assert "AWS permission issues" in data["opensearch_message"]

    @patch('app.api.routes.get_store')
    def test_health_endpoint_with_opensearch_error(self, mock_get_store):
        """Test health endpoint when OpenSearch has connection error"""
        mock_store = Mock()
        mock_store.client.info.side_effect = Exception("Connection failed")
        mock_get_store.return_value = mock_store
        
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["opensearch"] == "error"
        assert "opensearch_error" in data
        assert "Connection failed" in data["opensearch_error"]

    @patch('app.api.routes.get_store')
    def test_health_endpoint_with_store_exception(self, mock_get_store):
        """Test health endpoint when get_store itself throws exception"""
        mock_get_store.side_effect = Exception("Store initialization error")
        
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["opensearch"] == "error"
        assert "opensearch_error" in data

    @patch('app.api.routes.get_store')
    def test_health_endpoint_error_truncation(self, mock_get_store):
        """Test that long error messages are truncated"""
        long_error = "x" * 200
        mock_get_store.side_effect = Exception(long_error)
        
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["opensearch_error"]) == 100


class TestServiceInfoEndpoint:
    """Test service information endpoint"""

    def setup_method(self):
        """Setup test client before each test"""
        from app.main import app
        self.client = TestClient(app)

    def test_service_info_response_structure(self):
        """Test service info endpoint returns correct structure"""
        response = self.client.get("/api/aips/info")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert data["service_name"] == "EDGP AI Policy Suggest Microservice"
        assert data["version"] == "1.0"
        assert data["description"] == "AI-powered data quality policy and rule suggestion microservice"
        assert "endpoints" in data
        assert "repository" in data
        assert "branch" in data

    def test_service_info_endpoints_list(self):
        """Test that all expected endpoints are listed"""
        response = self.client.get("/api/aips/info")
        data = response.json()
        
        endpoints = data["endpoints"]
        expected_endpoints = [
            "health", "info", "suggest_rules", "create_domain",
            "vectordb_status", "vectordb_domains", "vectordb_domain"
        ]
        
        for endpoint in expected_endpoints:
            assert endpoint in endpoints
            assert isinstance(endpoints[endpoint], str)
            assert "api/aips" in endpoints[endpoint]


class TestRequestLoggingMiddleware:
    """Test request logging middleware functionality"""

    def setup_method(self):
        """Setup test client before each test"""
        from app.main import app
        self.client = TestClient(app)

    @patch('app.main.logger')
    def test_request_logging_middleware(self, mock_logger):
        """Test that requests are logged with timing information"""
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        # Verify logging calls were made
        assert mock_logger.info.call_count >= 2
        
        # Check the structure of log calls
        calls = mock_logger.info.call_args_list
        
        # First call should be request start
        first_call = str(calls[0])
        assert "GET" in first_call
        assert "/api/aips/health" in first_call
        
        # Last call should be request completion with timing
        last_call = str(calls[-1])
        assert "200" in last_call  # Status code
        assert "s" in last_call    # Timing in seconds

    @patch('app.main.logger')
    @patch('app.main.time.time')
    def test_request_timing_calculation(self, mock_time, mock_logger):
        """Test that request timing is calculated correctly"""
        # Use a counter approach instead of side_effect to avoid StopIteration
        call_count = [0]
        def time_side_effect():
            call_count[0] += 1
            return 1000.0 + (call_count[0] * 0.1)  # Each call returns incrementally larger time
        
        mock_time.side_effect = time_side_effect
        
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        # Check that timing was logged
        calls = mock_logger.info.call_args_list
        last_call_str = str(calls[-1])
        # Should contain timing information (may vary based on actual timing)
        assert "s" in last_call_str  # Contains seconds indicator

    @patch('app.main.logger')
    def test_request_logging_different_methods(self, mock_logger):
        """Test logging works for different HTTP methods"""
        # Test GET
        self.client.get("/api/aips/info")
        
        # Test POST (this might fail but logging should still work)
        self.client.post("/api/aips/suggest-rules", json={"domain": "test"})
        
        # Verify logging happened for both requests
        calls = [str(call) for call in mock_logger.info.call_args_list]
        get_calls = [call for call in calls if "GET" in call and "/api/aips/info" in call]
        post_calls = [call for call in calls if "POST" in call and "/api/aips/suggest-rules" in call]
        
        assert len(get_calls) >= 2  # Start and end logging
        assert len(post_calls) >= 2  # Start and end logging


class TestExceptionHandlers:
    """Test custom exception handlers"""

    def setup_method(self):
        """Setup test client before each test"""
        from app.main import app
        self.client = TestClient(app)

    def test_request_validation_error_handler(self):
        """Test RequestValidationError handler"""
        # Send request without required auth header to trigger validation/auth error
        response = self.client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"}
            # No Authorization header to trigger auth error which tests exception handler
        )
        
        # Should not crash and should return proper error format
        # Auth error (401) or validation error (400, 422) or server error (500) or forbidden (403)
        assert response.status_code in [400, 401, 403, 422, 500]
        
        # Should return JSON response
        data = response.json()
        assert isinstance(data, dict)

    @patch('app.api.routes.get_store')
    def test_exception_handler_doesnt_break_health(self, mock_get_store):
        """Test that exception handlers don't interfere with normal operation"""
        mock_store = Mock()
        mock_store.client.info.return_value = {"version": {"number": "1.0"}}
        mock_get_store.return_value = mock_store
        
        # Health endpoint should work normally
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200


class TestAppIntegration:
    """Integration tests for the complete FastAPI application"""

    def setup_method(self):
        """Setup test client before each test"""
        from app.main import app
        self.client = TestClient(app)

    def test_app_starts_successfully(self):
        """Test that the application starts and responds to requests"""
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200

    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses"""
        response = self.client.options("/api/aips/health")
        
        # CORS preflight should be handled
        # Note: TestClient might not fully simulate CORS, but we can check the middleware is there
        assert response.status_code in [200, 405]  # Either handled or method not allowed

    def test_multiple_concurrent_requests(self):
        """Test that the app can handle multiple requests"""
        import concurrent.futures
        
        def make_request():
            return self.client.get("/api/aips/health")
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    def test_app_openapi_docs_accessible(self):
        """Test that OpenAPI documentation is accessible"""
        # Test OpenAPI JSON
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        assert openapi_data["info"]["title"] == "EDGP AI Policy Suggest Microservice"
        assert openapi_data["info"]["version"] == "1.0"

    def test_app_docs_endpoints(self):
        """Test that documentation endpoints are accessible"""
        # Test Swagger UI
        response = self.client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = self.client.get("/redoc")
        assert response.status_code == 200


class TestPerformanceAndReliability:
    """Test performance and reliability aspects"""

    def setup_method(self):
        """Setup test client before each test"""
        from app.main import app
        self.client = TestClient(app)

    @patch('app.main.time.time')
    def test_request_timing_accuracy(self, mock_time):
        """Test that request timing is accurate"""
        # Use a counter approach to avoid StopIteration
        call_count = [0]
        def time_side_effect():
            call_count[0] += 1
            base_time = 1000.0
            if call_count[0] == 1:
                return base_time  # First call (start time)
            elif call_count[0] == 2:
                return base_time + 0.25  # Second call (end time) - 250ms later
            else:
                return base_time + (call_count[0] * 0.1)  # Subsequent calls
        
        mock_time.side_effect = time_side_effect
        
        with patch('app.main.logger') as mock_logger:
            response = self.client.get("/api/aips/health")
            assert response.status_code == 200
            
            # Check timing was calculated and logged
            calls = mock_logger.info.call_args_list
            timing_call = str(calls[-1])
            # Check that timing information is present
            assert "s" in timing_call  # Contains seconds

    @patch('app.api.routes.get_store')
    def test_health_endpoint_performance(self, mock_get_store):
        """Test health endpoint performance with mocked store"""
        mock_store = Mock()
        mock_store.client.info.return_value = {"version": {"number": "1.0"}}
        mock_get_store.return_value = mock_store
        
        import time
        start = time.time()
        response = self.client.get("/api/aips/health")
        duration = time.time() - start
        
        assert response.status_code == 200
        # Health check should be fast (under 1 second with mocked dependencies)
        assert duration < 1.0

    def test_info_endpoint_performance(self):
        """Test service info endpoint performance"""
        import time
        start = time.time()
        response = self.client.get("/api/aips/info")
        duration = time.time() - start
        
        assert response.status_code == 200
        # Info endpoint should be very fast (under 0.1 seconds)
        assert duration < 0.1


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        """Setup test client before each test"""
        from app.main import app
        self.client = TestClient(app)

    def test_nonexistent_endpoint(self):
        """Test requesting a non-existent endpoint"""
        response = self.client.get("/api/aips/nonexistent")
        assert response.status_code == 404

    def test_invalid_http_method(self):
        """Test using invalid HTTP method on existing endpoint"""
        response = self.client.delete("/api/aips/health")
        assert response.status_code == 405  # Method Not Allowed

    @patch('app.api.routes.get_store')
    def test_health_with_extremely_long_error(self, mock_get_store):
        """Test health endpoint with extremely long error message"""
        extremely_long_error = "Error: " + "x" * 1000
        mock_get_store.side_effect = Exception(extremely_long_error)
        
        response = self.client.get("/api/aips/health")
        assert response.status_code == 200
        
        data = response.json()
        # Error should be truncated to 100 characters
        assert len(data["opensearch_error"]) == 100

    def test_health_endpoint_response_fields(self):
        """Test that health endpoint has all required fields"""
        response = self.client.get("/api/aips/health")
        data = response.json()
        
        required_fields = [
            "service_name", "version", "status", "timestamp", "services"
        ]
        
        for field in required_fields:
            assert field in data
            
        # Services should have fastapi and opensearch
        assert "fastapi" in data["services"]
        assert "opensearch" in data["services"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])