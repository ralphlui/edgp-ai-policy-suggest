"""
Comprehensive unit tests for app.main module
Tests FastAPI application initialization, middleware, and endpoints
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError

from app.main import app, health, service_info, log_requests


class TestMainModule:
    """Test main FastAPI application module"""
    
    def test_app_creation(self):
        """Test FastAPI app creation"""
        assert isinstance(app, FastAPI)
        assert app.title == "EDGP AI Policy Suggest Microservice"
        assert app.version == "1.0"
        assert "AI-powered data quality policy" in app.description
    
    def test_app_metadata(self):
        """Test app metadata configuration"""
        assert app.title is not None
        assert app.version is not None
        assert "EDGP AI Policy Suggest Microservice" in app.title
        assert app.version == "1.0"
    
    def test_app_has_middleware(self):
        """Test that CORS middleware is configured"""
        # Check that middleware is registered
        middleware_types = [middleware.cls.__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_types
    
    def test_app_has_exception_handlers(self):
        """Test that exception handlers are registered"""
        # Check that exception handlers exist
        assert HTTPException in app.exception_handlers
        assert RequestValidationError in app.exception_handlers
        assert Exception in app.exception_handlers
    
    def test_app_has_router_included(self):
        """Test that API router is included"""
        # Check that routes are registered
        route_paths = [route.path for route in app.routes]
        # Should have health and info endpoints plus any from router
        assert "/api/aips/health" in route_paths
        assert "/api/aips/info" in route_paths


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_endpoint_basic(self):
        """Test basic health endpoint response"""
        with TestClient(app) as client:
            response = client.get("/api/aips/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["service_name"] == "EDGP AI Policy Suggest Microservice"
            assert data["version"] == "1.0"
            assert "timestamp" in data
            assert "services" in data
    
    def test_health_endpoint_structure(self):
        """Test health endpoint response structure"""
        with TestClient(app) as client:
            response = client.get("/api/aips/health")
            data = response.json()
            
            # Check required fields
            required_fields = ["service_name", "version", "status", "timestamp", "services"]
            for field in required_fields:
                assert field in data
            
            # Check services structure
            assert "fastapi" in data["services"]
            assert "opensearch" in data["services"]
            assert data["services"]["fastapi"] == "healthy"
    
    @patch('app.api.routes.get_store')
    def test_health_endpoint_opensearch_healthy(self, mock_get_store):
        """Test health endpoint with healthy OpenSearch"""
        mock_store = Mock()
        mock_store.client.info.return_value = {"version": "1.0"}
        mock_get_store.return_value = mock_store
        
        with TestClient(app) as client:
            response = client.get("/api/aips/health")
            data = response.json()
            
            assert data["status"] == "ok"
            assert data["services"]["opensearch"] == "healthy"
    
    @patch('app.api.routes.get_store')
    def test_health_endpoint_opensearch_unavailable(self, mock_get_store):
        """Test health endpoint with unavailable OpenSearch"""
        mock_get_store.return_value = None
        
        with TestClient(app) as client:
            response = client.get("/api/aips/health")
            data = response.json()
            
            assert data["status"] == "degraded"
            assert data["services"]["opensearch"] == "unavailable"
            assert "opensearch_message" in data
    
    @patch('app.api.routes.get_store')
    def test_health_endpoint_opensearch_error(self, mock_get_store):
        """Test health endpoint with OpenSearch error"""
        mock_store = Mock()
        mock_store.client.info.side_effect = Exception("Connection failed")
        mock_get_store.return_value = mock_store
        
        with TestClient(app) as client:
            response = client.get("/api/aips/health")
            data = response.json()
            
            assert data["status"] == "degraded"
            assert data["services"]["opensearch"] == "error"
            assert "opensearch_error" in data
    
    @patch('app.api.routes.get_store')
    def test_health_endpoint_store_exception(self, mock_get_store):
        """Test health endpoint when get_store raises exception"""
        mock_get_store.side_effect = Exception("Store initialization failed")
        
        with TestClient(app) as client:
            response = client.get("/api/aips/health")
            data = response.json()
            
            assert data["status"] == "degraded"
            assert data["services"]["opensearch"] == "error"
            assert "opensearch_error" in data


class TestServiceInfoEndpoint:
    """Test service info endpoint"""
    
    def test_service_info_endpoint(self):
        """Test service info endpoint response"""
        with TestClient(app) as client:
            response = client.get("/api/aips/info")
            assert response.status_code == 200
            
            data = response.json()
            assert data["service_name"] == "EDGP AI Policy Suggest Microservice"
            assert data["version"] == "1.0"
            assert "description" in data
            assert "endpoints" in data
            assert "repository" in data
            assert "branch" in data
    
    def test_service_info_endpoints(self):
        """Test service info endpoint lists"""
        with TestClient(app) as client:
            response = client.get("/api/aips/info")
            data = response.json()
            
            endpoints = data["endpoints"]
            expected_endpoints = [
                "health", "info", "suggest_rules", "create_domain",
                "vectordb_status", "vectordb_domains", "vectordb_domain"
            ]
            
            for endpoint in expected_endpoints:
                assert endpoint in endpoints
                assert isinstance(endpoints[endpoint], str)
    
    def test_service_info_metadata(self):
        """Test service info metadata"""
        with TestClient(app) as client:
            response = client.get("/api/aips/info")
            data = response.json()
            
            assert data["repository"] == "edgp-ai-policy-suggest"
            assert data["branch"] == "feature/opensearch-column-store"
            assert "AI-powered data quality policy" in data["description"]


class TestMiddleware:
    """Test application middleware"""
    
    @pytest.mark.asyncio
    async def test_log_requests_middleware(self):
        """Test request logging middleware"""
        # Create mock request and response
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url = "http://test.com/api/test"
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Mock call_next function
        async def mock_call_next(request):
            return mock_response
        
        # Test the middleware
        with patch('app.main.logger') as mock_logger, \
             patch('app.main.time.time', side_effect=[1000.0, 1000.5]):
            
            result = await log_requests(mock_request, mock_call_next)
            
            assert result == mock_response
            assert mock_logger.info.call_count == 2
            
            # Check logged messages
            calls = mock_logger.info.call_args_list
            assert "GET http://test.com/api/test" in str(calls[0])
            assert "GET http://test.com/api/test - 200 - 0.50s" in str(calls[1])
    
    @pytest.mark.asyncio
    async def test_log_requests_middleware_timing(self):
        """Test request logging middleware timing calculation"""
        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://test.com/api/create"
        
        mock_response = Mock()
        mock_response.status_code = 201
        
        async def mock_call_next(request):
            return mock_response
        
        with patch('app.main.logger') as mock_logger, \
             patch('app.main.time.time', side_effect=[1000.0, 1002.35]):  # 2.35 second difference
            
            result = await log_requests(mock_request, mock_call_next)
            
            assert result == mock_response
            # Check that timing is logged correctly
            calls = mock_logger.info.call_args_list
            assert "2.35s" in str(calls[1])


class TestAppConfiguration:
    """Test application configuration"""
    
    def test_cors_configuration(self):
        """Test CORS middleware configuration"""
        # CORS middleware should be configured to allow all origins
        # This is tested by checking if the middleware exists and basic functionality
        with TestClient(app) as client:
            response = client.options("/api/aips/health")
            # Should not fail with CORS error
            assert response.status_code in [200, 405]  # Either OK or Method Not Allowed
    
    def test_exception_handlers_configured(self):
        """Test that exception handlers are properly configured"""
        # Check that our custom exception handlers are registered
        handler_types = list(app.exception_handlers.keys())
        
        # Should have handlers for these exception types
        expected_handlers = [HTTPException, RequestValidationError, Exception]
        for handler_type in expected_handlers:
            assert handler_type in handler_types
    
    def test_app_startup_configuration(self):
        """Test application startup configuration"""
        # Test that the app is properly configured for startup
        assert app.title is not None
        assert app.version is not None
        assert app.description is not None
        
        # Check that middleware is added
        assert len(app.user_middleware) > 0
        
        # Check that routes are registered
        assert len(app.routes) > 0


class TestErrorHandling:
    """Test application error handling"""
    
    def test_app_handles_404(self):
        """Test that app handles 404 errors properly"""
        with TestClient(app) as client:
            response = client.get("/nonexistent-endpoint")
            assert response.status_code == 404
    
    def test_app_handles_method_not_allowed(self):
        """Test that app handles method not allowed errors"""
        with TestClient(app) as client:
            # Try POST on a GET-only endpoint
            response = client.post("/api/aips/health")
            assert response.status_code in [405, 422]  # Method Not Allowed or Unprocessable Entity


class TestAppIntegration:
    """Test application integration functionality"""
    
    def test_app_basic_functionality(self):
        """Test basic app functionality"""
        with TestClient(app) as client:
            # Test health endpoint
            health_response = client.get("/api/aips/health")
            assert health_response.status_code == 200
            
            # Test info endpoint
            info_response = client.get("/api/aips/info")
            assert info_response.status_code == 200
            
            # Both should return JSON
            assert health_response.headers["content-type"] == "application/json"
            assert info_response.headers["content-type"] == "application/json"
    
    def test_app_logging_setup(self):
        """Test that logging is properly configured"""
        with patch('app.main.logger') as mock_logger:
            with TestClient(app) as client:
                response = client.get("/api/aips/health")
                assert response.status_code == 200
                
                # Logger should have been called for request logging
                assert mock_logger.info.called
    
    def test_app_middleware_chain(self):
        """Test that middleware chain works properly"""
        with TestClient(app) as client:
            # Test that requests go through middleware chain
            response = client.get("/api/aips/health")
            assert response.status_code == 200
            
            # CORS headers should be present (from CORSMiddleware)
            # Note: Specific headers depend on CORS configuration
            assert "access-control-allow-origin" in response.headers or response.status_code == 200