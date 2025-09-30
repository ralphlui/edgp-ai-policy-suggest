"""
Comprehensive tests for API routes
Tests all endpoints to achieve high coverage on routes.py
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from fastapi import status
import json
import pandas as pd
import io
from app.main import app
from app.auth.bearer import UserInfo
from app.aoss.column_store import ColumnDoc
from tests.test_config import setup_test_environment

# Setup test environment
setup_test_environment()

class TestAPIRoutes:
    """Test suite for API routes"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user_info(self):
        """Mock user info for authentication"""
        return UserInfo(
            email="test@example.com",
            user_id="user123",
            scopes=["manage:policy"],
            token_payload={
                "userEmail": "test@example.com",
                "sub": "user123",
                "scope": "manage:policy",
                "iat": 1640995200,
                "exp": 1641081600
            }
        )
    
    @pytest.fixture
    def mock_auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer mock-jwt-token"}
    
    def test_health_endpoint_healthy(self, client):
        """Test health endpoint when services are healthy"""
        with patch('app.api.routes.get_store') as mock_get_store:
            mock_store = Mock()
            mock_store.client.info.return_value = {"cluster_name": "test"}
            mock_get_store.return_value = mock_store
            
            response = client.get("/api/aips/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["services"]["opensearch"] == "healthy"
            assert data["services"]["fastapi"] == "healthy"
    
    def test_health_endpoint_opensearch_unavailable(self, client):
        """Test health endpoint when OpenSearch is unavailable"""
        with patch('app.api.routes.get_store') as mock_get_store:
            mock_get_store.return_value = None
            
            response = client.get("/api/aips/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["services"]["opensearch"] == "unavailable"
    
    def test_health_endpoint_opensearch_error(self, client):
        """Test health endpoint when OpenSearch has error"""
        with patch('app.api.routes.get_store') as mock_get_store:
            mock_store = Mock()
            mock_store.client.info.side_effect = Exception("Connection failed")
            mock_get_store.return_value = mock_store
            
            response = client.get("/api/aips/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["services"]["opensearch"] == "error"
    
    def test_service_info_endpoint(self, client):
        """Test service info endpoint"""
        response = client.get("/api/aips/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service_name"] == "EDGP AI Policy Suggest Microservice"
        assert data["version"] == "1.0"
        assert "endpoints" in data
        assert "suggest_rules" in data["endpoints"]
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.get_schema_by_domain')
    @patch('app.api.routes.run_agent')
    def test_suggest_rules_success(self, mock_run_agent, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test successful rule suggestion"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_schema = {"column1": {"type": "string"}, "column2": {"type": "integer"}}
        mock_get_schema.return_value = mock_schema
        mock_run_agent.return_value = [{"rule": "test_rule"}]
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "rule_suggestions" in data
        assert data["rule_suggestions"] == [{"rule": "test_rule"}]
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.get_schema_by_domain')
    def test_suggest_rules_connection_failed(self, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test rule suggestion when vector DB connection fails"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_get_schema.side_effect = Exception("AuthorizationException: Access denied")
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Vector database connection failed"
        assert data["error_type"] == "connection_failed"
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.get_schema_by_domain')
    @patch('app.api.routes.bootstrap_schema_for_domain')
    def test_suggest_rules_domain_not_found(self, mock_bootstrap, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test rule suggestion when domain not found"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_get_schema.return_value = None
        mock_bootstrap.return_value = {"col1": {"type": "string"}, "col2": {"type": "integer"}}
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "Domain not found"
        assert "suggested_columns" in data
        assert data["suggested_columns"] == ["col1", "col2"]
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.get_schema_by_domain')
    def test_suggest_rules_internal_error(self, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test rule suggestion with internal error"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_get_schema.side_effect = Exception("Unexpected error")
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal server error"
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.get_store')
    @patch('app.api.routes.embed_column_names_batched_async')
    def test_create_domain_success_json(self, mock_embed, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test successful domain creation with JSON response"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = []  # No existing columns
        mock_store.upsert_columns.return_value = None
        mock_get_store.return_value = mock_store
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        payload = {
            "domain": "test_domain",
            "columns": ["col1", "col2"],
            "return_csv": False
        }
        
        response = client.post(
            "/api/aips/create/domain",
            json=payload,
            headers=mock_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["storage_status"] == "saved"
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.get_store')
    @patch('app.api.routes.embed_column_names_batched_async')
    def test_create_domain_success_csv(self, mock_embed, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test successful domain creation with CSV response"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = []
        mock_store.upsert_columns.return_value = None
        mock_get_store.return_value = mock_store
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        payload = {
            "domain": "test_domain",
            "columns": ["col1", "col2"],
            "return_csv": True
        }
        
        response = client.post(
            "/api/aips/create/domain",
            json=payload,
            headers=mock_auth_headers
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment" in response.headers["content-disposition"]
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.get_store')
    def test_create_domain_already_exists(self, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation when domain already exists"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = [{"column_name": "existing_col"}]
        mock_get_store.return_value = mock_store
        
        payload = {
            "domain": "existing_domain",
            "columns": ["col1", "col2"]
        }
        
        response = client.post(
            "/api/aips/create/domain",
            json=payload,
            headers=mock_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "exists"
        assert "existing_columns" in data
    
    @patch('app.auth.bearer.verify_any_scope_token')
    def test_create_domain_missing_domain(self, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation with missing domain field"""
        mock_auth.return_value = mock_user_info
        
        payload = {"columns": ["col1", "col2"]}
        
        response = client.post(
            "/api/aips/create/domain",
            json=payload,
            headers=mock_auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Missing required field: 'domain'"
    
    @patch('app.auth.bearer.verify_any_scope_token')
    def test_create_domain_invalid_columns_format(self, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation with invalid columns format"""
        mock_auth.return_value = mock_user_info
        
        payload = {
            "domain": "test_domain",
            "columns": "not_an_array"
        }
        
        response = client.post(
            "/api/aips/create/domain",
            json=payload,
            headers=mock_auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Invalid format: 'columns' must be an array"
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.embed_column_names_batched_async')
    def test_create_domain_embedding_failure(self, mock_embed, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation when embedding fails"""
        mock_auth.return_value = mock_user_info
        mock_embed.side_effect = Exception("Embedding service unavailable")
        
        payload = {
            "domain": "test_domain",
            "columns": ["col1", "col2"]
        }
        
        response = client.post(
            "/api/aips/create/domain",
            json=payload,
            headers=mock_auth_headers
        )
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Embedding generation failed"
    
    @patch('app.auth.bearer.verify_any_scope_token')
    @patch('app.api.routes.get_store')
    @patch('app.api.routes.embed_column_names_batched_async')
    def test_create_domain_storage_failure(self, mock_embed, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation when storage fails"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = []
        mock_store.upsert_columns.side_effect = Exception("Storage failed")
        mock_get_store.return_value = mock_store
        mock_embed.return_value = [[0.1, 0.2]]
        
        payload = {
            "domain": "test_domain",
            "columns": ["col1"]
        }
        
        response = client.post(
            "/api/aips/create/domain",
            json=payload,
            headers=mock_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["storage_status"] == "failed"
        assert "storage_error" in data
    
    @patch('app.api.routes.get_store')
    def test_vectordb_status_success(self, mock_get_store, client):
        """Test vector database status check success"""
        mock_store = Mock()
        mock_store.client.indices.exists.return_value = True
        mock_store.client.indices.stats.return_value = {
            "indices": {
                "test_index": {
                    "total": {
                        "docs": {"count": 100},
                        "store": {"size_in_bytes": 1024}
                    }
                }
            }
        }
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "connected"
        assert data["index_exists"] is True
        assert data["document_count"] == 100
    
    @patch('app.api.routes.get_store')
    def test_vectordb_status_store_unavailable(self, mock_get_store, client):
        """Test vector database status when store unavailable"""
        mock_get_store.return_value = None
        
        response = client.get("/api/aips/vectordb/status")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "error"
        assert data["connection"] == "failed"
    
    @patch('app.api.routes.get_store')
    def test_vectordb_status_connection_error(self, mock_get_store, client):
        """Test vector database status with connection error"""
        mock_store = Mock()
        mock_store.client.indices.exists.side_effect = Exception("Connection failed")
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/status")
        
        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert data["connection"] == "failed"
    
    @patch('app.api.routes.get_store')
    def test_list_domains_success(self, mock_get_store, client):
        """Test listing domains success"""
        mock_store = Mock()
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "column_name": "col1",
                            "metadata": {"domain": "domain1", "type": "string"}
                        }
                    },
                    {
                        "_source": {
                            "column_name": "col2",
                            "metadata": {"domain": "domain1", "type": "integer"}
                        }
                    }
                ]
            }
        }
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domains")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_domains"] == 1
        assert data["total_columns"] == 2
        assert "domain1" in data["domains"]
    
    @patch('app.api.routes.get_store')
    def test_list_domains_store_unavailable(self, mock_get_store, client):
        """Test listing domains when store unavailable"""
        mock_get_store.return_value = None
        
        response = client.get("/api/aips/vectordb/domains")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "OpenSearch store not available"
    
    @patch('app.api.routes.get_store')
    def test_get_domain_success(self, mock_get_store, client):
        """Test getting specific domain success"""
        mock_store = Mock()
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "column_name": "test_col",
                            "metadata": {"domain": "test_domain", "type": "string"},
                            "sample_values": ["val1", "val2"]
                        }
                    }
                ]
            }
        }
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domain/test_domain")
        
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "test_domain"
        assert data["found"] is True
        assert data["column_count"] == 1
    
    @patch('app.api.routes.get_store')
    def test_get_domain_not_found(self, mock_get_store, client):
        """Test getting domain when not found"""
        mock_store = Mock()
        mock_store.client.search.return_value = {
            "hits": {"total": {"value": 0}, "hits": []}
        }
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domain/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert data["domain"] == "nonexistent"
        assert data["found"] is False
    
    @patch('app.api.routes.get_store')
    def test_get_domain_store_unavailable(self, mock_get_store, client):
        """Test getting domain when store unavailable"""
        mock_get_store.return_value = None
        
        response = client.get("/api/aips/vectordb/domain/test_domain")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "OpenSearch store not available"
    
    @patch('app.api.routes.get_store')
    def test_get_domain_error(self, mock_get_store, client):
        """Test getting domain with error"""
        mock_store = Mock()
        mock_store.client.search.side_effect = Exception("Search failed")
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domain/test_domain")
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data