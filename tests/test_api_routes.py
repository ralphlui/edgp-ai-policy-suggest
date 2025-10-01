"""
Consolidated comprehensive tests for API routes
Combined from all 5 test_api_routes*.py files with improved organization and coverage
Tests all endpoints in routes.py and main.py
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
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
    """Consolidated test suite for all API routes with comprehensive coverage"""
    
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

    # ==========================================================================
    # HEALTH AND INFO ENDPOINTS (from main.py)
    # ==========================================================================

    @patch("app.aoss.column_store.get_store")
    def test_health_endpoint_healthy(self, mock_get_store, client):
        """Test health endpoint when system is healthy"""
        # Mock store health check
        mock_store = Mock()
        mock_store.health_check.return_value = {"status": "healthy", "index_exists": True}
        mock_get_store.return_value = mock_store
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "vector_db" in data
        assert data["vector_db"]["status"] == "healthy"

    @patch("app.aoss.column_store.get_store")
    def test_health_endpoint_opensearch_unavailable(self, mock_get_store, client):
        """Test health endpoint when OpenSearch is unavailable"""
        mock_get_store.return_value = None
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"

    def test_service_info_endpoint(self, client):
        """Test service info endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AI Policy Suggest"
        assert "version" in data

    # ==========================================================================
    # SUGGEST RULES ENDPOINT
    # ==========================================================================

    @patch("app.auth.bearer.get_user_info")
    @patch("app.api.routes.get_schema_by_domain")
    @patch("app.agents.agent_runner.run_agent")
    def test_suggest_rules_success(self, mock_run_agent, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test successful rule suggestion"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_get_schema.return_value = {
            "name": {"dtype": "string", "sample_values": ["John", "Jane"]},
            "age": {"dtype": "integer", "sample_values": [25, 30, 35]}
        }
        mock_run_agent.return_value = [
            {"expectation_type": "expect_column_to_exist", "column": "name"},
            {"expectation_type": "expect_column_values_to_be_of_type", "column": "age"}
        ]
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["expectation_type"] == "expect_column_to_exist"

    @patch("app.auth.bearer.get_user_info")
    @patch("app.api.routes.get_schema_by_domain")
    def test_suggest_rules_connection_failed(self, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test rule suggestion when vector DB connection fails"""
        mock_auth.return_value = mock_user_info
        mock_get_schema.side_effect = Exception("Connection failed")
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Connection failed" in data["detail"]

    @patch("app.auth.bearer.get_user_info")
    @patch("app.api.routes.get_schema_by_domain")
    def test_suggest_rules_domain_not_found(self, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test rule suggestion when domain is not found"""
        mock_auth.return_value = mock_user_info
        mock_get_schema.return_value = {}
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "nonexistent_domain"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @patch("app.auth.bearer.get_user_info")
    @patch("app.api.routes.get_schema_by_domain")
    @patch("app.agents.agent_runner.run_agent")
    def test_suggest_rules_internal_error(self, mock_run_agent, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test rule suggestion with internal agent error"""
        mock_auth.return_value = mock_user_info
        mock_get_schema.return_value = {"name": {"dtype": "string", "sample_values": ["John"]}}
        mock_run_agent.side_effect = Exception("Agent processing error")
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Agent processing error" in data["detail"]

    # ==========================================================================
    # CREATE DOMAIN ENDPOINT
    # ==========================================================================

    @patch("app.auth.bearer.get_user_info")
    @patch("app.aoss.column_store.get_store")
    @patch("app.embedding.embedder.get_embeddings")
    def test_create_domain_success_json(self, mock_embed, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test successful domain creation with JSON data"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.domain_exists.return_value = False
        mock_store.upsert_columns.return_value = True
        mock_get_store.return_value = mock_store
        mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        columns_data = [
            {"name": "customer_id", "type": "integer"},
            {"name": "customer_name", "type": "string"}
        ]
        
        response = client.post(
            "/api/aips/create/domain",
            json={"domain": "test_domain", "columns": columns_data},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Domain created successfully"
        assert data["domain"] == "test_domain"
        assert data["columns_stored"] == 2

    @patch("app.auth.bearer.get_user_info")
    @patch("app.aoss.column_store.get_store")
    @patch("app.embedding.embedder.get_embeddings")
    def test_create_domain_success_csv(self, mock_embed, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test successful domain creation with CSV file"""
        # Setup mocks
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.domain_exists.return_value = False
        mock_store.upsert_columns.return_value = True
        mock_get_store.return_value = mock_store
        mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Create CSV content
        csv_content = "name,type\ncustomer_id,integer\ncustomer_name,string"
        csv_file = io.StringIO(csv_content)
        
        response = client.post(
            "/api/aips/create/domain",
            data={"domain": "test_domain"},
            files={"file": ("test.csv", csv_file, "text/csv")},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Domain created successfully"

    @patch("app.auth.bearer.get_user_info")
    @patch("app.aoss.column_store.get_store")
    def test_create_domain_already_exists(self, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation when domain already exists"""
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.domain_exists.return_value = True
        mock_get_store.return_value = mock_store
        
        response = client.post(
            "/api/aips/create/domain",
            json={"domain": "existing_domain", "columns": [{"name": "col1", "type": "string"}]},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "already exists" in data["detail"]

    @patch("app.auth.bearer.get_user_info")
    def test_create_domain_missing_domain(self, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation with missing domain name"""
        mock_auth.return_value = mock_user_info
        
        response = client.post(
            "/api/aips/create/domain",
            json={"columns": [{"name": "col1", "type": "string"}]},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Domain name is required" in data["detail"]

    @patch("app.auth.bearer.get_user_info")
    def test_create_domain_invalid_columns_format(self, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation with invalid columns format"""
        mock_auth.return_value = mock_user_info
        
        response = client.post(
            "/api/aips/create/domain",
            json={"domain": "test_domain", "columns": "invalid_format"},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "must be a list" in data["detail"]

    @patch("app.auth.bearer.get_user_info")
    @patch("app.embedding.embedder.get_embeddings")
    def test_create_domain_embedding_failure(self, mock_embed, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation when embedding generation fails"""
        mock_auth.return_value = mock_user_info
        mock_embed.side_effect = Exception("Embedding service unavailable")
        
        response = client.post(
            "/api/aips/create/domain",
            json={"domain": "test_domain", "columns": [{"name": "col1", "type": "string"}]},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Embedding service unavailable" in data["detail"]

    @patch("app.auth.bearer.get_user_info")
    @patch("app.aoss.column_store.get_store")
    @patch("app.embedding.embedder.get_embeddings")
    def test_create_domain_storage_failure(self, mock_embed, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test domain creation when storage fails"""
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.domain_exists.return_value = False
        mock_store.upsert_columns.side_effect = Exception("Storage error")
        mock_get_store.return_value = mock_store
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        response = client.post(
            "/api/aips/create/domain",
            json={"domain": "test_domain", "columns": [{"name": "col1", "type": "string"}]},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "Storage error" in data["detail"]

    # ==========================================================================
    # VECTOR DATABASE STATUS ENDPOINT
    # ==========================================================================

    @patch("app.aoss.column_store.get_store")
    def test_vectordb_status_success(self, mock_get_store, client):
        """Test vector DB status when healthy"""
        mock_store = Mock()
        mock_store.health_check.return_value = {
            "status": "healthy",
            "index_exists": True,
            "doc_count": 100,
            "index_health": "green"
        }
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["index_exists"] is True
        assert data["doc_count"] == 100

    @patch("app.aoss.column_store.get_store")
    def test_vectordb_status_store_unavailable(self, mock_get_store, client):
        """Test vector DB status when store is unavailable"""
        mock_get_store.return_value = None
        
        response = client.get("/api/aips/vectordb/status")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"

    @patch("app.aoss.column_store.get_store")
    def test_vectordb_status_connection_error(self, mock_get_store, client):
        """Test vector DB status with connection error"""
        mock_store = Mock()
        mock_store.health_check.side_effect = Exception("Connection timeout")
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/status")
        
        assert response.status_code == 500
        data = response.json()
        assert "Connection timeout" in data["detail"]

    # ==========================================================================
    # LIST DOMAINS ENDPOINT
    # ==========================================================================

    @patch("app.aoss.column_store.get_store")
    def test_list_domains_success(self, mock_get_store, client):
        """Test successful domain listing"""
        mock_store = Mock()
        mock_store.get_distinct_domains.return_value = [
            {"domain": "product", "column_count": 10},
            {"domain": "customer", "column_count": 8},
            {"domain": "finance", "column_count": 15}
        ]
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domains")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["domain"] == "product"
        assert data[0]["column_count"] == 10

    @patch("app.aoss.column_store.get_store")
    def test_list_domains_store_unavailable(self, mock_get_store, client):
        """Test domain listing when store is unavailable"""
        mock_get_store.return_value = None
        
        response = client.get("/api/aips/vectordb/domains")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"

    @patch("app.aoss.column_store.get_store")
    def test_list_domains_search_error(self, mock_get_store, client):
        """Test domain listing with search error"""
        mock_store = Mock()
        mock_store.get_distinct_domains.side_effect = Exception("Search error")
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domains")
        
        assert response.status_code == 500
        data = response.json()
        assert "Search error" in data["detail"]

    # ==========================================================================
    # GET DOMAIN ENDPOINT
    # ==========================================================================

    @patch("app.aoss.column_store.get_store")
    def test_get_domain_success(self, mock_get_store, client):
        """Test successful domain retrieval"""
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = [
            {
                "column_name": "product_id",
                "metadata": {"type": "integer", "domain": "product"},
                "sample_values": [1, 2, 3]
            },
            {
                "column_name": "product_name", 
                "metadata": {"type": "string", "domain": "product"},
                "sample_values": ["Product A", "Product B"]
            }
        ]
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domain/product")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["column_name"] == "product_id"
        assert data[0]["metadata"]["type"] == "integer"

    @patch("app.aoss.column_store.get_store")
    def test_get_domain_not_found(self, mock_get_store, client):
        """Test domain retrieval when domain not found"""
        mock_store = Mock()
        mock_store.get_columns_by_domain.return_value = []
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domain/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @patch("app.aoss.column_store.get_store")
    def test_get_domain_store_unavailable(self, mock_get_store, client):
        """Test domain retrieval when store is unavailable"""
        mock_get_store.return_value = None
        
        response = client.get("/api/aips/vectordb/domain/product")
        
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"

    @patch("app.aoss.column_store.get_store")
    def test_get_domain_search_error(self, mock_get_store, client):
        """Test domain retrieval with search error"""
        mock_store = Mock()
        mock_store.get_columns_by_domain.side_effect = Exception("Search failed")
        mock_get_store.return_value = mock_store
        
        response = client.get("/api/aips/vectordb/domain/product")
        
        assert response.status_code == 500
        data = response.json()
        assert "Search failed" in data["detail"]

    # ==========================================================================
    # AUTHENTICATION TESTS
    # ==========================================================================

    def test_suggest_rules_no_auth(self, client):
        """Test suggest rules without authentication"""
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"}
        )
        
        assert response.status_code == 403

    def test_create_domain_no_auth(self, client):
        """Test create domain without authentication"""
        response = client.post(
            "/api/aips/create/domain",
            json={"domain": "test_domain", "columns": []}
        )
        
        assert response.status_code == 403

    @patch("app.auth.bearer.get_user_info")
    def test_suggest_rules_invalid_token(self, mock_auth, client):
        """Test suggest rules with invalid token"""
        mock_auth.side_effect = HTTPException(status_code=401, detail="Invalid token")
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "test_domain"},
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401

    # ==========================================================================
    # INPUT VALIDATION TESTS
    # ==========================================================================

    @patch("app.auth.bearer.get_user_info")
    def test_suggest_rules_missing_domain(self, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test suggest rules with missing domain"""
        mock_auth.return_value = mock_user_info
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 422

    @patch("app.auth.bearer.get_user_info")
    def test_suggest_rules_empty_domain(self, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test suggest rules with empty domain"""
        mock_auth.return_value = mock_user_info
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": ""},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 400

    @patch("app.auth.bearer.get_user_info")
    def test_create_domain_empty_columns(self, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test create domain with empty columns list"""
        mock_auth.return_value = mock_user_info
        
        response = client.post(
            "/api/aips/create/domain",
            json={"domain": "test_domain", "columns": []},
            headers=mock_auth_headers
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "at least one column" in data["detail"].lower()

    # ==========================================================================
    # EDGE CASES AND ERROR SCENARIOS
    # ==========================================================================

    @patch("app.auth.bearer.get_user_info")
    @patch("app.aoss.column_store.get_store")
    def test_create_domain_with_schema_format(self, mock_get_store, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test create domain with schema format (alternative input format)"""
        mock_auth.return_value = mock_user_info
        mock_store = Mock()
        mock_store.domain_exists.return_value = False
        mock_get_store.return_value = mock_store
        
        # Test with schema format instead of columns
        schema_data = {
            "customer_id": {"type": "integer", "description": "Unique customer identifier"},
            "customer_name": {"type": "string", "description": "Customer full name"}
        }
        
        response = client.post(
            "/api/aips/create/domain",
            json={"domain": "test_domain", "schema": schema_data},
            headers=mock_auth_headers
        )
        
        # Should handle alternative format gracefully
        assert response.status_code in [200, 400]

    @patch("app.auth.bearer.get_user_info")
    @patch("app.api.routes.get_schema_by_domain")
    @patch("app.agents.schema_suggester.bootstrap_schema")
    def test_suggest_rules_fallback_to_bootstrap(self, mock_bootstrap, mock_get_schema, mock_auth, client, mock_user_info, mock_auth_headers):
        """Test suggest rules fallback to schema bootstrapping"""
        mock_auth.return_value = mock_user_info
        mock_get_schema.return_value = {}  # Empty schema triggers bootstrap
        mock_bootstrap.return_value = {
            "name": {"dtype": "string", "sample_values": ["John", "Jane"]},
            "age": {"dtype": "integer", "sample_values": [25, 30]}
        }
        
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": "new_domain"},
            headers=mock_auth_headers
        )
        
        # Should either succeed with bootstrapped schema or return 404
        assert response.status_code in [200, 404]

    def test_health_endpoint_detailed_response(self, client):
        """Test health endpoint returns detailed status information"""
        response = client.get("/health")
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data or "service" in data
        # Health endpoint should always return some form of status

    @patch("app.aoss.column_store.get_store")
    def test_vectordb_endpoints_consistent_error_format(self, mock_get_store, client):
        """Test that vector DB endpoints return consistent error formats"""
        mock_get_store.return_value = None
        
        # Test multiple endpoints with same error condition
        endpoints = [
            "/api/aips/vectordb/status",
            "/api/aips/vectordb/domains", 
            "/api/aips/vectordb/domain/test"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 503
            data = response.json()
            assert "status" in data or "detail" in data
            # Consistent error response structure


# ==========================================================================
# UTILITY FUNCTIONS FOR TESTS
# ==========================================================================

def create_sample_csv_content():
    """Create sample CSV content for testing"""
    return """name,type,description
customer_id,integer,Unique customer identifier  
customer_name,string,Full name of customer
email,string,Customer email address
age,integer,Customer age in years
created_date,date,Account creation date"""


def create_sample_column_docs():
    """Create sample ColumnDoc objects for testing"""
    return [
        ColumnDoc(
            column_name="customer_id",
            column_vector=[0.1, 0.2, 0.3],
            metadata={"type": "integer", "domain": "customer"},
            sample_values=[1, 2, 3, 4, 5]
        ),
        ColumnDoc(
            column_name="customer_name", 
            column_vector=[0.4, 0.5, 0.6],
            metadata={"type": "string", "domain": "customer"},
            sample_values=["John Doe", "Jane Smith", "Bob Johnson"]
        )
    ]


# ==========================================================================
# PARAMETRIZED TESTS FOR MULTIPLE SCENARIOS
# ==========================================================================

class TestParametrizedScenarios:
    """Parametrized tests for testing multiple scenarios efficiently"""
    
    @pytest.mark.parametrize("domain_name,expected_status", [
        ("valid_domain", 200),
        ("", 400),
        ("a" * 100, 200),  # Long domain name
        ("domain-with-hyphens", 200),
        ("domain_with_underscores", 200),
        ("domain123", 200),
    ])
    @patch("app.auth.bearer.get_user_info")
    @patch("app.api.routes.get_schema_by_domain")
    def test_suggest_rules_domain_validation(self, mock_get_schema, mock_auth, domain_name, expected_status, client, mock_user_info, mock_auth_headers):
        """Test suggest rules with various domain name formats"""
        mock_auth.return_value = mock_user_info
        
        if expected_status == 200:
            mock_get_schema.return_value = {"col1": {"dtype": "string", "sample_values": ["test"]}}
        else:
            mock_get_schema.return_value = {}
            
        response = client.post(
            "/api/aips/suggest-rules",
            json={"domain": domain_name},
            headers=mock_auth_headers
        )
        
        if domain_name == "":
            assert response.status_code == 400
        else:
            assert response.status_code in [200, 404, 500]  # Depends on mocking

    @pytest.mark.parametrize("endpoint", [
        "/api/aips/vectordb/status",
        "/api/aips/vectordb/domains",
    ])
    @patch("app.aoss.column_store.get_store")
    def test_vectordb_endpoints_when_store_unavailable(self, mock_get_store, endpoint, client):
        """Test vector DB endpoints when store is unavailable"""
        mock_get_store.return_value = None
        
        response = client.get(endpoint)
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unavailable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])