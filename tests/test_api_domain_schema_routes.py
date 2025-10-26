"""
Unit tests for domain schema routes
Tests routes defined in app.api.domain_schema_routes
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from app.api.domain_schema_routes import (
    router, get_store, is_safe_filename, register_csv_file,
    clear_file_mappings, get_test_file_id, download_csv_file
)

class TestDomainSchemaRoutes:
    """Test domain schema routes functionality"""

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_create_domain_missing_domain(self, mock_get_store, test_client):
        """Test create domain endpoint with missing domain field"""
        response = test_client.post(
            "/api/aips/domains/create",
            json={"columns": ["col1", "col2"]}
        )
        assert response.status_code == 400
        assert "Missing required field: 'domain'" in response.json()["error"]

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    @patch('app.api.domain_schema_routes.embed_column_names_batched_async')
    async def test_create_domain_success(self, mock_embed, mock_get_store, test_client, mock_store):
        """Test successful domain creation"""
        mock_get_store.return_value = mock_store
        mock_store.check_domain_exists_case_insensitive.return_value = {"exists": False}
        mock_embed.return_value = [[1.0] * 1536] * 2  # Mock embeddings

        response = test_client.post(
            "/api/aips/domains/create",
            json={
                "domain": "test_domain",
                "columns": ["col1", "col2"]
            }
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["domain"] == "test_domain"

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_create_domain_store_unavailable(self, mock_get_store, test_client):
        """Test domain creation when store is unavailable"""
        mock_get_store.return_value = None

        response = test_client.post(
            "/api/aips/domains/create",
            json={
                "domain": "test_domain",
                "columns": ["col1", "col2"]
            }
        )
        assert response.status_code == 503

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_get_domains_success(self, mock_get_store, test_client, mock_store):
        """Test successful domain list retrieval"""
        mock_get_store.return_value = mock_store
        mock_store.get_all_domains_realtime.return_value = ["domain1", "domain2"]

        response = test_client.get("/api/aips/domains")
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert len(response.json()["data"]) == 2

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_verify_domain_exists(self, mock_get_store, test_client, mock_store):
        """Test domain verification endpoint"""
        mock_get_store.return_value = mock_store
        mock_store.get_all_domains_realtime.return_value = ["test_domain"]

        response = test_client.get("/api/aips/domains/verify/test_domain")
        assert response.status_code == 200
        assert response.json()["exists"] is True

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_list_domains_in_vectordb(self, mock_get_store, test_client, mock_store):
        """Test listing domains from vector DB"""
        mock_get_store.return_value = mock_store
        mock_store.client.indices.exists.return_value = True
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "metadata": {"domain": "test_domain"},
                        "column_name": "test_col",
                        "type": "string"
                    }
                }]
            }
        }

        response = test_client.get("/api/aips/domains/schema")
        assert response.status_code == 200
        assert "domains" in response.json()

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_get_domain_from_vectordb(self, mock_get_store, test_client, mock_store):
        """Test getting specific domain from vector DB"""
        mock_get_store.return_value = mock_store
        mock_store.client.indices.exists.return_value = True
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "column_name": "test_col",
                        "metadata": {"type": "string"},
                        "sample_values": []
                    }
                }]
            }
        }

        response = test_client.get("/api/aips/domains/test_domain")
        assert response.status_code == 200
        assert response.json()["found"] is True

    def test_is_safe_filename(self):
        """Test filename safety validation"""
        assert is_safe_filename("test.csv") is True
        assert is_safe_filename("test-file.csv") is True
        assert is_safe_filename("test_file.csv") is True
        assert is_safe_filename("../test.csv") is False
        assert is_safe_filename("test.txt") is False

    def test_register_csv_file(self):
        """Test CSV file registration"""
        clear_file_mappings()  # Start with clean state
        file_id = register_csv_file("test.csv", "/tmp/test.csv")
        assert file_id is not None
        clear_file_mappings()

    @pytest.mark.asyncio
    async def test_download_csv_file(self, test_client):
        """Test CSV file download endpoint"""
        # Create a temporary file for testing
        import os
        import tempfile
        from pathlib import Path

        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.csv")
            
            # Write some test content
            with open(file_path, "w") as f:
                f.write("test,data\n1,2\n")

            # Register the file
            file_id = register_csv_file("test.csv", file_path)
            
            # Test the download
            response = test_client.get(f"/api/aips/domains/download-csv/{file_id}")
            assert response.status_code == 200
            clear_file_mappings()

    @pytest.mark.asyncio
    @patch('app.agents.schema_suggester.SchemaSuggesterEnhanced')
    async def test_regenerate_suggestions(self, mock_suggester, test_client):
        """Test schema suggestion regeneration"""
        mock_suggester_instance = Mock()
        mock_suggester_instance.bootstrap_schema_with_preferences = AsyncMock(
            return_value={"columns": [{"column_name": "suggested_col"}]}
        )
        mock_suggester.return_value = mock_suggester_instance

        response = test_client.post(
            "/api/aips/domains/suggest-schema",
            json={"domain": "test_domain"}
        )
        assert response.status_code == 200
        assert "suggested_columns" in response.json()

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    @patch('app.api.domain_schema_routes.embed_column_names_batched_async')
    async def test_extend_domain(self, mock_embed, mock_get_store, test_client, mock_store):
        """Test domain extension endpoint"""
        mock_get_store.return_value = mock_store
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "column_name": "existing_col",
                        "metadata": {"type": "string"}
                    }
                }]
            }
        }
        mock_embed.return_value = [[1.0] * 1536]

        response = test_client.put(
            "/api/aips/domains/extend-schema",
            json={
                "domain": "test_domain",
                "columns": ["new_col"]
            }
        )
        assert response.status_code == 200
        assert "extension_summary" in response.json()

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    @patch('app.agents.schema_suggester.SchemaSuggesterEnhanced')
    async def test_suggest_extensions(self, mock_suggester, mock_get_store, test_client, mock_store):
        """Test domain extension suggestions"""
        mock_get_store.return_value = mock_store
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "column_name": "existing_col",
                        "metadata": {"type": "string"}
                    }
                }]
            }
        }

        mock_suggester_instance = Mock()
        mock_suggester_instance.bootstrap_schema_with_preferences = AsyncMock(
            return_value={"columns": [{"column_name": "suggested_col"}]}
        )
        mock_suggester.return_value = mock_suggester_instance

        response = test_client.post("/api/aips/domains/suggest-extend-schema/test_domain")
        assert response.status_code == 200
        assert "suggestions" in response.json()

# Additional test classes for error cases and edge conditions
class TestErrorHandling:
    """Test error handling in domain schema routes"""

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_create_domain_duplicate(self, mock_get_store, test_client, mock_store):
        """Test handling of duplicate domain creation"""
        mock_get_store.return_value = mock_store
        mock_store.check_domain_exists_case_insensitive.return_value = {
            "exists": True,
            "existing_domain": "test_domain"
        }
        mock_store.get_columns_by_domain.return_value = [{"column_name": "existing_col"}]

        response = test_client.post(
            "/api/aips/domains/create",
            json={
                "domain": "test_domain",
                "columns": ["col1", "col2"]
            }
        )
        assert response.status_code == 409
        assert "exists" in response.json()["status"]

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_create_domain_invalid_format(self, mock_get_store, test_client):
        """Test handling of invalid request format"""
        response = test_client.post(
            "/api/aips/domains/create",
            json={
                "domain": "test_domain",
                "invalid_field": ["col1", "col2"]
            }
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    @patch('app.api.domain_schema_routes.embed_column_names_batched_async')
    async def test_create_domain_embedding_failure(self, mock_embed, mock_get_store, test_client, mock_store):
        """Test handling of embedding generation failure"""
        mock_get_store.return_value = mock_store
        mock_store.check_domain_exists_case_insensitive.return_value = {"exists": False}
        mock_embed.side_effect = Exception("Embedding failed")

        response = test_client.post(
            "/api/aips/domains/create",
            json={
                "domain": "test_domain",
                "columns": ["col1", "col2"]
            }
        )
        assert response.status_code == 503
        assert "Embedding generation failed" in response.json()["error"]

class TestEdgeCases:
    """Test edge cases in domain schema routes"""

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    @patch('app.api.domain_schema_routes.embed_column_names_batched_async')
    @patch('app.agents.agent_runner.run_agent')
    async def test_create_domain_empty_columns(self, mock_run_agent, mock_embed, mock_get_store, test_client, mock_store):
        """Test creation with empty column list"""
        mock_get_store.return_value = mock_store
        mock_store.check_domain_exists_case_insensitive.return_value = {"exists": False}
        mock_store.get_all_domains_realtime.return_value = []
        mock_store.force_refresh_index.return_value = True
        mock_store.client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
        mock_embed.return_value = []  # No embeddings needed for empty columns
        mock_run_agent.return_value = []  # No rules generated
        
        response = test_client.post(
            "/api/aips/domains/create",
            json={
                "domain": "test_domain",
                "columns": []
            }
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    async def test_extend_domain_all_duplicates(self, mock_get_store, test_client, mock_store):
        """Test extension when all new columns are duplicates"""
        mock_get_store.return_value = mock_store
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "column_name": "col1",
                        "metadata": {"type": "string"}
                    }
                }]
            }
        }

        response = test_client.put(
            "/api/aips/domains/extend-schema",
            json={
                "domain": "test_domain",
                "columns": ["col1"]
            }
        )
        assert response.status_code == 400
        assert "all requested columns are duplicates" in response.json()["message"].lower()

    def test_similar_column_names(self):
        """Test similar column name detection"""
        from app.api.domain_schema_routes import _is_similar_column_name
        
        # Direct substring matches
        assert _is_similar_column_name("user_id", "user_id") is True
        assert _is_similar_column_name("customer", "customer_name") is True
        assert _is_similar_column_name("order_date", "date") is True
        
        # With common prefixes/suffixes removed
        assert _is_similar_column_name("is_active", "active") is True
        assert _is_similar_column_name("user_name", "name") is True
        assert _is_similar_column_name("customer_id", "customer") is True
        
        # Different columns should not match
        assert _is_similar_column_name("completely_different", "another_field") is False
        assert _is_similar_column_name("abc", "xyz") is False

class TestPerformance:
    """Test performance aspects of domain schema routes"""

    @pytest.mark.asyncio
    @patch('app.api.domain_schema_routes.get_store')
    @patch('app.api.domain_schema_routes.embed_column_names_batched_async')
    async def test_large_domain_creation(self, mock_embed, mock_get_store, test_client, mock_store):
        """Test creation of domain with large number of columns"""
        mock_get_store.return_value = mock_store
        mock_store.check_domain_exists_case_insensitive.return_value = {"exists": False}
        mock_embed.return_value = [[1.0] * 1536] * 100  # Mock embeddings for 100 columns

        large_columns = [f"col_{i}" for i in range(100)]
        response = test_client.post(
            "/api/aips/domains/create",
            json={
                "domain": "large_test_domain",
                "columns": large_columns
            }
        )
        assert response.status_code == 200
        assert response.json()["columns_created"] == 100

if __name__ == "__main__":
    pytest.main([__file__, "-v"])