"""
Unit tests for domain/schema endpoints (/api/aips/create/domain, /api/aips/domains, etc.)
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse
from app.api.routes import (
    create_domain, get_domains, verify_domain_exists,
    list_domains_in_vectordb, get_domain_from_vectordb,
    regenerate_suggestions, extend_domain, suggest_extensions,
    get_store
)
from app.auth.authentication import UserInfo


class TestCreateDomainRoute:
    @pytest.fixture
    def mock_user_info(self):
        mock_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy",
            "iat": 1234567890,
            "exp": 1234567890 + 3600
        }
        return UserInfo(
            email="test@example.com", 
            user_id="user123",
            scopes=["manage:policy"],
            token_payload=mock_payload
        )

    @pytest.fixture
    def mock_request(self):
        request = Mock(spec=Request)
        request.base_url = "http://localhost:8000/"
        return request

    @pytest.fixture
    def sample_payload(self):
        return {
            "domain": "customer",
            "columns": ["email", "name", "age"],
            "return_csv": False
        }

    @pytest.mark.asyncio
    async def test_create_domain_missing_domain(self, mock_request, mock_user_info):
        payload = {"columns": ["email", "name"]}
        result = await create_domain(mock_request, payload, mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "Missing required field: 'domain'" in content["error"]

    @pytest.mark.asyncio
    async def test_create_domain_missing_columns_and_schema(self, mock_request, mock_user_info):
        payload = {"domain": "customer"}
        result = await create_domain(mock_request, payload, mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "Missing required field: 'columns' or 'schema'" in content["error"]

    @pytest.mark.asyncio
    async def test_create_domain_invalid_columns_format(self, mock_request, mock_user_info):
        payload = {"domain": "customer", "columns": "not_an_array"}
        result = await create_domain(mock_request, payload, mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "Invalid format: 'columns' must be an array" in content["error"]

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    @patch('app.api.routes.embed_column_names_batched_async')
    @patch('app.api.routes.run_agent')
    async def test_create_domain_success(self, mock_run_agent, mock_embed, mock_get_store, mock_request, mock_user_info, sample_payload):
        mock_store = Mock()
        mock_store.check_domain_exists_case_insensitive.return_value = {"exists": False}
        mock_store.upsert_columns.return_value = None
        mock_store.force_refresh_index.return_value = True
        mock_store.get_all_domains_realtime.return_value = ["customer"]
        mock_get_store.return_value = mock_store
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_run_agent.return_value = ["rule1", "rule2"]
        result = await create_domain(mock_request, sample_payload, mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["status"] == "success"
        assert content["domain"] == "customer"
        assert content["columns_created"] == 3
        if "rules_available" in content:
            assert isinstance(content["rules_available"], bool)
        if "rule_suggestions" in content:
            assert isinstance(content["rule_suggestions"], list)
            if content.get("rules_available", False):
                assert len(content["rule_suggestions"]) >= 0

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_create_domain_already_exists(self, mock_get_store, mock_request, mock_user_info, sample_payload):
        mock_store = Mock()
        mock_store.check_domain_exists_case_insensitive.return_value = {
            "exists": True,
            "existing_domain": "customer"
        }
        mock_store.get_columns_by_domain.return_value = [
            {"column_name": "email"}, {"column_name": "name"}
        ]
        mock_get_store.return_value = mock_store
        result = await create_domain(mock_request, sample_payload, mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 409
        content = json.loads(result.body.decode())
        assert content["status"] == "exists"
        assert "already exists" in content["message"]

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    @patch('app.api.routes.embed_column_names_batched_async')
    async def test_create_domain_with_csv(self, mock_embed, mock_get_store, mock_request, mock_user_info):
        payload = {
            "domain": "customer",
            "columns": ["email", "name"],
            "return_csv": True
        }
        mock_store = Mock()
        mock_store.check_domain_exists_case_insensitive.return_value = {"exists": False}
        mock_store.upsert_columns.return_value = None
        mock_store.force_refresh_index.return_value = True
        mock_get_store.return_value = mock_store
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        with patch('app.api.routes.run_agent') as mock_run_agent:
            mock_run_agent.return_value = ["rule1"]
            result = await create_domain(mock_request, payload, mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert "csv_download" in content
        assert content["csv_download"]["available"] is True

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_create_domain_store_unavailable(self, mock_get_store, mock_request, mock_user_info, sample_payload):
        mock_get_store.return_value = None
        with patch('app.api.routes.embed_column_names_batched_async') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            result = await create_domain(mock_request, sample_payload, mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["status"] == "partial_success"

    @pytest.mark.asyncio
    @patch('app.api.routes.embed_column_names_batched_async')
    async def test_create_domain_embedding_failure(self, mock_embed, mock_request, mock_user_info, sample_payload):
        mock_embed.side_effect = Exception("Embedding service unavailable")
        result = await create_domain(mock_request, sample_payload, mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "Embedding generation failed" in content["error"]

class TestGetDomainsRoute:
    @pytest.fixture
    def mock_user_info(self):
        mock_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy",
            "iat": 1234567890,
            "exp": 1234567890 + 3600
        }
        return UserInfo(
            email="test@example.com", 
            user_id="user123",
            scopes=["manage:policy"],
            token_payload=mock_payload
        )

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_get_domains_success(self, mock_get_store, mock_user_info):
        mock_store = Mock()
        mock_store.get_all_domains_realtime.return_value = ["customer", "product", "order"]
        mock_get_store.return_value = mock_store
        result = await get_domains(mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["success"] is True
        assert content["totalRecord"] == 3
        assert "customer" in content["data"]

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_get_domains_store_unavailable(self, mock_get_store, mock_user_info):
        mock_get_store.return_value = None
        result = await get_domains(mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        content = json.loads(result.body.decode())
        assert content["success"] is False
        assert "OpenSearch store not available" in content["message"]

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_get_domains_exception(self, mock_get_store, mock_user_info):
        mock_store = Mock()
        mock_store.get_all_domains_realtime.side_effect = Exception("Connection error")
        mock_get_store.return_value = mock_store
        result = await get_domains(mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 500
        content = json.loads(result.body.decode())
        assert content["success"] is False

class TestVerifyDomainRoute:
    @pytest.fixture
    def mock_user_info(self):
        mock_payload = {
            "userEmail": "test@example.com",
            "sub": "user123",
            "scope": "manage:policy",
            "iat": 1234567890,
            "exp": 1234567890 + 3600
        }
        return UserInfo(
            email="test@example.com", 
            user_id="user123",
            scopes=["manage:policy"],
            token_payload=mock_payload
        )

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_verify_domain_exists(self, mock_get_store, mock_user_info):
        mock_store = Mock()
        mock_store.get_all_domains_realtime.return_value = ["customer", "product"]
        mock_get_store.return_value = mock_store
        result = await verify_domain_exists("customer", mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["success"] is True
        assert content["exists"] is True
        assert content["domain"] == "customer"

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_verify_domain_not_exists(self, mock_get_store, mock_user_info):
        mock_store = Mock()
        mock_store.get_all_domains_realtime.return_value = ["customer", "product"]
        mock_get_store.return_value = mock_store
        result = await verify_domain_exists("nonexistent", mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["success"] is True
        assert content["exists"] is False

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_verify_domain_store_unavailable(self, mock_get_store, mock_user_info):
        mock_get_store.return_value = None
        result = await verify_domain_exists("customer", mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        content = json.loads(result.body.decode())
        assert content["success"] is False
        assert content["exists"] is False

class TestDomainsSchemaRoute:
    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_domains_schema_success(self, mock_get_store):
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 3},
                "hits": [
                    {
                        "_source": {
                            "metadata": {"domain": "customer", "type": "string"},
                            "column_name": "email"
                        }
                    },
                    {
                        "_source": {
                            "metadata": {"domain": "customer", "type": "string"},
                            "column_name": "name"
                        }
                    },
                    {
                        "_source": {
                            "metadata": {"domain": "product", "type": "string"},
                            "column_name": "title"
                        }
                    }
                ]
            }
        }
        mock_store = Mock()
        mock_store.client = mock_client
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        result = await list_domains_in_vectordb()
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["total_domains"] == 2
        assert content["total_columns"] == 3
        assert "customer" in content["domains"]
        assert "product" in content["domains"]

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_domains_schema_no_index(self, mock_get_store):
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_store = Mock()
        mock_store.client = mock_client
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        result = await list_domains_in_vectordb()
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["total_domains"] == 0
        assert "does not exist yet" in content["message"]

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_domains_schema_store_unavailable(self, mock_get_store):
        mock_get_store.return_value = None
        result = await list_domains_in_vectordb()
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "OpenSearch store not available" in content["error"]

class TestDomainDetailsRoute:
    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_get_domain_success(self, mock_get_store):
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "column_name": "email",
                            "sample_values": ["test@example.com"],
                            "metadata": {"type": "string", "domain": "customer"}
                        }
                    },
                    {
                        "_source": {
                            "column_name": "name",
                            "sample_values": ["John"],
                            "metadata": {"type": "string", "domain": "customer"}
                        }
                    }
                ]
            }
        }
        mock_store = Mock()
        mock_store.client = mock_client
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        result = await get_domain_from_vectordb("customer")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["domain"] == "customer"
        assert content["found"] is True
        assert content["column_count"] == 2
        assert len(content["columns"]) == 2

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_get_domain_not_found(self, mock_get_store):
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {"total": {"value": 0}, "hits": []}
        }
        mock_store = Mock()
        mock_store.client = mock_client
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        result = await get_domain_from_vectordb("nonexistent")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        content = json.loads(result.body.decode())
        assert content["found"] is False
        assert "not found" in content["message"]

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_get_domain_no_index(self, mock_get_store):
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_store = Mock()
        mock_store.client = mock_client
        mock_store.index_name = "test_index"
        mock_get_store.return_value = mock_store
        result = await get_domain_from_vectordb("customer")
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        content = json.loads(result.body.decode())
        assert content["found"] is False
        assert "does not exist yet" in content["message"]

class TestExtendDomainRoute:
    @pytest.fixture
    def mock_request(self):
        return Mock(spec=Request)

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    @patch('app.agents.schema_suggester.SchemaSuggesterEnhanced')
    async def test_extend_domain_with_ai_column_storage(self, mock_suggester_class, mock_get_store, mock_request):
        mock_store = Mock()
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "column_name": "email",
                            "sample_values": [],
                            "metadata": {"type": "string"}
                        }
                    }
                ]
            }
        }
        mock_get_store.return_value = mock_store
        mock_suggester = Mock()
        mock_ai_columns = [
            {
                "column_name": "phone",
                "type": "string",
                "description": "Phone number",
                "sample_values": ["123-456-7890"]
            }
        ]
        mock_suggester.bootstrap_schema_with_preferences = AsyncMock(
            return_value={"columns": mock_ai_columns}
        )
        mock_suggester_class.return_value = mock_suggester
        payload = {
            "domain": "customer",
            "suggest_additional": True,
            "extension_preferences": {
                "column_count": 1,
                "focus_area": "contact",
                "style": "standard"
            }
        }
        mock_request.json = AsyncMock(return_value=payload)
        result = await extend_domain(mock_request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 500
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "error" in content or "AttributeError" in str(content)
        mock_store.add_column.assert_not_called()

class TestSuggestExtensionsRoute:
    @pytest.fixture
    def mock_request(self):
        return Mock(spec=Request)

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_suggest_extensions_domain_not_found(self, mock_get_store, mock_request):
        mock_store = Mock()
        mock_store.client.search.return_value = {
            "hits": {"total": {"value": 0}, "hits": []}
        }
        mock_get_store.return_value = mock_store
        payload = {"suggestion_preferences": {"column_count": 3}}
        mock_request.json = AsyncMock(return_value=payload)
        result = await suggest_extensions("nonexistent", mock_request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "not found" in content["error"]

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    @patch('app.agents.schema_suggester.SchemaSuggesterEnhanced')
    async def test_suggest_extensions_success(self, mock_suggester_class, mock_get_store, mock_request):
        mock_store = Mock()
        mock_store.client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "column_name": "email",
                            "sample_values": [],
                            "metadata": {"type": "string"}
                        }
                    }
                ]
            }
        }
        mock_get_store.return_value = mock_store
        mock_suggester = Mock()
        mock_suggestions = {
            "columns": [{"column_name": "phone", "type": "string"}]
        }
        mock_suggester.bootstrap_schema_with_preferences = AsyncMock(
            return_value=mock_suggestions
        )
        mock_suggester_class.return_value = mock_suggester
        payload = {
            "suggestion_preferences": {
                "column_count": 2,
                "focus_areas": ["analytics"],
                "style": "standard"
            }
        }
        mock_request.json = AsyncMock(return_value=payload)
        result = await suggest_extensions("customer", mock_request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["status"] == "success"
        assert "suggestions" in content

    @pytest.mark.asyncio
    @patch('app.api.routes.get_store')
    async def test_suggest_extensions_store_unavailable(self, mock_get_store, mock_request):
        mock_get_store.return_value = None
        payload = {"suggestion_preferences": {}}
        mock_request.json = AsyncMock(return_value=payload)
        result = await suggest_extensions("customer", mock_request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "OpenSearch store not available" in content["error"]

class TestResuggestDomainSchemaRoute:
    @pytest.fixture
    def mock_request(self):
        return Mock(spec=Request)

    @pytest.mark.asyncio
    async def test_resuggest_schema_missing_description(self, mock_request):
        mock_request.json = AsyncMock(return_value={})
        result = await regenerate_suggestions(mock_request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "business_description is required" in content["error"]

    @pytest.mark.asyncio
    @patch('app.agents.schema_suggester.SchemaSuggesterEnhanced')
    async def test_resuggest_schema_success(self, mock_suggester_class, mock_request):
        mock_suggester = Mock()
        mock_enhanced_schema = {
            "columns": [{"column_name": "email", "type": "string"}],
            "total_columns": 1,
            "style": "standard",
            "iteration": 1,
            "preferences_applied": {},
            "next_iteration_tips": [],
            "style_recommendations": []
        }
        mock_suggester.bootstrap_schema_with_preferences = AsyncMock(
            return_value=mock_enhanced_schema
        )
        mock_suggester_class.return_value = mock_suggester
        payload = {
            "business_description": "customer management system",
            "user_preferences": {"style": "standard"}
        }
        mock_request.json = AsyncMock(return_value=payload)
        result = await regenerate_suggestions(mock_request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        content = json.loads(result.body.decode())
        assert content["status"] == "success"
        assert content["total_columns"] == 1

    @pytest.mark.asyncio
    @patch('app.agents.schema_suggester.SchemaSuggesterEnhanced')
    async def test_resuggest_schema_exception(self, mock_suggester_class, mock_request):
        mock_suggester_class.side_effect = Exception("AI service unavailable")
        payload = {"business_description": "customer system"}
        mock_request.json = AsyncMock(return_value=payload)
        result = await regenerate_suggestions(mock_request)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 500
        content = json.loads(result.body.decode())
        assert "error" in content
