"""
Unit tests for rule suggestion endpoints (/api/aips/suggest-rules)
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.responses import JSONResponse
from app.api.rule_suggestion_routes import suggest_rules
from app.auth.authentication import UserInfo

class TestSuggestRulesRoute:
    """Tests for /api/aips/rule/suggest endpoint"""

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
    def mock_schema(self):
        return {
            "email": {"type": "string"},
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }

    @pytest.mark.asyncio
    @patch('app.vector_db.schema_loader.get_schema_by_domain')
    @patch('app.agents.agent_runner.run_agent')
    async def test_suggest_rules_success(self, mock_run_agent, mock_get_schema, mock_user_info, mock_schema):
        mock_get_schema.return_value = mock_schema
        mock_run_agent.return_value = ["rule1", "rule2"]
        result = await suggest_rules("customer", mock_user_info)
        assert isinstance(result, dict)
        assert result.get("rule_suggestions") == ["rule1", "rule2"]
        mock_get_schema.assert_called_once_with("customer")
        mock_run_agent.assert_called_once_with(mock_schema)

    @pytest.mark.asyncio
    @patch('app.vector_db.schema_loader.get_schema_by_domain')
    @patch('app.agents.schema_suggester.bootstrap_schema_for_domain')
    async def test_suggest_rules_domain_not_found(self, mock_bootstrap, mock_get_schema, mock_user_info):
        mock_get_schema.return_value = None
        mock_bootstrap.return_value = {"col1": {"type": "string"}, "col2": {"type": "integer"}}
        result = await suggest_rules("nonexistent", mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404  # HTTP 404 Not Found
        content = json.loads(result.body)
        assert "error" in content
        assert "Domain not found" in content["error"]
        assert "suggested_columns" in content

    @pytest.mark.asyncio
    @patch('app.vector_db.schema_loader.get_schema_by_domain')
    async def test_suggest_rules_connection_failed(self, mock_get_schema, mock_user_info):
        mock_get_schema.side_effect = Exception("AuthorizationException: Access denied")
        result = await suggest_rules("customer", mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 503  # HTTP 503 Service Unavailable
        content = json.loads(result.body)
        assert "error" in content
        assert "Vector database connection failed" in content["error"]

    @pytest.mark.asyncio
    @patch('app.vector_db.schema_loader.get_schema_by_domain')
    async def test_suggest_rules_unexpected_error(self, mock_get_schema, mock_user_info):
        mock_get_schema.side_effect = Exception("Unexpected error")
        result = await suggest_rules("customer", mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        content = json.loads(result.body.decode())
        assert "error" in content
        assert "Domain not found" in content["error"]

    @pytest.mark.asyncio  
    @patch('app.agents.schema_suggester.bootstrap_schema_for_domain')
    @patch('app.vector_db.schema_loader.get_schema_by_domain')
    async def test_suggest_rules_bootstrap_error(self, mock_get_schema, mock_bootstrap, mock_user_info):
        mock_get_schema.side_effect = Exception("DB error")
        mock_bootstrap.side_effect = Exception("Bootstrap failed")
        result = await suggest_rules("customer", mock_user_info)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 500  # HTTP 500 Internal Server Error
        content = json.loads(result.body)
        assert "error" in content
        assert "Internal server error" in content["error"]
