"""
Tests for app/tools/rule_tools.py module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os


class TestRuleToolsModule:
    """Test rule_tools.py module functionality"""
    
    def test_rule_tools_module_import(self):
        """Test rule_tools module can be imported"""
        from app.tools import rule_tools
        assert hasattr(rule_tools, '__name__')

    def test_tool_functions_exist(self):
        """Test all tool functions exist"""
        from app.tools.rule_tools import (
            fetch_gx_rules, suggest_column_rules, suggest_column_names_only,
            format_gx_rules, normalize_rule_suggestions, convert_to_rule_ms_format
        )
        
        assert callable(fetch_gx_rules)
        assert callable(suggest_column_rules)
        assert callable(suggest_column_names_only)
        assert callable(format_gx_rules)
        assert callable(normalize_rule_suggestions)
        assert callable(convert_to_rule_ms_format)

    @patch('app.tools.rule_tools.requests.get')
    def test_fetch_gx_rules_success(self, mock_get):
        """Test successful rule fetching"""
        mock_response = Mock()
        mock_response.json.return_value = [{"rule": "test"}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        from app.tools.rule_tools import fetch_gx_rules
        
        # Call the underlying function directly
        result = fetch_gx_rules.func("")
        
        assert isinstance(result, list)
        mock_get.assert_called_once()

    @patch('app.tools.rule_tools.requests.get')
    def test_fetch_gx_rules_connection_error(self, mock_get):
        """Test rule fetching with connection error"""
        from app.tools.rule_tools import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        from app.tools.rule_tools import fetch_gx_rules
        
        # Call the underlying function directly
        result = fetch_gx_rules.func("")
        
        assert isinstance(result, list)
        assert len(result) > 0  # Should return default rules

    @patch('app.tools.rule_tools.requests.get')
    def test_fetch_gx_rules_http_error(self, mock_get):
        """Test rule fetching with HTTP error"""
        mock_get.side_effect = Exception("HTTP Error")
        
        from app.tools.rule_tools import fetch_gx_rules
        
        result = fetch_gx_rules.func("")
        
        assert isinstance(result, list)
        assert len(result) > 0  # Should return default rules

    def test_format_gx_rules_valid_json(self):
        """Test formatting rules with valid JSON"""
        from app.tools.rule_tools import format_gx_rules
        
        valid_json = '[{"column": "test", "expectations": []}]'
        result = format_gx_rules.func(valid_json)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_format_gx_rules_invalid_json(self):
        """Test formatting rules with invalid JSON"""
        from app.tools.rule_tools import format_gx_rules
        
        invalid_json = 'not json at all'
        result = format_gx_rules.func(invalid_json)
        
        assert isinstance(result, list)

    def test_format_gx_rules_malformed_objects(self):
        """Test formatting rules with malformed JSON objects"""
        from app.tools.rule_tools import format_gx_rules
        
        malformed = '{"column": "test", "expectations":}'  # Invalid JSON
        result = format_gx_rules.func(malformed)
        
        assert isinstance(result, list)

    def test_normalize_rule_suggestions_valid_input(self):
        """Test rule normalization with valid input"""
        from app.tools.rule_tools import normalize_rule_suggestions
        
        input_data = {
            "raw": [
                {"column": "test_col", "expectations": [{"type": "test"}]}
            ]
        }
        result = normalize_rule_suggestions.func(input_data)
        
        assert isinstance(result, dict)
        assert "test_col" in result

    def test_normalize_rule_suggestions_invalid_input(self):
        """Test rule normalization with invalid input"""
        from app.tools.rule_tools import normalize_rule_suggestions
        
        input_data = {"raw": "not a list"}
        result = normalize_rule_suggestions.func(input_data)
        
        assert isinstance(result, dict)
        assert "error" in result

    def test_normalize_rule_suggestions_malformed_items(self):
        """Test rule normalization with malformed items"""
        from app.tools.rule_tools import normalize_rule_suggestions
        
        input_data = {
            "raw": [
                {"no_column_key": "test"},  # Missing column key
                {"column": "valid_col", "expectations": []}
            ]
        }
        result = normalize_rule_suggestions.func(input_data)
        
        assert isinstance(result, dict)
        assert "valid_col" in result

    def test_convert_to_rule_ms_format_valid(self):
        """Test conversion to rule microservice format"""
        from app.tools.rule_tools import convert_to_rule_ms_format
        
        input_data = {
            "suggestions": {
                "test_col": {
                    "expectations": [
                        {"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {}}
                    ]
                }
            }
        }
        result = convert_to_rule_ms_format.func(input_data)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["column_name"] == "test_col"

    def test_convert_to_rule_ms_format_empty(self):
        """Test conversion with empty suggestions"""
        from app.tools.rule_tools import convert_to_rule_ms_format
        
        input_data = {"suggestions": {}}
        result = convert_to_rule_ms_format.func(input_data)
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_convert_to_rule_ms_format_complex_rules(self):
        """Test conversion with complex rule expectations"""
        from app.tools.rule_tools import convert_to_rule_ms_format
        
        input_data = {
            "suggestions": {
                "email_col": {
                    "expectations": [
                        {
                            "expectation_type": "expect_column_values_to_match_regex",
                            "kwargs": {"regex": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"}
                        }
                    ]
                }
            }
        }
        result = convert_to_rule_ms_format.func(input_data)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "ExpectColumnValuesToMatchRegex" in result[0]["rule_name"]

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=False)
    @patch('app.aws.aws_secrets_service.require_openai_api_key', return_value='test-key')
    @patch('app.tools.rule_tools.ChatOpenAI')
    def test_suggest_column_rules_mock(self, mock_chat_openai, mock_require_api_key):
        """Test column rule suggestion with mocked OpenAI"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '[{"column": "test", "expectations": []}]'
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        from app.tools.rule_tools import suggest_column_rules
        
        schema = {"columns": [{"name": "test", "type": "string"}]}
        rules = [{"rule_name": "test_rule"}]
        
        result = suggest_column_rules.func(schema, rules)
        
        assert isinstance(result, str)
        mock_llm.invoke.assert_called_once()

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=False)
    @patch('app.aws.aws_secrets_service.require_openai_api_key', return_value='test-key')
    @patch('app.tools.rule_tools.ChatOpenAI')
    def test_suggest_column_names_only_mock(self, mock_chat_openai, mock_require_api_key):
        """Test column name suggestion with mocked OpenAI"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '["col1", "col2", "col3"]'
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        from app.tools.rule_tools import suggest_column_names_only
        
        result = suggest_column_names_only.func("customer")
        
        assert isinstance(result, str)
        mock_llm.invoke.assert_called_once()


class TestRuleToolsUtilities:
    """Test utility functions and edge cases"""
    
    def test_format_gx_rules_regex_extraction(self):
        """Test regex-based JSON object extraction"""
        from app.tools.rule_tools import format_gx_rules
        
        # Text with embedded JSON objects
        mixed_text = 'Some text {"column": "test1", "expectations": []} more text {"column": "test2", "expectations": []}'
        result = format_gx_rules.func(mixed_text)
        
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_rule_name_conversion_logic(self):
        """Test GX rule name conversion logic"""
        from app.tools.rule_tools import convert_to_rule_ms_format
        
        input_data = {
            "suggestions": {
                "test_col": {
                    "expectations": [
                        {"expectation_type": "expect_column_values_to_be_between", "kwargs": {"min": 0, "max": 100}}
                    ]
                }
            }
        }
        result = convert_to_rule_ms_format.func(input_data)
        
        assert len(result) > 0
        assert result[0]["rule_name"] == "ExpectColumnValuesToBeBetween"
        assert result[0]["value"]["min"] == 0
        assert result[0]["value"]["max"] == 100

    def test_default_gx_rules_structure(self):
        """Test default GX rules returned on connection error"""
        from app.tools.rule_tools import fetch_gx_rules
        
        # Mock connection error to get default rules
        with patch('app.tools.rule_tools.requests.get') as mock_get:
            from app.tools.rule_tools import requests
            mock_get.side_effect = requests.exceptions.ConnectionError()
            
            result = fetch_gx_rules.func("")
            
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Check structure of default rules
            for rule in result:
                assert "rule_name" in rule
                assert "description" in rule
                assert "applies_to" in rule