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
        
        # Test all different rule types
        input_data = {
            "suggestions": {
                "numeric_col": {
                    "expectations": [
                        {
                            "expectation_type": "expect_column_values_to_be_in_type_list",
                            "kwargs": {"type_list": ["number"]}
                        },
                        {
                            "expectation_type": "expect_column_values_to_be_in_range",
                            "kwargs": {"min_value": 0, "max_value": 100}
                        }
                    ],
                    "column_info": {"type": "number"}
                },
                "string_col": {
                    "expectations": [
                        {
                            "expectation_type": "expect_column_values_to_match_regex",
                            "kwargs": {"regex": "^[\\w\\.-]+$"}
                        }
                    ],
                    "column_info": {"type": "string"}
                },
                "date_col": {
                    "expectations": [
                        {
                            "expectation_type": "expect_column_values_to_be_dateutil_parseable"
                        }
                    ],
                    "column_info": {"type": "date"}
                },
                "bool_col": {
                    "expectations": [
                        {
                            "expectation_type": "expect_column_values_to_be_in_set",
                            "kwargs": {"value_set": [True, False]}
                        }
                    ],
                    "column_info": {"type": "boolean"}
                }
            }
        }
        result = convert_to_rule_ms_format.func(input_data)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verify numeric rules
        numeric_rules = [r for r in result if r["column_name"] == "numeric_col"]
        assert any(r["rule_name"] == "ExpectColumnValuesToBeInTypeList" and 
                  r["value"]["type_list"] == ["number"] for r in numeric_rules)
        assert any(r["rule_name"] == "ExpectColumnValuesToBeInRange" and 
                  r["value"]["min_value"] == 0 and r["value"]["max_value"] == 100 
                  for r in numeric_rules)
        
        # Verify string rules
        string_rules = [r for r in result if r["column_name"] == "string_col"]
        assert any(r["rule_name"] == "ExpectColumnValuesToMatchRegex" and 
                  r["value"]["regex"] == "^[\\w\\.-]+$" for r in string_rules)
        
        # Verify date rules
        date_rules = [r for r in result if r["column_name"] == "date_col"]
        assert any(r["rule_name"] == "ExpectColumnValuesToBeDateutilParseable" 
                  for r in date_rules)
        
        # Verify boolean rules
        bool_rules = [r for r in result if r["column_name"] == "bool_col"]
        assert any(r["rule_name"] == "ExpectColumnValuesToBeInSet" and 
                  r["value"]["value_set"] == [True, False] for r in bool_rules)
        
    def test_convert_to_rule_ms_format_with_inferred_types(self):
        """Test rule conversion with type inference"""
        from app.tools.rule_tools import convert_to_rule_ms_format
        
        input_data = {
            "suggestions": {
                "ORDER_DATE": {
                    "expectations": [
                        {
                            "expectation_type": "expect_column_values_to_not_be_null"
                        }
                    ],
                    "column_info": {"type": "unknown"}
                },
                "CUSTOMER_NUM": {
                    "expectations": [
                        {
                            "expectation_type": "expect_column_values_to_not_be_null"
                        }
                    ],
                    "column_info": {"type": "number"}
                },
                "ACTIVE_YN": {
                    "expectations": [
                        {
                            "expectation_type": "expect_column_values_to_be_in_set",
                            "kwargs": {"value_set": ["Y", "N"]}
                        }
                    ],
                    "column_info": {"type": "string"}
                }
            }
        }
        result = convert_to_rule_ms_format.func(input_data)
        
        # Verify basic rule conversion
        assert isinstance(result, list)
        assert len(result) >= 3  # At least one rule per column
        
        # Verify not null rules are converted
        assert any(r["rule_name"] == "ExpectColumnValuesToNotBeNull" 
                  for r in result)
        
        # Verify set-based rules
        yn_rules = [r for r in result if r["column_name"] == "ACTIVE_YN"]
        assert any(r["rule_name"] == "ExpectColumnValuesToBeInSet" and 
                  r["value"]["value_set"] == ["Y", "N"] for r in yn_rules)

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
        
        # Test with different column types
        schema = {
            "domain": "test_domain",
            "id_column": {
                "type": "number",
                "name": "id_column",
                "description": "ID column",
                "format": "",
                "constraints": {}
            },
            "date_column": {
                "type": "date",
                "name": "date_column",
                "description": "Date column",
                "format": "",
                "constraints": {}
            },
            "name_column": {
                "type": "string",
                "name": "name_column",
                "description": "Name column",
                "format": "",
                "constraints": {}
            },
            "active_flag": {
                "type": "boolean",
                "name": "active_flag",
                "description": "Active flag",
                "format": "",
                "constraints": {}
            }
        }
        
        rules = [
            {"rule_name": "test_rule", "applies_to": ["all"]},
            {"rule_name": "number_rule", "applies_to": ["number"]},
            {"rule_name": "date_rule", "applies_to": ["date"]},
            {"rule_name": "string_rule", "applies_to": ["string"]}
        ]
        
        result = suggest_column_rules.func(schema, rules)
        assert isinstance(result, str)
        assert mock_llm.invoke.call_count > 0
        
        # Test with empty response from LLM
        mock_response.content = "[]"
        result = suggest_column_rules.func(schema, rules)
        assert isinstance(result, str)
        assert "[]" != result  # Should get fallback rules
        
        # Test with invalid JSON response from LLM
        mock_response.content = "invalid json"
        result = suggest_column_rules.func(schema, rules)
        assert isinstance(result, str)
        assert "[]" != result  # Should get fallback rules
        
        # Test with LLM error
        mock_llm.invoke.side_effect = Exception("LLM Error")
        result = suggest_column_rules.func(schema, rules)
        assert isinstance(result, str)
        assert "[]" != result  # Should get fallback rules

    def test_generate_type_specific_fallback(self):
        """Test generation of type-specific fallback rules"""
        from app.tools.rule_tools import generate_type_specific_fallback
        
        # Test numeric type
        result = generate_type_specific_fallback("test_num", "number")
        assert isinstance(result, dict)
        assert result["column"] == "test_num"
        assert len(result["expectations"]) >= 2  # Should have not_null and type rules at minimum
        
        # Test string type
        result = generate_type_specific_fallback("test_str", "string")
        assert isinstance(result, dict)
        assert result["column"] == "test_str"
        assert len(result["expectations"]) >= 2
        
        # Test date type
        result = generate_type_specific_fallback("test_date", "date")
        assert isinstance(result, dict)
        assert result["column"] == "test_date"
        assert len(result["expectations"]) >= 2
        
        # Test boolean type
        result = generate_type_specific_fallback("test_bool", "boolean")
        assert isinstance(result, dict)
        assert result["column"] == "test_bool"
        assert len(result["expectations"]) >= 2
        
        # Test array type
        result = generate_type_specific_fallback("test_array", "array")
        assert isinstance(result, dict)
        assert result["column"] == "test_array"
        assert len(result["expectations"]) >= 1
        
        # Test json type
        result = generate_type_specific_fallback("test_json", "json")
        assert isinstance(result, dict)
        assert result["column"] == "test_json"
        assert len(result["expectations"]) >= 1

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=False)
    @patch('app.aws.aws_secrets_service.require_openai_api_key', return_value='test-key')
    @patch('app.tools.rule_tools.ChatOpenAI')
    def test_column_type_inference(self, mock_chat_openai, mock_require_api_key):
        """Test column type inference from names"""
        from app.tools.rule_tools import suggest_column_rules
        
        schema = {
            "domain": "test_domain",
            "ORDER_DATE": {
                "type": "unknown",
                "name": "ORDER_DATE",
                "description": "",
                "format": "",
                "constraints": {}
            },
            "CUSTOMER_NUMBER": {
                "type": "unknown",
                "name": "CUSTOMER_NUMBER",
                "description": "",
                "format": "",
                "constraints": {}
            },
            "STATUS_FLAG": {
                "type": "unknown",
                "name": "STATUS_FLAG",
                "description": "",
                "format": "",
                "constraints": {}
            }
        }
        
        # Create test rules for each type
        rules = [
            {"rule_name": "expect_column_values_to_not_be_null", "applies_to": ["all"]},
            {"rule_name": "expect_column_values_to_be_dateutil_parseable", "applies_to": ["date"]},
            {"rule_name": "expect_column_values_to_be_in_type_list", "applies_to": ["number", "string", "boolean"]},
            {"rule_name": "expect_column_values_to_be_in_range", "applies_to": ["number"]}
        ]
        
        # Mock LLM to return empty response to force type inference and fallback
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "[]"
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm
        
        result = suggest_column_rules.func(schema, rules)
        assert isinstance(result, str)
        result_dict = json.loads(result)
        
        # Find and verify each column's rules
        date_rules = None
        number_rules = None
        flag_rules = None
        
        for rule_set in result_dict:
            print(f"Checking rules for column {rule_set['column']}: {rule_set['expectations']}")  # Debug print
            if rule_set["column"] == "ORDER_DATE":
                date_rules = rule_set
            elif rule_set["column"] == "CUSTOMER_NUMBER":
                number_rules = rule_set
            elif rule_set["column"] == "STATUS_FLAG":
                flag_rules = rule_set
        
        # Verify ORDER_DATE rules - Comprehensive date validation check
        assert date_rules is not None, "ORDER_DATE rules not found"
        date_expectations = date_rules["expectations"]
        
        # Should have at least 2 expectations: null check and date validation
        assert len(date_expectations) >= 2, f"Expected 2+ date rules, got: {date_expectations}"
        
        # Check for not null rule
        assert any(
            exp["expectation_type"] == "expect_column_values_to_not_be_null"
            for exp in date_expectations
        ), "Missing not null validation for ORDER_DATE"
        
        # Check for dateutil rule
        assert any(
            "dateutil" in exp["expectation_type"].lower()
            for exp in date_expectations
        ), f"Missing dateutil validation for ORDER_DATE. Rules: {date_expectations}"
        
        # Check for type list rule with datetime
        has_datetime_type = False
        for exp in date_expectations:
            if "expect_column_values_to_be_in_type_list" == exp["expectation_type"]:
                type_list = exp.get("kwargs", {}).get("type_list", [])
                if "datetime" in type_list or (
                    "string" in type_list and 
                    any("dateutil" in e["expectation_type"].lower() for e in date_expectations)
                ):
                    has_datetime_type = True
                    break
        
        assert has_datetime_type, f"Missing datetime type validation for ORDER_DATE. Rules: {date_expectations}"
            
        # Verify CUSTOMER_NUMBER rules
        assert number_rules is not None, "CUSTOMER_NUMBER rules not found"
        num_expectations = number_rules["expectations"]
        
        # Should have number type validation
        assert any(
            exp["expectation_type"] == "expect_column_values_to_be_in_type_list" and
            "number" in exp.get("kwargs", {}).get("type_list", [])
            for exp in num_expectations
        ), f"Missing number type validation for CUSTOMER_NUMBER. Rules: {num_expectations}"
            
        # Verify STATUS_FLAG rules
        assert flag_rules is not None, "STATUS_FLAG rules not found"
        flag_expectations = flag_rules["expectations"]
        
        # Should have either boolean type or value set rule
        has_flag_validation = any(
            (exp["expectation_type"] == "expect_column_values_to_be_in_type_list" and
             any(t in exp.get("kwargs", {}).get("type_list", []) for t in ["boolean", "string"])) or
            ("expect_column_values_to_be_in_set" in exp["expectation_type"])
            for exp in flag_expectations
        )
        assert has_flag_validation, f"Missing boolean/value set validation for STATUS_FLAG. Rules: {flag_expectations}"

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
        
        # Test with embedded JSON objects
        mixed_text = 'Some text {"column": "test1", "expectations": []} more text {"column": "test2", "expectations": []}'
        result = format_gx_rules.func(mixed_text)
        assert isinstance(result, list)
        assert len(result) >= 2
        
        # Test with regex patterns that need escaping
        text_with_regex = '''
        {
            "column": "email",
            "expectations": [{
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {"regex": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"}
            }]
        }
        '''
        result = format_gx_rules.func(text_with_regex)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "email" == result[0]["column"]
        assert "regex" in result[0]["expectations"][0]["kwargs"]
        
        # Test with completely invalid JSON
        invalid_text = "This is not JSON at all {{"
        result = format_gx_rules.func(invalid_text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        
        # Test with partial JSON
        partial_json = '{"column": "test", "expectations": [{bad json'
        result = format_gx_rules.func(partial_json)
        assert isinstance(result, list)
        
        # Test with nested JSON objects
        nested_json = '''
        {
            "column": "test",
            "expectations": [
                {
                    "expectation_type": "test",
                    "kwargs": {
                        "nested": {"key": "value"},
                        "array": [1, 2, 3]
                    }
                }
            ]
        }
        '''
        result = format_gx_rules.func(nested_json)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "test" == result[0]["column"]
        assert isinstance(result[0]["expectations"][0]["kwargs"]["nested"], dict)
        assert isinstance(result[0]["expectations"][0]["kwargs"]["array"], list)

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