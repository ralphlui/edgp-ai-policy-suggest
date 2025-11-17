import json
import types
import pytest
from unittest.mock import patch

from app.tools import rule_tools
from app.tools.rule_tools import (
    _process_llm_request,
    generate_type_specific_fallback,
)


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyLLM:
    def __init__(self, payload: str):
        self._payload = payload

    def invoke(self, *args, **kwargs):
        return DummyResponse(self._payload)


@pytest.mark.parametrize(
    "payload,expected_prefix",
    [
        ("""```json\n[{\"column\": \"id\", \"expectations\": []}]```""", "```json"),
        (json.dumps({"column": "id", "expectations": []}), "{\"column"),
        (json.dumps({"rules": [{"column": "id", "expectations": []}]}), "{\"rules"),
        ("not json at all", "not json at all"),
    ],
)
def test_process_llm_request_various_payloads(payload, expected_prefix):
    # Mock the validation middleware to avoid the complex validation path
    with patch('app.tools.rule_tools.settings') as mock_settings, \
         patch('app.validation.middleware.AgentValidationContext') as mock_context_class:
        
        # Mock settings to avoid validation config issues
        mock_settings.get_llm_validation_config.return_value = {"enabled": False}
        
        # Mock validation context to return the input unchanged
        mock_context = mock_context_class.return_value.__enter__.return_value
        mock_context.validate_input.return_value = "prompt"
        mock_context.validate_output.return_value = payload
        
        llm = DummyLLM(payload)
        result = _process_llm_request(llm, "prompt")
        assert isinstance(result, str)
        assert result.startswith(expected_prefix)


def test_generate_type_specific_fallbacks_for_types():
    # number
    num = generate_type_specific_fallback("amount", "number")
    names = [e["expectation_type"] for e in num["expectations"]]
    assert "expect_column_values_to_be_of_type" in names
    assert any(e.get("kwargs", {}).get("type_") == "INTEGER" for e in num["expectations"])

    # string
    s = generate_type_specific_fallback("name", "string")
    assert any(e.get("kwargs", {}).get("type_list") == ["string"] for e in s["expectations"])
    assert any(e.get("expectation_type") == "expect_column_values_to_match_regex" for e in s["expectations"])

    # date
    d = generate_type_specific_fallback("created_date", "date")
    assert any(e.get("expectation_type") == "expect_column_values_to_be_dateutil_parseable" for e in d["expectations"])

    # boolean
    b = generate_type_specific_fallback("is_active", "boolean")
    assert any(e.get("expectation_type") == "expect_column_values_to_be_in_set" for e in b["expectations"])

    # array
    arr = generate_type_specific_fallback("tags", "array")
    assert any(e.get("kwargs", {}).get("type_list") == ["array"] for e in arr["expectations"])

    # object
    obj = generate_type_specific_fallback("meta", "object")
    assert any(e.get("expectation_type") == "expect_column_values_to_be_json_parseable" for e in obj["expectations"])


def test_format_gx_rules_parsing_paths():
    # Direct list JSON
    raw_list = json.dumps([{"column": "id", "expectations": []}])
    out = rule_tools.format_gx_rules.invoke(raw_list)
    assert isinstance(out, list) and out and out[0]["column"] == "id"

    # Single object JSON
    raw_obj = json.dumps({"column": "name", "expectations": []})
    out = rule_tools.format_gx_rules.invoke(raw_obj)
    assert isinstance(out, list) and out[0]["column"] == "name"

    # Regex unescaped case should be auto-fixed
    bad_regex = '{"column": "email", "expectations": [{"expectation_type": "x", "kwargs": {"regex": "\\d{3}-\\d{2}-\\d{4}"}}]}'
    out = rule_tools.format_gx_rules.invoke(bad_regex)
    assert isinstance(out, list) and out[0]["column"] == "email"

    # Fallback extraction path
    fallback_raw = 'random text {"column": "age", "expectations": [{"expectation_type": "y"}] } garbage'
    out = rule_tools.format_gx_rules.invoke(fallback_raw)
    assert isinstance(out, list)
    assert any(isinstance(item, dict) and item.get("column") == "age" for item in out)

    # No matches at all -> returns error object
    no_match = "completely unrelated text without any column objects"
    out = rule_tools.format_gx_rules.invoke(no_match)
    assert isinstance(out, list) and out and isinstance(out[0], dict)
    assert out[0].get("error")


def test_normalize_rule_suggestions_various_items():
    # Non-list under raw
    out = rule_tools.normalize_rule_suggestions.invoke({"rule_input": {"raw": {}}})
    assert out["error"] == "Invalid input type"

    # Mixed list
    raw = [
        {"column": "id", "expectations": []},
        {"not_column": True},
        [1, 2, 3],
        "string",
        {"column": "email", "expectations": [{"expectation_type": "x"}]},
    ]
    out = rule_tools.normalize_rule_suggestions.invoke({"rule_input": {"raw": raw}})
    assert "id" in out and "email" in out and len(out) == 2


def test_convert_to_rule_ms_format_with_various_rules():
    suggestions = {
        "suggestions": {
            "gender": {"expectations": []},  # triggers gender specific default
            "amount": {
                "expectations": [
                    {"expectation_type": "expect_column_values_to_be_in_type_list", "kwargs": {"type_list": ["number"]}},
                    {"expectation_type": "expect_column_values_to_be_in_range", "kwargs": {"min_value": 0, "max_value": 10}},
                    {"expectation_type": "expect_column_values_to_be_greater_than", "kwargs": {"min_value": 1}},
                    {"expectation_type": "expect_column_values_to_be_less_than", "kwargs": {"max_value": 99}},
                    {"expectation_type": "expect_column_values_to_be_in_set", "kwargs": {"value_set": [1,2]}},
                    {"expectation_type": "ignore_me"},
                ]
            },
            "created_date": {"expectations": []},  # should map to date parseable
            "name": {
                "expectations": [
                    {"expectation_type": "expect_column_values_to_match_regex", "kwargs": {"regex": "^.+$"}}
                ]
            },
            "unknown": {"expectations": []},  # default to string type list
        }
    }

    out = rule_tools.convert_to_rule_ms_format.invoke({"rule_input": suggestions})

    # Basic shape
    assert isinstance(out, list) and out

    # Contains converted GX rule names
    names = [r["rule_name"] for r in out]
    assert any(n.startswith("Expect") for n in names)

    # Gender default rule present
    assert any(r["column_name"] == "gender" and r["rule_name"] == "ExpectColumnValuesToBeInSet" for r in out)

    # Date parseable rule for created_date
    assert any(r["column_name"] == "created_date" and r["rule_name"] == "ExpectColumnValuesToBeDateutilParseable" for r in out)

    # Default string type rule for unknown
    assert any(r["column_name"] == "unknown" and r["rule_name"] == "ExpectColumnValuesToBeInTypeList" for r in out)


def test_fetch_gx_rules_default_when_url_missing(monkeypatch):
    # Ensure no RULE_URL in env and empty setting
    monkeypatch.setattr(rule_tools.settings, "rule_api_url", "")
    monkeypatch.delenv("RULE_URL", raising=False)

    rules = rule_tools.fetch_gx_rules.invoke("")
    assert isinstance(rules, list)
    assert any(r.get("rule_name") == "ExpectColumnValuesToBeUnique" for r in rules)


def test_fetch_gx_rules_http_success_list(monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"rule_name": "ExpectColumnValuesToBeInSet", "column_name": "status", "value": ["A", "B"]}
            ]

    monkeypatch.setattr(rule_tools.settings, "rule_api_url", "http://example/rules")
    monkeypatch.setattr(rule_tools, "requests", types.SimpleNamespace(get=lambda url, timeout=3: Resp()))

    rules = rule_tools.fetch_gx_rules.invoke("")
    assert isinstance(rules, list) and rules[0]["rule_name"].startswith("Expect")


def test_fetch_gx_rules_http_success_non_list(monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": "ok"}

    monkeypatch.setattr(rule_tools.settings, "rule_api_url", "http://example/rules")
    monkeypatch.setattr(rule_tools, "requests", types.SimpleNamespace(get=lambda url, timeout=3: Resp()))

    rules = rule_tools.fetch_gx_rules.invoke("")
    assert isinstance(rules, dict) and rules.get("message") == "ok"


def test_fetch_gx_rules_http_exception(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rule_tools.settings, "rule_api_url", "http://example/rules")
    monkeypatch.setattr(rule_tools, "requests", types.SimpleNamespace(get=_raise))

    rules = rule_tools.fetch_gx_rules.invoke("")
    assert isinstance(rules, list)
    assert any(r.get("rule_name") == "ExpectColumnValuesToBeUnique" for r in rules)


def test_normalize_rule_name_empty():
    # Test line 27 - empty rule_name
    result = rule_tools._normalize_rule_name("")
    assert result == ""


def test_normalize_rule_name_special_words():
    # Test lines 73-81 - special word capitalization
    result = rule_tools._normalize_rule_name("column_id_value")
    assert "Id" in result
    
    result = rule_tools._normalize_rule_name("url_validator")
    assert "Url" in result
    
    result = rule_tools._normalize_rule_name("api_json_uuid")
    assert "Api" in result and "Json" in result and "Uuid" in result


def test_generate_type_specific_fallback_unknown_type():
    # Test line 625 - default fallback for unknown types
    result = rule_tools.generate_type_specific_fallback("unknown_col", "weird_type")
    assert result["column"] == "unknown_col"
    assert any(e["expectation_type"] == "expect_column_values_to_be_unique" 
              for e in result["expectations"])


def test_normalize_rule_name_word_boundaries():
    # Test lines 45-64 - regex word boundary patterns
    result = rule_tools._normalize_rule_name("columnvaluestobeinset")
    assert "Column" in result and "Values" in result
    
    result = rule_tools._normalize_rule_name("valuestobeintype")
    assert "Values" in result and "Type" in result


def test_process_llm_request_validation_exception():
    # Test lines 671-673 - validation exception fallback
    from unittest.mock import Mock, MagicMock
    
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = '{"column": "test", "expectations": []}'
    mock_llm.invoke.return_value = mock_response
    
    with patch('app.tools.rule_tools.settings') as mock_settings, \
         patch('app.validation.middleware.AgentValidationContext') as mock_context_class:
        
        # Make validation context raise an exception
        mock_context_class.return_value.__enter__.side_effect = Exception("Validation failed")
        mock_settings.get_llm_validation_config.return_value = {"enabled": True}
        
        result = rule_tools._process_llm_request(mock_llm, "test prompt")
        # Should fallback to direct LLM call
        assert "column" in result or result == '{"column": "test", "expectations": []}'
