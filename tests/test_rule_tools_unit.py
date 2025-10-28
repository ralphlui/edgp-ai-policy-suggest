import json
import types
import pytest

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
        ("""```json\n[{\"column\": \"id\", \"expectations\": []}]```""", "[\n"),
        (json.dumps({"column": "id", "expectations": []}), "[\n"),
        (json.dumps({"rules": [{"column": "id", "expectations": []}]}), "[\n"),
        ("not json at all", "[]"),
    ],
)
def test_process_llm_request_various_payloads(payload, expected_prefix):
    llm = DummyLLM(payload)
    result = _process_llm_request(llm, "prompt")
    assert isinstance(result, str)
    assert result.startswith(expected_prefix)


def test_generate_type_specific_fallbacks_for_types():
    # number
    num = generate_type_specific_fallback("amount", "number")
    names = [e["expectation_type"] for e in num["expectations"]]
    assert "expect_column_values_to_be_in_range" in names
    assert any(e.get("kwargs", {}).get("type_list") == ["number"] for e in num["expectations"])

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
