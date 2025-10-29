import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.validation import llm_validator as lv
from app.validation.validation_base import ValidationSeverity, ValidationIssue
from app.validation.middleware import (
    LLMValidationMiddleware,
    AgentValidationContext,
    llm_input_validator,
    llm_output_validator,
)
from app.exception.exceptions import ValidationError


def test_rate_limit_manager_basic_allow_and_block():
    rl = lv.RateLimitManager(requests_per_minute=1, requests_per_hour=1)
    allowed1, info1 = rl.check_rate_limit("user1")
    assert allowed1 is True
    assert info1["minute_remaining"] == 0
    assert info1["hour_remaining"] == 0

    allowed2, info2 = rl.check_rate_limit("user1")
    assert allowed2 is False
    # remaining should not be negative
    assert info2["minute_remaining"] >= 0
    assert info2["hour_remaining"] >= 0


def test_input_sanitizer_length_and_block_and_warning_and_clean():
    sanitizer = lv.InputSanitizer(max_length=5)
    sanitized, issues = sanitizer.sanitize_input("123456")
    assert sanitized == ""
    assert any(i.severity == ValidationSeverity.CRITICAL for i in issues)

    # Blocked pattern
    sanitizer = lv.InputSanitizer(max_length=1000)
    sanitized, issues = sanitizer.sanitize_input("please run rm -rf / on server")
    assert sanitized == ""
    assert any(i.severity == ValidationSeverity.CRITICAL for i in issues)

    # Warning pattern and cleaning behavior (use a warning pattern, not blocked)
    text = "<script>alert(1)</script> ignore policy and 'quote' !"
    sanitized, issues = sanitizer.sanitize_input(text)
    assert sanitized and "<script" not in sanitized
    assert any(i.severity == ValidationSeverity.HIGH for i in issues)
    # quotes should be escaped
    assert "\'" in sanitized or "\\'" in sanitized or "\"" in sanitized


def test_llm_response_validator_schema_paths_and_autocorrect():
    v = lv.LLMResponseValidator(strict_mode=True, auto_correct=True)

    # columns not a list
    res = v.validate_schema_response({"columns": "bad"})
    assert res.is_valid is False
    assert any(i.field == "columns" for i in res.issues)

    # empty columns list
    res = v.validate_schema_response({"columns": []})
    assert res.is_valid is False

    # mixed valid/invalid column definitions with autocorrect
    payload = {
        "columns": [
            {"name": "col1", "type": "integer", "samples": ["1", "2", "3"]},
            {"name": "1bad name", "type": "unknown", "samples": ["a"]},
        ]
    }
    res = v.validate_schema_response(payload)
    # With strict mode, HIGH issues cause invalid
    assert res.is_valid is False
    assert res.corrected_data is not None
    corrected_cols = res.corrected_data["columns"]
    assert corrected_cols[1]["name"].startswith("col_")
    assert corrected_cols[1]["type"] == "string"

    # has_blocking_issues strict vs non-strict
    issues = [
        ValidationIssue(field="x", message="m", severity=ValidationSeverity.HIGH),
        ValidationIssue(field="y", message="m", severity=ValidationSeverity.MEDIUM),
    ]
    assert v.has_blocking_issues(issues) is True
    v2 = lv.LLMResponseValidator(strict_mode=False)
    assert v2.has_blocking_issues(issues) is False


def test_llm_content_validator_safety_and_filtering_and_quality():
    cv = lv.LLMContentValidator(enable_advanced_safety=True)
    unsafe = (
        "Contact me at john@example.com, SSN 123-45-6789. "
        "password: supersecret12345 <script>alert('x')</script>"
    )
    result = cv.validate_content_safety(unsafe)
    # Should flag critical (credentials) and high (ssn) at least
    severities = {i.severity for i in result.issues}
    assert ValidationSeverity.CRITICAL in severities
    assert any("pii_ssn" in i.message or "ssn" in i.message.lower() for i in result.issues)
    assert result.is_valid is False

    filtered, removed = cv.filter_unsafe_content(unsafe)
    assert "[EMAIL_FILTERED]" in filtered
    assert "[SCRIPT_REMOVED]" in filtered
    assert any(x in removed for x in ["credentials", "email", "script_tags"])  # at least some

    # Quality: too short and punctuation missing
    q = cv.validate_content_quality("hi")
    assert any(i.field == "content" for i in q.issues)


def test_comprehensive_validator_request_and_response_and_stats():
    cfg = {
        "rate_limit_per_minute": 1,
        "rate_limit_per_hour": 1,
        "enable_advanced_safety": True,
        "strict_mode": True,
        "auto_correct": False,
    }
    comp = lv.ComprehensiveLLMValidator(cfg)

    ok = comp.validate_llm_request("Hello world with punctuation.", user_id="u1")
    assert ok.is_valid is True

    # Immediately exceed rate limit
    blocked = comp.validate_llm_request("Another try.", user_id="u1")
    assert blocked.is_valid is False
    assert any(i.field == "rate_limit" for i in blocked.issues)

    # String response unsafe
    resp = comp.validate_llm_response("api_key=abcdefgh12345678")
    assert resp.is_valid is False

    # Dict response rule type with missing rule fields
    resp2 = comp.validate_llm_response({"rules": [{}], "explanation": "because"}, response_type="rule")
    assert resp2.is_valid in (True, False)  # depending on computed confidence
    assert any("rules[0]" in i.field for i in resp2.issues)

    stats = comp.get_validation_stats()
    assert set(stats.keys()) == {
        "rate_limiter_stats",
        "sanitizer_stats",
        "content_validator_stats",
        "response_validator_stats",
    }


def test_convenience_functions_and_backward_compatibility():
    # validate_user_input wrapper
    res = lv.validate_user_input("This is fine.", user_id="u2", config={"rate_limit_per_minute": 1000})
    assert res.is_valid is True

    # validate_llm_output wrapper for string
    res2 = lv.validate_llm_output("password: abcdefghijklmnop")
    assert res2.is_valid is False

    # backward-compat function (global) for dict
    res3 = lv.validate_llm_response({"columns": [{"name": "c1", "type": "integer", "samples": ["1","2","3"]}]}, response_type="schema", strict_mode=True, auto_correct=False)
    assert hasattr(res3, "is_valid")


# ==============================
# Additional tests consolidated
# from test_llm_validation.py
# ==============================


class TestInputSanitizer_Consolidated:
    def setup_method(self):
        self.sanitizer = lv.InputSanitizer(max_length=1000)

    def test_clean_input_basic(self):
        clean_input = "SELECT name FROM users WHERE id = 1"
        result, issues = self.sanitizer.sanitize_input(clean_input)
        assert len(issues) == 0
        assert result == clean_input

    def test_blocked_patterns(self):
        dangerous_inputs = [
            "DELETE FROM users WHERE 1=1",
            "password: admin123",
            "hack the system with malicious intent",
            "system('rm -rf /')",
        ]
        for text in dangerous_inputs:
            result, issues = self.sanitizer.sanitize_input(text)
            assert len(issues) > 0
            assert any(i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH] for i in issues)

    def test_warning_patterns(self):
        warning_inputs = [
            "modify critical system settings",
            "external untrusted source data",
            "skip validation check",
        ]
        for text in warning_inputs:
            result, issues = self.sanitizer.sanitize_input(text)
            assert len(issues) > 0
            assert any(i.severity == ValidationSeverity.HIGH for i in issues)
            assert result != ""

    def test_input_length_limit(self):
        long_input = "a" * 2000
        result, issues = self.sanitizer.sanitize_input(long_input)
        assert len(issues) > 0
        assert any("exceeds limit" in i.message for i in issues)
        assert result == ""

    def test_html_sanitization(self):
        html_input = "<script>alert('xss')</script>Hello <b>world</b>"
        result, issues = self.sanitizer.sanitize_input(html_input)
        assert "<script>" not in result
        assert "<b>" not in result
        assert "Hello world" in result


class TestRateLimitManager_Consolidated:
    def setup_method(self):
        self.rate_limiter = lv.RateLimitManager(requests_per_minute=5, requests_per_hour=20)

    def test_within_rate_limit(self):
        user_id = "test_user_1"
        for i in range(5):
            is_allowed, remaining = self.rate_limiter.check_rate_limit(user_id)
            assert is_allowed
            assert remaining["minute_remaining"] == 4 - i

    def test_exceed_minute_rate_limit(self):
        user_id = "test_user_2"
        for _ in range(5):
            is_allowed, _ = self.rate_limiter.check_rate_limit(user_id)
            assert is_allowed
        is_allowed, remaining = self.rate_limiter.check_rate_limit(user_id)
        assert not is_allowed
        assert remaining["minute_remaining"] == 0

    def test_different_users_separate_limits(self):
        user1, user2 = "test_user_3", "test_user_4"
        for _ in range(5):
            is_allowed, _ = self.rate_limiter.check_rate_limit(user1)
            assert is_allowed
        is_allowed, _ = self.rate_limiter.check_rate_limit(user1)
        assert not is_allowed
        is_allowed, _ = self.rate_limiter.check_rate_limit(user2)
        assert is_allowed


class TestLLMContentValidator_Consolidated:
    def setup_method(self):
        self.validator = lv.LLMContentValidator(enable_advanced_safety=True)

    def test_safe_content(self):
        safe_content = "This is a normal business rule about customer data validation."
        result = self.validator.validate_content_safety(safe_content)
        assert result.is_valid
        assert result.confidence_score > 0.8
        assert len(result.issues) == 0

    def test_pii_detection(self):
        pii_content = "Contact user at john.doe@example.com or call 555-123-4567"
        result = self.validator.validate_content_safety(pii_content)
        assert len(result.issues) >= 2
        assert any("pii email" in i.message for i in result.issues)
        assert any("pii phone" in i.message for i in result.issues)

    def test_credential_detection(self):
        credential_content = "Use password = secretkey123 to access the system"
        result = self.validator.validate_content_safety(credential_content)
        assert not result.is_valid
        assert any("credentials" in i.message for i in result.issues)
        assert any(i.severity == ValidationSeverity.CRITICAL for i in result.issues)

    def test_sql_injection_detection(self):
        sql_content = "SELECT * FROM users UNION SELECT password FROM admin"
        result = self.validator.validate_content_safety(sql_content)
        assert not result.is_valid
        assert any("sql injection" in i.message for i in result.issues)

    def test_content_filtering(self):
        unsafe_content = "password = admin123 and email user@domain.com"
        filtered_content, removed_patterns = self.validator.filter_unsafe_content(unsafe_content)
        assert "credentials" in removed_patterns
        assert "email" in removed_patterns
        assert "[FILTERED]" in filtered_content or "[EMAIL_FILTERED]" in filtered_content

    def test_empty_content(self):
        result = self.validator.validate_content_safety("")
        assert not result.is_valid
        assert any(i.severity == ValidationSeverity.CRITICAL for i in result.issues)
        assert any("empty" in i.message.lower() for i in result.issues)


class TestComprehensiveLLMValidator_Consolidated:
    def setup_method(self):
        config = {
            "max_input_length": 1000,
            "rate_limit_per_minute": 10,
            "rate_limit_per_hour": 100,
            "strict_mode": True,
            "auto_correct": False,
            "enable_advanced_safety": True,
        }
        self.validator = lv.ComprehensiveLLMValidator(config)

    def test_valid_request(self):
        user_input = "Create a schema for customer data with name and age fields"
        user_id = "test_user_valid"
        result = self.validator.validate_llm_request(user_input, user_id)
        assert result.is_valid
        assert result.confidence_score > 0.7
        assert result.corrected_data is not None
        assert "sanitized_input" in result.corrected_data

    def test_dangerous_request(self):
        dangerous_input = "DELETE FROM users WHERE password = admin123"
        user_id = "test_user_dangerous"
        result = self.validator.validate_llm_request(dangerous_input, user_id)
        assert not result.is_valid
        assert result.confidence_score < 0.5
        assert any(i.severity == ValidationSeverity.CRITICAL for i in result.issues)

    def test_rate_limit_enforcement(self):
        user_id = "test_user_rate_limit"
        safe_input = "Generate a simple schema"
        for _ in range(10):
            result = self.validator.validate_llm_request(safe_input, user_id)
            assert result.is_valid
        result = self.validator.validate_llm_request(safe_input, user_id)
        assert not result.is_valid
        assert any("rate limit" in i.message.lower() for i in result.issues)

    def test_schema_response_validation(self):
        valid_schema_response = {
            "columns": [
                {"name": "customer_id", "type": "integer", "samples": ["1", "2", "3"]},
                {"name": "name", "type": "string", "samples": ["John", "Jane", "Bob"]},
            ]
        }
        result = self.validator.validate_llm_response(valid_schema_response, "schema")
        assert result.is_valid
        assert result.confidence_score > 0.8

    def test_invalid_schema_response(self):
        invalid_schema_response = {
            "columns": [
                {"name": "", "type": "invalid_type", "samples": []},
                {"missing_name": "value"},
            ]
        }
        result = self.validator.validate_llm_response(invalid_schema_response, "schema")
        assert not result.is_valid
        assert len(result.issues) > 0
        assert result.confidence_score < 0.7


class TestLLMValidationMiddleware_Consolidated:
    def setup_method(self):
        config = {"max_input_length": 1000, "rate_limit_per_minute": 10, "strict_mode": True}
        self.middleware = LLMValidationMiddleware(config)

    def test_middleware_input_validation(self):
        safe_input = "Create a customer schema"
        user_id = "test_middleware_user"
        result = self.middleware.validate_input(safe_input, user_id)
        assert result["is_valid"]
        assert "sanitized_input" in result
        assert "confidence_score" in result
        assert "validation_metadata" in result

    def test_middleware_blocks_dangerous_input(self):
        dangerous_input = "password = secret123"
        user_id = "test_middleware_dangerous"
        with pytest.raises(ValidationError):
            self.middleware.validate_input(dangerous_input, user_id)

    def test_middleware_output_validation(self):
        response = {"columns": [{"name": "test", "type": "string", "samples": ["a", "b", "c"]}]}
        result = self.middleware.validate_output(response, "schema")
        assert result["is_valid"]
        assert "filtered_response" in result
        assert "confidence_score" in result

    def test_middleware_metrics(self):
        safe_input = "test input"
        user_id = "test_metrics_user"
        try:
            self.middleware.validate_input(safe_input, user_id)
        except Exception:
            pass
        metrics = self.middleware.get_metrics()
        assert "total_requests" in metrics
        assert "validator_stats" in metrics
        assert metrics["total_requests"] > 0


class TestValidationDecorators_Consolidated:
    def test_input_validator_decorator(self):
        @llm_input_validator({"rate_limit_per_minute": 100})
        def test_function(user_input: str, user_id: str, **kwargs):
            return f"Processed: {user_input}"
        result = test_function("safe input", "test_user")
        assert "Processed:" in result
        with pytest.raises(ValidationError):
            test_function("password = secret123", "test_user")

    def test_output_validator_decorator(self):
        @llm_output_validator("schema")
        def test_function():
            return {"columns": [{"name": "test", "type": "string", "samples": ["a", "b", "c"]}]}
        result = test_function()
        assert "_validation" in result
        assert result["_validation"]["is_valid"]


class TestAgentValidationContext_Consolidated:
    def test_context_manager_basic(self):
        user_id = "test_context_user"
        with AgentValidationContext(user_id) as validator:
            sanitized = validator.validate_input("Create a schema for users")
            assert isinstance(sanitized, str)
            assert len(sanitized) > 0
            validated_output = validator.validate_output(
                {"columns": [{"name": "id", "type": "integer", "samples": ["1", "2", "3"]}]},
                "schema",
            )
            assert isinstance(validated_output, dict)

    def test_context_manager_blocks_dangerous_input(self):
        user_id = "test_context_dangerous"
        with AgentValidationContext(user_id) as validator:
            with pytest.raises(ValidationError):
                validator.validate_input("DELETE FROM users")

    def test_context_manager_metrics(self):
        user_id = "test_context_metrics"
        with AgentValidationContext(user_id) as validator:
            try:
                validator.validate_input("safe input")
            except Exception:
                pass
            try:
                validator.validate_output("safe output", "content")
            except Exception:
                pass
            metrics = validator.get_metrics()
            assert metrics["user_id"] == user_id
            assert metrics["validations_performed"] >= 1


class TestValidationIntegration_Consolidated:
    def test_end_to_end_validation_flow(self):
        config = {
            "max_input_length": 1000,
            "rate_limit_per_minute": 100,
            "strict_mode": True,
            "enable_advanced_safety": True,
        }
        validator = lv.ComprehensiveLLMValidator(config)
        user_id = "integration_test_user"
        user_input = "Create a schema for customer data with fields: name, email, age"
        input_result = validator.validate_llm_request(user_input, user_id)
        assert input_result.is_valid
        sanitized_input = input_result.corrected_data["sanitized_input"]
        llm_response = {
            "columns": [
                {"name": "name", "type": "string", "samples": ["John Doe", "Jane Smith", "Bob Wilson"]},
                {"name": "email", "type": "string", "samples": ["john@email.com", "jane@email.com", "bob@email.com"]},
                {"name": "age", "type": "integer", "samples": ["25", "30", "35"]},
            ]
        }
        output_result = validator.validate_llm_response(llm_response, "schema")
        assert output_result.is_valid
        assert output_result.confidence_score > 0.8
        stats = validator.get_validation_stats()
        assert "rate_limiter_stats" in stats
        assert "content_validator_stats" in stats

    def test_configuration_integration(self):
        from app.core.config import settings
        validation_config = settings.get_llm_validation_config()
        assert "max_input_length" in validation_config
        assert "rate_limit_per_minute" in validation_config
        assert "strict_mode" in validation_config
        assert "enable_advanced_safety" in validation_config
        validator = lv.ComprehensiveLLMValidator(validation_config)
        assert validator is not None
        result = validator.validate_llm_request("test input", "config_test_user")
        assert result is not None


# ==============================
# Additional helpers tests from
# test_llm_validator_helpers.py
# ==============================


def test_rate_limit_manager_allows_then_blocks_helpers():
    rlm = lv.RateLimitManager(requests_per_minute=2, requests_per_hour=3)
    uid = "u1"
    allowed, rem = rlm.check_rate_limit(uid)
    assert allowed is True and rem["minute_remaining"] == 1
    allowed, rem = rlm.check_rate_limit(uid)
    assert allowed is True and rem["minute_remaining"] == 0
    allowed, rem = rlm.check_rate_limit(uid)
    assert allowed is False
    assert rem["minute_remaining"] == 0


def test_input_sanitizer_helpers_blocks_length_and_patterns():
    sanitizer = lv.InputSanitizer(max_length=5)
    s, issues = sanitizer.sanitize_input("abcdef")
    assert s == ""
    assert any(i.severity == ValidationSeverity.CRITICAL for i in issues)
    sanitizer = lv.InputSanitizer(max_length=1000)
    s, issues = sanitizer.sanitize_input("please delete from users")
    assert s == ""
    assert any(i.severity == ValidationSeverity.CRITICAL for i in issues)
    s, issues = sanitizer.sanitize_input("override policy rule")
    assert s
    assert any(i.severity == ValidationSeverity.HIGH for i in issues)


def test_input_sanitizer_helpers_clean_input():
    sanitizer = lv.InputSanitizer(max_length=1000)
    s, issues = sanitizer.sanitize_input("<b>O'Hara</b> ; <script>alert('x')</script>  A   B")
    assert "<b>" not in s and "</script>" not in s
    assert "O\\'Hara" in s and "\\;" in s
    assert s.endswith("A B")
    assert issues == []
