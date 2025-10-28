import pytest
from app.validation.llm_validator import RateLimitManager, InputSanitizer
from app.validation.validation_base import ValidationSeverity


def test_rate_limit_manager_allows_then_blocks():
    rlm = RateLimitManager(requests_per_minute=2, requests_per_hour=3)
    uid = "u1"

    allowed, rem = rlm.check_rate_limit(uid)
    assert allowed is True and rem["minute_remaining"] == 1

    allowed, rem = rlm.check_rate_limit(uid)
    assert allowed is True and rem["minute_remaining"] == 0

    # Third call within the same minute should be blocked by per-minute limit
    allowed, rem = rlm.check_rate_limit(uid)
    assert allowed is False
    assert rem["minute_remaining"] == 0


def test_input_sanitizer_blocks_length_and_patterns():
    sanitizer = InputSanitizer(max_length=5)
    # Exceeds length
    s, issues = sanitizer.sanitize_input("abcdef")
    assert s == ""
    assert any(i.severity == ValidationSeverity.CRITICAL for i in issues)

    # Blocked pattern
    sanitizer = InputSanitizer(max_length=1000)
    s, issues = sanitizer.sanitize_input("please delete from users")
    assert s == ""
    assert any(i.severity == ValidationSeverity.CRITICAL for i in issues)

    # Warning pattern
    s, issues = sanitizer.sanitize_input("override policy rule")
    assert s
    assert any(i.severity.value in ("high", "HIGH") for i in issues)


def test_input_sanitizer_clean_input():
    sanitizer = InputSanitizer(max_length=1000)
    s, issues = sanitizer.sanitize_input("<b>O'Hara</b> ; <script>alert('x')</script>  A   B")
    # Tags removed and quotes/semicolon escaped, script removed, whitespace normalized
    assert "<b>" not in s and "</script>" not in s
    # Quotes and semicolon should be escaped
    assert "O\\'Hara" in s and "\\;" in s
    assert s.endswith("A B")
    assert issues == []
