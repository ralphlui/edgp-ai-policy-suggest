import os
from typing import Optional, Dict, List
import pytest
from app.validation.config import (
    ValidationConfig,
    ValidationProfile,
    ValidationRules,
    get_validation_config,
    get_domain_validation_rules,
    register_domain_validation_rules,
    load_validation_config_from_env,
    load_validation_config,
)


def test_validation_config_defaults():
    """Test default values of ValidationConfig"""
    config = ValidationConfig()
    assert config.strict_mode is True
    assert config.auto_correct is False
    assert config.min_confidence_score == 0.7
    assert config.allowed_data_types == {
        "string", "integer", "float", "date", "boolean", "text"
    }


def test_get_validation_config_profiles():
    """Test different validation profiles"""
    # Test STRICT profile
    strict_config = get_validation_config(ValidationProfile.STRICT)
    assert strict_config.strict_mode is True
    assert strict_config.min_confidence_score == 0.9
    assert strict_config.validation_timeout == 60.0

    # Test LENIENT profile
    lenient_config = get_validation_config(ValidationProfile.LENIENT)
    assert lenient_config.strict_mode is False
    assert lenient_config.min_confidence_score == 0.5
    assert lenient_config.validation_timeout == 15.0

    # Test DEVELOPMENT profile
    dev_config = get_validation_config(ValidationProfile.DEVELOPMENT)
    assert dev_config.strict_mode is False
    assert dev_config.min_confidence_score == 0.3
    assert dev_config.enable_caching is False

    # Test STANDARD profile
    standard_config = get_validation_config(ValidationProfile.STANDARD)
    assert standard_config.strict_mode is True
    assert standard_config.min_confidence_score == 0.7
    assert standard_config.validation_timeout == 30.0


def test_get_validation_config_from_environment():
    """Test validation profile from environment variable"""
    # Test with valid profile
    os.environ["VALIDATION_PROFILE"] = "lenient"
    config = get_validation_config()
    assert config.strict_mode is False
    assert config.min_confidence_score == 0.5

    # Test with invalid profile (should default to STANDARD)
    os.environ["VALIDATION_PROFILE"] = "invalid_profile"
    config = get_validation_config()
    assert config.strict_mode is True
    assert config.min_confidence_score == 0.7

    # Clean up
    os.environ.pop("VALIDATION_PROFILE", None)


def test_validation_rules():
    """Test ValidationRules initialization and defaults"""
    rules = ValidationRules(domain="test")
    assert rules.required_columns == []
    assert rules.forbidden_columns == []
    assert rules.column_patterns == {}
    assert rules.custom_validators == {}

    # Test with custom values
    rules = ValidationRules(
        domain="test",
        required_columns=["id", "name"],
        forbidden_columns=["password"],
        column_patterns={"email": r".*@.*"},
    )
    assert rules.required_columns == ["id", "name"]
    assert rules.forbidden_columns == ["password"]
    assert rules.column_patterns == {"email": r".*@.*"}


def test_domain_validation_rules():
    """Test domain validation rules retrieval and registration"""
    # Test getting existing domain rules
    product_rules = get_domain_validation_rules("product")
    assert product_rules.domain == "product"
    assert all(col in product_rules.required_columns for col in ["product_id", "name", "price"])
    assert all(col in product_rules.forbidden_columns for col in ["internal_cost", "margin"])
    assert "sku" in product_rules.column_patterns
    assert "price" in product_rules.column_patterns

    # Test getting non-existent domain rules
    assert get_domain_validation_rules("nonexistent") is None

    # Test registering new domain rules
    new_rules = ValidationRules(
        domain="test",
        required_columns=["test_id"],
        forbidden_columns=["private"],
        column_patterns={"code": r"^[A-Z]{3}$"}
    )
    register_domain_validation_rules("test", new_rules)
    retrieved_rules = get_domain_validation_rules("test")
    assert retrieved_rules.domain == "test"
    assert retrieved_rules.required_columns == ["test_id"]
    assert retrieved_rules.forbidden_columns == ["private"]
    assert retrieved_rules.column_patterns["code"] == r"^[A-Z]{3}$"


def test_load_validation_config_from_env():
    """Test loading validation config from environment variables"""
    # Set environment variables
    os.environ["VALIDATION_STRICT_MODE"] = "false"
    os.environ["VALIDATION_AUTO_CORRECT"] = "true"
    os.environ["VALIDATION_MIN_CONFIDENCE"] = "0.85"
    os.environ["VALIDATION_MIN_COLUMNS"] = "5"
    os.environ["VALIDATION_CHECK_PII"] = "false"

    config = load_validation_config_from_env()
    assert config.strict_mode is False
    assert config.auto_correct is True
    assert config.min_confidence_score == 0.85
    assert config.min_columns == 5
    assert config.check_pii is False

    # Test invalid values
    os.environ["VALIDATION_MIN_CONFIDENCE"] = "invalid"
    os.environ["VALIDATION_MIN_COLUMNS"] = "invalid"
    config = load_validation_config_from_env()
    assert isinstance(config.min_confidence_score, float)
    assert isinstance(config.min_columns, int)

    # Clean up
    for key in [
        "VALIDATION_STRICT_MODE",
        "VALIDATION_AUTO_CORRECT",
        "VALIDATION_MIN_CONFIDENCE",
        "VALIDATION_MIN_COLUMNS",
        "VALIDATION_CHECK_PII",
    ]:
        os.environ.pop(key, None)


def test_load_validation_config_alias():
    """Test load_validation_config alias function"""
    # Test that it returns the same result as get_validation_config
    assert load_validation_config(ValidationProfile.STRICT).__dict__ == \
           get_validation_config(ValidationProfile.STRICT).__dict__
    assert load_validation_config(ValidationProfile.LENIENT).__dict__ == \
           get_validation_config(ValidationProfile.LENIENT).__dict__