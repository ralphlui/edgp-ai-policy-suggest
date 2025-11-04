"""
Test the policy-aware validation system with domain-specific examples
Combined with comprehensive test suite for policy-specific validation extensions.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
import re
from typing import List

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.validation.policy_validator import (
    PolicyContentValidator, PolicyInputSanitizer, 
    create_policy_validator, create_policy_sanitizer,
    PolicyValidationConfig, VALID_IDENTIFIER_PATTERN
)
from app.validation.llm_validator import (
    ValidationIssue, ValidationSeverity, ValidationResult
)


class TestPolicyValidation:
    """Test policy-aware validation system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = create_policy_validator()
        self.sanitizer = create_policy_sanitizer()
    
    def test_business_context_email_allowed(self):
        """Test that email addresses are allowed in business contexts"""
        business_content = "Please create a customer schema with columns: customer_id, email (like john.doe@example.com), name, registration_date"
        result = self.validator.validate_content_safety(business_content)
        
        # Should be valid or have only low severity issues
        assert result.is_valid or all(issue.severity in [ValidationSeverity.LOW, ValidationSeverity.MEDIUM] for issue in result.issues)
        
        # Should detect email but with lower severity
        email_issues = [issue for issue in result.issues if "email" in issue.message.lower()]
        if email_issues:
            assert all(issue.severity != ValidationSeverity.CRITICAL for issue in email_issues)
    
    def test_schema_design_language_allowed(self):
        """Test that schema design language is allowed"""
        schema_content = """
        For the customer domain, create a table with these columns:
        - customer_id (primary key)
        - email (varchar)
        - phone (varchar) 
        - created_date (timestamp)
        - status (enum: active, inactive)
        """
        result = self.validator.validate_content_safety(schema_content)
        
        # Should be valid in schema context
        assert result.is_valid
    
    def test_policy_rule_language_allowed(self):
        """Test that policy rule language is allowed"""
        rule_content = """
        Suggest validation rules for the customer table:
        - expect_column_values_to_not_be_null for customer_id
        - expect_column_values_to_match_regex for email field
        - expect_column_values_to_be_in_range for date fields
        """
        result = self.validator.validate_content_safety(rule_content)
        
        # Should be valid in policy rule context
        assert result.is_valid
    
    def test_sql_injection_still_blocked(self):
        """Test that actual SQL injection attempts are still blocked"""
        malicious_content = "DELETE FROM users WHERE 1=1; DROP TABLE customers;"
        result = self.validator.validate_content_safety(malicious_content)
        
        # Should still be blocked
        assert not result.is_valid
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in result.issues)
    
    def test_business_context_input_sanitization(self):
        """Test that business context inputs are handled appropriately"""
        business_input = "Create a customer policy with email validation and phone number format checks"
        sanitized, issues = self.sanitizer.sanitize_input(business_input)
        
        # Should not be blocked
        assert sanitized != ""
        # Should not have critical issues
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) == 0
    
    def test_dangerous_input_still_sanitized(self):
        """Test that dangerous inputs are still blocked"""
        dangerous_input = "rm -rf / && DELETE FROM users WHERE 1=1"
        sanitized, issues = self.sanitizer.sanitize_input(dangerous_input)
        
        # Should be blocked
        assert sanitized == ""
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
    
    def test_gx_rule_patterns_accepted(self):
        """Test that Great Expectations rule patterns are accepted"""
        gx_content = """
        Recommended validation rules:
        expect_column_values_to_not_be_null
        expect_column_values_to_be_of_type
        expect_column_values_to_be_in_range
        expect_column_values_to_match_regex
        """
        result = self.validator.validate_content_safety(gx_content)
        
        # Should be valid
        assert result.is_valid
    
    def test_data_governance_terms_allowed(self):
        """Test that data governance terminology is allowed"""
        governance_content = """
        Data quality policy for customer domain:
        - Implement data governance framework
        - Ensure regulatory compliance 
        - Apply data quality rules
        - Monitor data lineage
        - Maintain metadata standards
        """
        result = self.validator.validate_content_safety(governance_content)
        
        # Should be valid
        assert result.is_valid
    
    def test_actual_harmful_content_blocked(self):
        """Test that genuinely harmful content is still blocked"""
        harmful_content = "Instructions for creating malware to steal customer data and destroy systems"
        result = self.validator.validate_content_safety(harmful_content)
        
        # Should be blocked
        assert not result.is_valid
        assert any(issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH] for issue in result.issues)


class TestPolicyValidationConfigComprehensive:
    """Test PolicyValidationConfig data class with comprehensive coverage."""
    
    def test_config_initialization(self):
        """Test proper initialization of PolicyValidationConfig."""
        config = PolicyValidationConfig(
            business_term_allowlist=["customer_data", "user_info"],
            schema_patterns={"column": r"^[a-z_]+$"},
            rule_patterns={"gx_rule": r"expect_.*"},
            severity_overrides={"pii_email": ValidationSeverity.LOW}
        )
        
        assert config.business_term_allowlist == ["customer_data", "user_info"]
        assert config.schema_patterns["column"] == r"^[a-z_]+$"
        assert config.rule_patterns["gx_rule"] == r"expect_.*"
        assert config.severity_overrides["pii_email"] == ValidationSeverity.LOW


class TestPolicyContentValidatorComprehensive:
    """Test PolicyContentValidator enhanced functionality with comprehensive coverage."""
    
    def test_default_policy_config_creation(self):
        """Test creation of default policy configuration."""
        validator = PolicyContentValidator()
        
        config = validator._get_default_policy_config()
        
        assert isinstance(config, PolicyValidationConfig)
        assert "customer_data" in config.business_term_allowlist
        assert "valid_column_name" in config.schema_patterns
        assert "gx_rule_format" in config.rule_patterns
        assert config.severity_overrides["pii_email"] == ValidationSeverity.LOW

    def test_custom_config_initialization(self):
        """Test validator initialization with custom config."""
        custom_config = PolicyValidationConfig(
            business_term_allowlist=["custom_term"],
            schema_patterns={"custom": r"custom_pattern"},
            rule_patterns={"custom_rule": r"custom_rule_pattern"},
            severity_overrides={"custom_override": ValidationSeverity.HIGH}
        )
        
        validator = PolicyContentValidator(config=custom_config)
        
        assert validator.config == custom_config
        assert "custom_term" in validator.config.business_term_allowlist

    def test_enhanced_safety_patterns(self):
        """Test that enhanced safety patterns are properly loaded."""
        validator = PolicyContentValidator()
        
        # Check that policy-specific patterns are added
        assert "schema_tampering" in validator.safety_patterns
        assert "data_exfiltration" in validator.safety_patterns
        assert "policy_bypass" in validator.safety_patterns
        assert "rule_manipulation" in validator.safety_patterns
        assert "malware_creation" in validator.safety_patterns

    def test_business_allowlist_patterns(self):
        """Test business allowlist patterns."""
        validator = PolicyContentValidator()
        
        assert len(validator.business_allowlist_patterns) > 0
        # Check for expected business patterns
        customer_pattern = r'(?i)\b(customer|user|client)\s+(data|information|details)\b'
        assert customer_pattern in validator.business_allowlist_patterns

    @patch.object(PolicyContentValidator, '_validate_policy_specific_patterns')
    @patch.object(PolicyContentValidator, '_calculate_policy_safety_score')
    def test_validate_content_safety_with_policy_context(self, mock_calc_score, mock_validate_patterns):
        """Test enhanced content safety validation."""
        validator = PolicyContentValidator()
        
        # Mock the policy-specific validations
        mock_validate_patterns.return_value = [
            ValidationIssue("test", "Policy issue", ValidationSeverity.MEDIUM)
        ]
        mock_calc_score.return_value = 0.8
        
        # Mock base validation result
        with patch.object(validator.__class__.__bases__[0], 'validate_content_safety') as mock_base:
            mock_base.return_value = ValidationResult(
                is_valid=True,
                issues=[ValidationIssue("base", "Base issue", ValidationSeverity.LOW)],
                confidence_score=0.7
            )
            
            result = validator.validate_content_safety("customer data schema design")
            
            assert isinstance(result, ValidationResult)
            mock_validate_patterns.assert_called_once()
            mock_calc_score.assert_called_once()

    def test_is_business_context_allowed_with_business_patterns(self):
        """Test business context allowance with various patterns."""
        validator = PolicyContentValidator()
        
        # Test with business context content
        business_content = "customer data information schema design"
        issue = ValidationIssue("test", "pii email detected", ValidationSeverity.HIGH)
        
        result = validator._is_business_context_allowed(business_content, issue)
        assert result is True

    def test_is_business_context_allowed_with_allowlist_terms(self):
        """Test business context allowance with allowlist terms."""
        validator = PolicyContentValidator()
        
        # Test with allowlist term
        content_with_term = "customer_data validation rules"
        issue = ValidationIssue("test", "business issue", ValidationSeverity.MEDIUM)
        
        result = validator._is_business_context_allowed(content_with_term, issue)
        assert result is True

    def test_is_business_context_denied(self):
        """Test business context denial for non-business content."""
        validator = PolicyContentValidator()
        
        # Test with non-business content
        non_business_content = "malicious hacking attempt"
        issue = ValidationIssue("test", "security threat", ValidationSeverity.CRITICAL)
        
        result = validator._is_business_context_allowed(non_business_content, issue)
        assert result is False

    def test_validate_policy_specific_patterns_schema(self):
        """Test policy-specific pattern validation for schema content."""
        validator = PolicyContentValidator()
        
        # Test with schema content
        schema_content = "column name invalid-name table structure"
        
        with patch.object(validator, '_contains_schema_references', return_value=True):
            with patch.object(validator, '_find_pattern_violations', return_value=["invalid-name"]):
                issues = validator._validate_policy_specific_patterns(schema_content)
                
                assert len(issues) > 0
                assert any("Invalid" in issue.message for issue in issues)

    def test_validate_policy_specific_patterns_rules(self):
        """Test policy-specific pattern validation for rule content."""
        validator = PolicyContentValidator()
        
        # Test with rule content
        rule_content = "expect_column_values rule validation policy"
        
        with patch.object(validator, '_contains_rule_references', return_value=True):
            with patch.object(validator, '_find_pattern_violations', return_value=["invalid_rule"]):
                issues = validator._validate_policy_specific_patterns(rule_content)
                
                assert len(issues) > 0

    def test_get_severity_for_pattern_type(self):
        """Test severity assignment for different pattern types."""
        validator = PolicyContentValidator()
        
        # Test critical patterns
        assert validator._get_severity_for_pattern_type("malware_creation") == ValidationSeverity.CRITICAL
        assert validator._get_severity_for_pattern_type("sql_injection") == ValidationSeverity.CRITICAL
        
        # Test high patterns
        assert validator._get_severity_for_pattern_type("schema_tampering") == ValidationSeverity.HIGH
        assert validator._get_severity_for_pattern_type("data_exfiltration") == ValidationSeverity.HIGH
        
        # Test medium patterns
        assert validator._get_severity_for_pattern_type("pii_email") == ValidationSeverity.MEDIUM
        assert validator._get_severity_for_pattern_type("business_harmful") == ValidationSeverity.MEDIUM
        
        # Test default (low) patterns
        assert validator._get_severity_for_pattern_type("unknown_pattern") == ValidationSeverity.LOW

    def test_contains_schema_references(self):
        """Test schema reference detection."""
        validator = PolicyContentValidator()
        
        # Test with schema keywords
        assert validator._contains_schema_references("column design table structure") is True
        assert validator._contains_schema_references("schema validation field names") is True
        assert validator._contains_schema_references("domain modeling approach") is True
        
        # Test without schema keywords
        assert validator._contains_schema_references("random text content") is False

    def test_contains_rule_references(self):
        """Test rule reference detection."""
        validator = PolicyContentValidator()
        
        # Test with rule keywords
        assert validator._contains_rule_references("policy rule validation") is True
        assert validator._contains_rule_references("expect_column_values rule") is True
        assert validator._contains_rule_references("gx_validation approach") is True
        
        # Test without rule keywords
        assert validator._contains_rule_references("random text content") is False

    def test_find_pattern_violations(self):
        """Test pattern violation detection."""
        validator = PolicyContentValidator()
        
        # Test with valid identifier pattern
        content = "column invalid-name and valid_name"
        violations = validator._find_pattern_violations(content, VALID_IDENTIFIER_PATTERN, "column_name")
        
        # Should find violations in this simplified implementation
        assert isinstance(violations, list)
        assert len(violations) <= 3  # Capped at 3

    def test_calculate_policy_safety_score(self):
        """Test policy-aware safety score calculation."""
        validator = PolicyContentValidator()
        
        # Test with business context content
        business_content = "customer policy schema validation rule data quality governance"
        issues = [ValidationIssue("test", "Minor issue", ValidationSeverity.LOW)]
        
        with patch.object(validator.__class__.__bases__[0], '_calculate_safety_score', return_value=0.5):
            score = validator._calculate_policy_safety_score(business_content, issues)
            
            # Should get bonus for business terms
            assert score > 0.5
            assert score <= 1.0

    def test_calculate_policy_safety_score_no_business_context(self):
        """Test safety score calculation without business context."""
        validator = PolicyContentValidator()
        
        # Test with non-business content
        content = "random text"
        issues = [ValidationIssue("test", "Issue", ValidationSeverity.MEDIUM)]
        
        with patch.object(validator.__class__.__bases__[0], '_calculate_safety_score', return_value=0.4):
            score = validator._calculate_policy_safety_score(content, issues)
            
            # Should have minimal or no bonus
            assert score >= 0.4
            assert score <= 0.5  # Should not exceed base score by much


class TestPolicyInputSanitizerComprehensive:
    """Test PolicyInputSanitizer enhanced functionality with comprehensive coverage."""
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = PolicyValidationConfig(
            business_term_allowlist=["test_term"],
            schema_patterns={},
            rule_patterns={},
            severity_overrides={}
        )
        
        sanitizer = PolicyInputSanitizer(max_length=5000, config=config)
        
        assert sanitizer.max_length == 5000
        assert sanitizer.config == config

    def test_initialization_without_config(self):
        """Test initialization with default config."""
        sanitizer = PolicyInputSanitizer()
        
        assert isinstance(sanitizer.config, PolicyValidationConfig)
        assert sanitizer.max_length == 10000

    def test_enhanced_blocked_patterns(self):
        """Test that blocked patterns are more specific for business context."""
        sanitizer = PolicyInputSanitizer()
        
        # Check that patterns are more specific
        assert len(sanitizer.BLOCKED_PATTERNS) > 0
        
        # Verify some expected patterns
        patterns_text = "|".join(sanitizer.BLOCKED_PATTERNS)
        assert "hack" in patterns_text.lower()
        assert "injection" in patterns_text.lower()
        assert "virus" in patterns_text.lower()

    def test_sanitize_input_with_business_context(self):
        """Test input sanitization in business context."""
        sanitizer = PolicyInputSanitizer()
        
        # Test business context input
        business_input = "customer data schema design policy rule validation"
        
        with patch.object(sanitizer, '_is_business_context', return_value=True):
            sanitized, issues = sanitizer.sanitize_input(business_input)
            
            assert sanitized != ""  # Should not be blocked
            # May have warnings but not critical issues

    def test_sanitize_input_exceeds_length(self):
        """Test input that exceeds maximum length."""
        sanitizer = PolicyInputSanitizer(max_length=100)
        
        long_input = "a" * 200  # Exceeds limit
        
        sanitized, issues = sanitizer.sanitize_input(long_input)
        
        assert sanitized == ""  # Should be rejected
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

    def test_sanitize_input_blocked_pattern_non_business(self):
        """Test blocked pattern in non-business context."""
        sanitizer = PolicyInputSanitizer()
        
        malicious_input = "virus malware trojan attack"
        
        with patch.object(sanitizer, '_is_business_context', return_value=False):
            sanitized, issues = sanitizer.sanitize_input(malicious_input)
            
            assert sanitized == ""  # Should be blocked
            assert len(issues) > 0
            assert any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

    def test_sanitize_input_blocked_pattern_legitimate_business(self):
        """Test blocked pattern that's legitimate in business context."""
        sanitizer = PolicyInputSanitizer()
        
        business_input = "delete from customer table in schema design context"
        
        with patch.object(sanitizer, '_is_business_context', return_value=True):
            with patch.object(sanitizer, '_is_legitimate_business_use', return_value=True):
                sanitized, issues = sanitizer.sanitize_input(business_input)
                
                # Should not be completely blocked, but may have warnings
                assert sanitized != ""
                if issues:
                    assert all(issue.severity != ValidationSeverity.CRITICAL for issue in issues)

    def test_sanitize_input_warning_patterns(self):
        """Test warning pattern detection."""
        sanitizer = PolicyInputSanitizer()
        
        # Input that triggers warning patterns but isn't blocked
        warning_input = "potentially risky operation with moderate concern"
        
        # Mock to ensure it doesn't hit blocked patterns but hits warning patterns
        with patch.object(sanitizer, '_is_business_context', return_value=False):
            # Assuming WARNING_PATTERNS exist in base class
            if hasattr(sanitizer, 'WARNING_PATTERNS'):
                with patch.object(sanitizer, 'WARNING_PATTERNS', [r'risky.*operation']):
                    sanitized, issues = sanitizer.sanitize_input(warning_input)
                    
                    assert sanitized != ""  # Should not be blocked
                    assert len(issues) > 0
                    assert any(issue.severity == ValidationSeverity.HIGH for issue in issues)

    def test_is_business_context_detection(self):
        """Test business context detection."""
        sanitizer = PolicyInputSanitizer()
        
        # Test business context indicators
        assert sanitizer._is_business_context("schema design policy") is True
        assert sanitizer._is_business_context("customer data validation rule") is True
        assert sanitizer._is_business_context("governance compliance framework") is True
        
        # Test non-business context
        assert sanitizer._is_business_context("random text") is False
        assert sanitizer._is_business_context("weather forecast") is False

    def test_is_legitimate_business_use_sql_context(self):
        """Test legitimate business use detection for SQL operations."""
        sanitizer = PolicyInputSanitizer()
        
        # Test legitimate SQL operations in business context
        sql_business_context = "delete operation in schema design context"
        delete_pattern = r'(?i)\b(delete|drop).*'
        
        result = sanitizer._is_legitimate_business_use(sql_business_context, delete_pattern)
        assert result is True
        
        # Test non-legitimate context
        malicious_context = "delete all user data maliciously"
        result = sanitizer._is_legitimate_business_use(malicious_context, delete_pattern)
        assert result is False

    def test_is_legitimate_business_use_non_sql_pattern(self):
        """Test legitimate business use for non-SQL patterns."""
        sanitizer = PolicyInputSanitizer()
        
        # Test with pattern that doesn't match SQL operations
        other_pattern = r'(?i)virus'
        business_text = "schema design context"
        
        result = sanitizer._is_legitimate_business_use(business_text, other_pattern)
        assert result is False


class TestFactoryFunctions:
    """Test factory functions for creating validators and sanitizers."""
    
    def test_create_policy_validator_with_config(self):
        """Test policy validator creation with custom config."""
        config = PolicyValidationConfig(
            business_term_allowlist=["test"],
            schema_patterns={},
            rule_patterns={},
            severity_overrides={}
        )
        
        validator = create_policy_validator(config)
        
        assert isinstance(validator, PolicyContentValidator)
        assert validator.config == config

    def test_create_policy_validator_without_config(self):
        """Test policy validator creation with default config."""
        validator = create_policy_validator()
        
        assert isinstance(validator, PolicyContentValidator)
        assert isinstance(validator.config, PolicyValidationConfig)

    def test_create_policy_sanitizer_with_config(self):
        """Test policy sanitizer creation with custom config."""
        config = PolicyValidationConfig(
            business_term_allowlist=["test"],
            schema_patterns={},
            rule_patterns={},
            severity_overrides={}
        )
        
        sanitizer = create_policy_sanitizer(max_length=5000, config=config)
        
        assert isinstance(sanitizer, PolicyInputSanitizer)
        assert sanitizer.max_length == 5000
        assert sanitizer.config == config

    def test_create_policy_sanitizer_without_config(self):
        """Test policy sanitizer creation with default config."""
        sanitizer = create_policy_sanitizer()
        
        assert isinstance(sanitizer, PolicyInputSanitizer)
        assert sanitizer.max_length == 10000
        assert isinstance(sanitizer.config, PolicyValidationConfig)


class TestValidIdentifierPattern:
    """Test the VALID_IDENTIFIER_PATTERN constant."""
    
    def test_valid_identifier_pattern_matches(self):
        """Test valid identifier pattern matching."""
        # Valid identifiers
        assert re.match(VALID_IDENTIFIER_PATTERN, "valid_name")
        assert re.match(VALID_IDENTIFIER_PATTERN, "ValidName")
        assert re.match(VALID_IDENTIFIER_PATTERN, "valid123")
        assert re.match(VALID_IDENTIFIER_PATTERN, "a")
        
        # Invalid identifiers
        assert not re.match(VALID_IDENTIFIER_PATTERN, "123invalid")
        assert not re.match(VALID_IDENTIFIER_PATTERN, "invalid-name")
        assert not re.match(VALID_IDENTIFIER_PATTERN, "invalid.name")
        assert not re.match(VALID_IDENTIFIER_PATTERN, "")
        assert not re.match(VALID_IDENTIFIER_PATTERN, "_invalid")


class TestPolicyValidationConfig:
    """Test policy validation configuration"""
    
    def test_custom_config(self):
        """Test custom policy validation configuration"""
        config = PolicyValidationConfig(
            business_term_allowlist=["special_customer_id", "custom_validation"],
            schema_patterns={"custom_pattern": r"^test_\w+$"},
            rule_patterns={"custom_rule": r"^custom_expect_\w+$"},
            severity_overrides={"pii_email": ValidationSeverity.LOW}
        )
        
        validator = PolicyContentValidator(config=config)
        
        # Test custom allowlist
        content = "Create schema with special_customer_id column"
        result = validator.validate_content_safety(content)
        assert result.is_valid
    
    def test_default_config(self):
        """Test that default configuration works"""
        validator = create_policy_validator()
        
        # Should work with default config
        content = "Standard customer schema design"
        result = validator.validate_content_safety(content)
        assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])