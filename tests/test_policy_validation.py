"""
Test the policy-aware validation system with domain-specific examples
"""

import pytest
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.validation.policy_validator import (
    PolicyContentValidator, PolicyInputSanitizer, 
    create_policy_validator, create_policy_sanitizer,
    PolicyValidationConfig
)
from app.validation.llm_validator import ValidationSeverity


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