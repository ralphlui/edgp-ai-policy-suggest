"""
Integration test to verify LLM validation works with the policy suggestion system
"""

import pytest
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.validation.middleware import AgentValidationContext
from app.validation.policy_validator import create_policy_validator
from app.validation.llm_validator import ValidationSeverity


class TestPolicySystemIntegration:
    """Test integration of validation with the policy suggestion system"""
    
    def test_schema_suggestion_input_validation(self):
        """Test validation of schema suggestion inputs"""
        user_id = "test_user"
        
        # Test legitimate schema creation request
        schema_input = """
        Create a customer domain schema with the following columns:
        - customer_id (primary key)
        - email (contact information)
        - phone (contact number)
        - registration_date
        - status (active/inactive)
        """
        
        with AgentValidationContext(user_id) as validator:
            # Should not raise an exception for valid input
            sanitized = validator.validate_input(schema_input)
            assert isinstance(sanitized, str), "Should return sanitized string"
            assert len(sanitized) > 0, "Sanitized input should not be empty"
    
    def test_rule_suggestion_input_validation(self):
        """Test validation of rule suggestion inputs"""
        user_id = "test_user"
        
        # Test legitimate rule suggestion request
        rule_input = """
        Suggest validation rules for customer table:
        - expect_column_values_to_not_be_null for customer_id
        - expect_column_values_to_match_regex for email field
        - expect_column_values_to_be_in_range for date fields
        """
        
        with AgentValidationContext(user_id) as validator:
            # Should not raise an exception for valid input
            sanitized = validator.validate_input(rule_input)
            assert isinstance(sanitized, str), "Should return sanitized string"
            assert len(sanitized) > 0, "Sanitized input should not be empty"
    
    def test_malicious_input_blocked(self):
        """Test that malicious inputs are still blocked"""
        user_id = "test_user"
        
        # Test malicious SQL injection attempt
        malicious_input = "DROP TABLE customers; DELETE FROM users WHERE 1=1;"
        
        with AgentValidationContext(user_id) as validator:
            # Should raise ValidationError for malicious input
            with pytest.raises(Exception):  # ValidationError or similar
                validator.validate_input(malicious_input)
    
    def test_schema_output_validation(self):
        """Test validation of schema generation outputs"""
        user_id = "test_user"
        
        # Simulate a schema response from LLM
        schema_response = {
            "columns": [
                {"name": "customer_id", "type": "integer", "description": "Unique customer identifier"},
                {"name": "email", "type": "varchar", "description": "Customer email address"},
                {"name": "created_at", "type": "timestamp", "description": "Account creation date"}
            ],
            "domain": "customer",
            "metadata": {
                "generated_by": "llm",
                "confidence": 0.9
            }
        }
        
        with AgentValidationContext(user_id) as validator:
            # Should not raise an exception for valid output
            filtered = validator.validate_output(schema_response, "schema")
            assert filtered is not None, "Should return filtered output"
    
    def test_rule_output_validation(self):
        """Test validation of rule suggestion outputs"""
        user_id = "test_user"
        
        # Simulate a rule response from LLM
        rule_response = [
            {
                "column": "customer_id",
                "rule": "expect_column_values_to_not_be_null",
                "rationale": "Primary key should never be null",
                "confidence": 0.95
            },
            {
                "column": "email",
                "rule": "expect_column_values_to_match_regex",
                "parameters": {"regex": r"^[^\s@]+@[^\s@]+\.[^\s@]+$"},
                "rationale": "Email format validation",
                "confidence": 0.85
            }
        ]
        
        with AgentValidationContext(user_id) as validator:
            # Should not raise an exception for valid output
            filtered = validator.validate_output(rule_response, "rule")
            assert filtered is not None, "Should return filtered output"
    
    def test_business_terminology_allowed(self):
        """Test that business terminology is allowed in policy context"""
        user_id = "test_user"
        
        # Test business governance language
        governance_input = """
        Implement data quality policy for customer domain:
        - Data governance framework compliance
        - Regulatory requirements (GDPR, CCPA)
        - PII data handling procedures
        - Data lineage tracking
        - Master data management rules
        """
        
        with AgentValidationContext(user_id) as validator:
            # Should not raise an exception for business terminology
            sanitized = validator.validate_input(governance_input)
            assert isinstance(sanitized, str), "Should return sanitized string"
    
    def test_gx_rule_terminology_allowed(self):
        """Test that Great Expectations rule terminology is allowed"""
        user_id = "test_user"
        
        # Test GX rule language
        gx_input = """
        Apply these Great Expectations rules:
        - expect_column_values_to_not_be_null
        - expect_column_values_to_be_of_type
        - expect_column_values_to_be_in_range
        - expect_column_values_to_match_regex
        - expect_table_row_count_to_be_between
        """
        
        with AgentValidationContext(user_id) as validator:
            # Should not raise an exception for GX terminology
            sanitized = validator.validate_input(gx_input)
            assert isinstance(sanitized, str), "Should return sanitized string"
    
    def test_email_in_business_context_allowed(self):
        """Test that email addresses in business context are handled appropriately"""
        validator = create_policy_validator()
        
        # Business context with email
        business_content = "Create customer table with email column to store addresses like user@company.com"
        result = validator.validate_content_safety(business_content)
        
        # Should be valid or have only low/medium severity issues
        if not result.is_valid:
            critical_issues = [issue for issue in result.issues if issue.severity == ValidationSeverity.CRITICAL]
            assert len(critical_issues) == 0, "Email in business context should not have critical issues"
    
    def test_sql_context_awareness(self):
        """Test that SQL terminology in schema design context is handled appropriately"""
        validator = create_policy_validator()
        
        # SQL in schema design context
        sql_content = """
        For the customer table design:
        - CREATE TABLE customers with appropriate columns
        - ALTER TABLE to add indexes for performance
        - UPDATE policies for data modification
        This is for schema design documentation purposes.
        """
        result = validator.validate_content_safety(sql_content)
        
        # Should be valid in business context
        assert result.is_valid, "SQL terminology in schema design context should be allowed"
    
    def test_rate_limiting_in_policy_context(self):
        """Test that rate limiting works appropriately for policy operations"""
        user_id = "policy_user"
        
        # Test multiple schema requests within limits
        with AgentValidationContext(user_id) as validator:
            for i in range(5):
                input_text = f"Create schema for domain_{i} with basic columns"
                # Should not raise an exception for requests within limits
                sanitized = validator.validate_input(input_text)
                assert isinstance(sanitized, str), f"Request {i+1} should return sanitized string"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])