"""
Enhanced unit tests for app_validator.py
"""

import pytest
from app.validation.app_validator import validate_domain_name, validate_column_schema


class TestDomainValidation:
    """Test domain name validation functionality"""
    
    def test_valid_domain_names(self):
        """Test valid domain name patterns"""
        valid_domains = [
            "customer",
            "user-profile", 
            "product_catalog",
            "order123",
            "test-domain_v2",
            "a",  # Single character
            "very-long-domain-name-with-many-hyphens-and_underscores123"
        ]
        
        for domain in valid_domains:
            result = validate_domain_name(domain)
            assert result == domain, f"Domain '{domain}' should be valid"
    
    def test_invalid_domain_names(self):
        """Test invalid domain name patterns"""
        invalid_domains = [
            "Customer",  # Uppercase
            "user Profile",  # Space
            "product@catalog",  # Special char
            "order.123",  # Dot
            "test/domain",  # Slash
            "domain#name",  # Hash
            "user%data",  # Percent
            "",  # Empty string
            "UPPERCASE",  # All caps
            "Mixed_Case",  # Mixed case
        ]
        
        for domain in invalid_domains:
            with pytest.raises(ValueError) as exc_info:
                validate_domain_name(domain)
            assert "Invalid domain name" in str(exc_info.value)
            assert "Only lowercase letters" in str(exc_info.value)
    
    def test_domain_validation_error_message(self):
        """Test specific error message format"""
        with pytest.raises(ValueError) as exc_info:
            validate_domain_name("Invalid@Domain")
        
        error_msg = str(exc_info.value)
        assert "Invalid domain name 'Invalid@Domain'" in error_msg
        assert "Only lowercase letters, digits, underscores, and hyphens are allowed" in error_msg


class TestColumnSchemaValidation:
    """Test column schema validation functionality"""
    
    def test_valid_column_schema(self):
        """Test valid column schema structures"""
        valid_schemas = [
            {
                "customer_id": {
                    "dtype": "int64",
                    "sample_values": ["1", "2", "3"]
                },
                "name": {
                    "dtype": "string", 
                    "sample_values": ["John", "Jane", "Bob"]
                }
            },
            {
                "email": {
                    "dtype": "string",
                    "sample_values": ["user@example.com"]
                }
            },
            {},  # Empty schema should be valid
        ]
        
        for schema in valid_schemas:
            result = validate_column_schema(schema)
            assert result is True, f"Schema should be valid: {schema}"
    
    def test_invalid_schema_not_dict(self):
        """Test invalid schema - not a dictionary"""
        invalid_schemas = [
            "not_a_dict",
            ["list", "not", "dict"],
            42,
            None,
            True
        ]
        
        for schema in invalid_schemas:
            result = validate_column_schema(schema)
            assert result is False, f"Schema should be invalid: {schema}"
    
    def test_invalid_column_info_not_dict(self):
        """Test invalid column info - not a dictionary"""
        invalid_schemas = [
            {
                "column1": "not_a_dict"
            },
            {
                "column1": ["not", "dict"]
            },
            {
                "column1": 42
            },
            {
                "column1": None
            }
        ]
        
        for schema in invalid_schemas:
            result = validate_column_schema(schema)
            assert result is False, f"Schema should be invalid: {schema}"
    
    def test_missing_required_fields(self):
        """Test schemas missing required fields"""
        invalid_schemas = [
            {
                "column1": {
                    "dtype": "string"
                    # Missing sample_values
                }
            },
            {
                "column1": {
                    "sample_values": ["value1", "value2"]
                    # Missing dtype
                }
            },
            {
                "column1": {}  # Missing both fields
            }
        ]
        
        for schema in invalid_schemas:
            result = validate_column_schema(schema)
            assert result is False, f"Schema should be invalid: {schema}"
    
    def test_invalid_sample_values_not_list(self):
        """Test invalid sample_values - not a list"""
        invalid_schemas = [
            {
                "column1": {
                    "dtype": "string",
                    "sample_values": "not_a_list"
                }
            },
            {
                "column1": {
                    "dtype": "string", 
                    "sample_values": {"not": "list"}
                }
            },
            {
                "column1": {
                    "dtype": "string",
                    "sample_values": 42
                }
            }
        ]
        
        for schema in invalid_schemas:
            result = validate_column_schema(schema)
            assert result is False, f"Schema should be invalid: {schema}"
    
    def test_edge_cases(self):
        """Test edge cases for column schema validation"""
        edge_cases = [
            # Empty sample_values list should be valid
            {
                "column1": {
                    "dtype": "string",
                    "sample_values": []
                }
            },
            # Extra fields should be valid
            {
                "column1": {
                    "dtype": "string",
                    "sample_values": ["value"],
                    "extra_field": "should_be_ignored"
                }
            }
        ]
        
        for schema in edge_cases:
            result = validate_column_schema(schema)
            assert result is True, f"Schema should be valid: {schema}"
    
    def test_complex_valid_schema(self):
        """Test a complex but valid schema"""
        complex_schema = {
            "id": {
                "dtype": "int64",
                "sample_values": ["1", "2", "3", "4", "5"]
            },
            "email": {
                "dtype": "string",
                "sample_values": ["user1@example.com", "user2@test.com"]
            },
            "age": {
                "dtype": "int32", 
                "sample_values": ["25", "30", "35"]
            },
            "is_active": {
                "dtype": "boolean",
                "sample_values": ["true", "false"]
            },
            "created_date": {
                "dtype": "datetime",
                "sample_values": ["2023-01-01", "2023-02-01"]
            }
        }
        
        result = validate_column_schema(complex_schema)
        assert result is True, "Complex schema should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])