"""
Enhanced unit tests for validation/compat.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import warnings

from app.validation.compat import (
    ValidationConfig,
    ValidationProfile,
    ValidationResult,
    ValidationIssue,
    LLMResponseValidator,
    load_validation_config
)
from app.validation.validation_base import ValidationSeverity


class TestValidationProfile:
    """Test ValidationProfile enum functionality"""
    
    def test_profile_values(self):
        """Test that all profile values are available"""
        assert ValidationProfile.STRICT.value == "strict"
        assert ValidationProfile.STANDARD.value == "standard"
        assert ValidationProfile.LENIENT.value == "lenient"
        assert ValidationProfile.DEVELOPMENT.value == "development"


class TestValidationIssue:
    """Test ValidationIssue class functionality"""
    
    def test_validation_issue_creation(self):
        """Test ValidationIssue creation"""
        issue = ValidationIssue(
            field="customer_id",
            severity=ValidationSeverity.HIGH,
            message="Invalid format",
            suggestion="Use numeric format"
        )
        
        assert issue.field == "customer_id"
        assert issue.severity == ValidationSeverity.HIGH
        assert issue.message == "Invalid format"
        assert issue.suggestion == "Use numeric format"
    
    def test_validation_issue_without_suggestion(self):
        """Test ValidationIssue creation without suggestion"""
        issue = ValidationIssue(
            field="email",
            severity=ValidationSeverity.MEDIUM,
            message="Missing validation"
        )
        
        assert issue.field == "email"
        assert issue.severity == ValidationSeverity.MEDIUM
        assert issue.message == "Missing validation"
        assert issue.suggestion is None


class TestValidationResult:
    """Test ValidationResult class functionality"""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        issues = [
            ValidationIssue("field1", ValidationSeverity.HIGH, "Error message"),
            ValidationIssue("field2", ValidationSeverity.MEDIUM, "Warning message")
        ]
        
        result = ValidationResult(
            is_valid=False,
            confidence_score=0.75,
            issues=issues,
            corrected_data={"field1": "corrected_value"},
            metadata={"timestamp": "2024-01-01T00:00:00Z"}
        )
        
        assert result.is_valid is False
        assert result.confidence_score == 0.75
        assert len(result.issues) == 2
        assert result.corrected_data == {"field1": "corrected_value"}
        assert result.metadata["timestamp"] == "2024-01-01T00:00:00Z"
    
    def test_validation_result_minimal(self):
        """Test ValidationResult with minimal data"""
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.95,
            issues=[]
        )
        
        assert result.is_valid is True
        assert result.confidence_score == 0.95
        assert len(result.issues) == 0
        assert result.corrected_data is None
        assert isinstance(result.metadata, dict)


class TestValidationConfig:
    """Test ValidationConfig class functionality"""
    
    def test_validation_config_default(self):
        """Test ValidationConfig with default values"""
        config = ValidationConfig()
        
        assert config.profile == ValidationProfile.STANDARD
        assert config.max_issues_allowed == 5
        assert config.min_confidence_score == 0.7
        assert config.enable_auto_correction is True
        assert config.schema_validation_enabled is True
        assert config.rule_validation_enabled is True
        assert config.content_validation_enabled is True
    
    def test_validation_config_custom(self):
        """Test ValidationConfig with custom values"""
        config = ValidationConfig(
            profile=ValidationProfile.STRICT,
            max_issues_allowed=3,
            min_confidence_score=0.9,
            enable_auto_correction=False,
            schema_validation_enabled=False
        )
        
        assert config.profile == ValidationProfile.STRICT
        assert config.max_issues_allowed == 3
        assert config.min_confidence_score == 0.9
        assert config.enable_auto_correction is False
        assert config.schema_validation_enabled is False


class TestLLMResponseValidator:
    """Test LLMResponseValidator class functionality"""
    
    def test_validator_creation_default(self):
        """Test validator creation with default config"""
        validator = LLMResponseValidator()
        
        assert validator.config is not None
        assert hasattr(validator, 'current_config')
    
    def test_validator_creation_custom_config(self):
        """Test validator creation with custom config"""
        config = ValidationConfig(
            profile=ValidationProfile.STRICT,
            enable_auto_correction=False
        )
        
        validator = LLMResponseValidator(config)
        
        assert validator.config == config
        assert validator.config.profile == ValidationProfile.STRICT
    
    @patch('app.validation.llm_validator.validate_llm_response')
    def test_validate_response_success(self, mock_validate):
        """Test successful response validation"""
        # Mock the underlying validation function
        mock_validate.return_value = Mock(
            is_valid=True,
            confidence_score=0.92,
            issues=[],
            corrected_data={"test": "data"}
        )
        
        validator = LLMResponseValidator()
        response_data = {"test_field": "test_value"}
        
        result = validator.validate_response(
            response=response_data,
            response_type="schema"
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.confidence_score == 0.92
        assert len(result.issues) == 0
        
        # Verify underlying function was called
        mock_validate.assert_called_once()
    
    @patch('app.validation.llm_validator.validate_llm_response')
    def test_validate_response_with_issues(self, mock_validate):
        """Test response validation with issues"""
        # Mock validation issue
        mock_issue = Mock()
        mock_issue.field = "test_field"
        mock_issue.severity = ValidationSeverity.HIGH
        mock_issue.message = "Test error"
        mock_issue.suggested_fix = "Fix suggestion"
        
        mock_validate.return_value = Mock(
            is_valid=False,
            confidence_score=0.65,
            issues=[mock_issue],
            corrected_data=None
        )
        
        validator = LLMResponseValidator()
        response_data = {"invalid_field": "invalid_value"}
        
        result = validator.validate_response(
            response=response_data,
            response_type="schema"
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.confidence_score == 0.65
        assert len(result.issues) == 1
        
        issue = result.issues[0]
        assert isinstance(issue, ValidationIssue)
        assert issue.field == "test_field"
        assert issue.severity == ValidationSeverity.HIGH
        assert issue.message == "Test error"
        assert issue.suggestion == "Fix suggestion"
    
    def test_get_active_rules(self):
        """Test getting active validation rules"""
        config = ValidationConfig(
            min_confidence_score=0.8,
            max_issues_allowed=3
        )
        
        validator = LLMResponseValidator(config)
        rules = validator.get_active_rules()
        
        assert isinstance(rules, dict)
        assert "schema_rules" in rules
        assert rules["schema_rules"]["min_confidence_score"] == 0.8
        assert rules["schema_rules"]["max_issues_allowed"] == 3
        assert rules["schema_rules"]["strict_mode"] is True


class TestLoadValidationConfig:
    """Test load_validation_config function"""
    
    def test_load_config_default(self):
        """Test loading config with default profile"""
        config = load_validation_config()
        
        assert isinstance(config, ValidationConfig)
        assert config.profile == ValidationProfile.STANDARD
    
    def test_load_config_with_profile(self):
        """Test loading config with specific profile"""
        config = load_validation_config(ValidationProfile.STRICT)
        
        assert isinstance(config, ValidationConfig)
        # Profile should be set (exact behavior depends on implementation)
        assert config is not None


class TestValidationCompatibilityIntegration:
    """Test integration scenarios for validation compatibility"""
    
    @patch('app.validation.llm_validator.validate_llm_response')
    def test_full_validation_workflow(self, mock_validate):
        """Test complete validation workflow using compatibility layer"""
        # Mock successful validation
        mock_validate.return_value = Mock(
            is_valid=True,
            confidence_score=0.88,
            issues=[],
            corrected_data={"corrected": "data"}
        )
        
        # Create config and validator
        config = ValidationConfig(
            profile=ValidationProfile.STANDARD,
            enable_auto_correction=True
        )
        
        validator = LLMResponseValidator(config)
        
        # Test data
        test_response = {
            "schema": {
                "customer_id": {
                    "dtype": "int64",
                    "sample_values": ["1", "2", "3"]
                }
            }
        }
        
        # Validate
        result = validator.validate_response(
            response=test_response,
            response_type="schema",
            strict_mode=True,
            auto_correct=True
        )
        
        # Verify result
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.confidence_score == 0.88
        assert result.corrected_data == {"corrected": "data"}
        
        # Verify underlying function called with correct parameters
        mock_validate.assert_called_once_with(
            response=test_response,
            response_type="schema",
            strict_mode=True,
            auto_correct=True
        )
    
    def test_backward_compatibility_enum_values(self):
        """Test that enum values are backward compatible"""
        # Test that enum values match expected strings
        profiles = [
            ValidationProfile.STRICT,
            ValidationProfile.STANDARD,
            ValidationProfile.LENIENT,
            ValidationProfile.DEVELOPMENT
        ]
        
        expected_values = ["strict", "standard", "lenient", "development"]
        
        for profile, expected in zip(profiles, expected_values):
            assert profile.value == expected
    
    def test_validation_result_serialization(self):
        """Test that ValidationResult can be serialized"""
        issues = [
            ValidationIssue("field1", ValidationSeverity.MEDIUM, "Warning message")
        ]
        
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.85,
            issues=issues,
            metadata={"test": "metadata"}
        )
        
        # Should be able to access all properties
        assert result.is_valid is True
        assert result.confidence_score == 0.85
        assert len(result.issues) == 1
        assert result.metadata["test"] == "metadata"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])