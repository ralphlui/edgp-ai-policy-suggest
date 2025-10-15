"""
Compatibility layer for validation system

This module provides compatibility functions and wrappers to ensure
backward compatibility with existing code and tests.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

from app.validation.config import ValidationConfig as CurrentConfig
from app.validation.config import ValidationProfile as CurrentProfile
from app.validation.config import get_validation_config


# Import base definitions
from app.validation.validation_base import ValidationSeverity

# Backwards-compatible ValidationProfile enum
class ValidationProfile(Enum):
    """Legacy compatibility for ValidationProfile enum"""
    STRICT = "strict"
    STANDARD = "standard" 
    LENIENT = "lenient"
    DEVELOPMENT = "development"
    

# Backwards-compatible ValidationIssue class
class ValidationIssue:
    """Legacy compatibility for ValidationIssue class"""
    def __init__(self, field: str, severity: ValidationSeverity, message: str, suggestion: str = None):
        self.field = field
        self.severity = severity
        self.message = message
        self.suggestion = suggestion


# Backwards-compatible ValidationResult class
class ValidationResult:
    """Legacy compatibility for ValidationResult class"""
    def __init__(self, is_valid: bool, confidence_score: float, issues: List[ValidationIssue], 
                 corrected_data: Optional[Dict[str, Any]] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.is_valid = is_valid
        self.confidence_score = confidence_score
        self.issues = issues
        self.corrected_data = corrected_data
        self.metadata = metadata or {}


# Backwards-compatible ValidationConfig class
class ValidationConfig:
    """Legacy compatibility for ValidationConfig class"""
    def __init__(self, profile=ValidationProfile.STANDARD, max_issues_allowed=5,
                 min_confidence_score=0.7, enable_auto_correction=True,
                 schema_validation_enabled=True, rule_validation_enabled=True,
                 content_validation_enabled=True):
        self.profile = profile
        self.max_issues_allowed = max_issues_allowed
        self.min_confidence_score = min_confidence_score
        self.enable_auto_correction = enable_auto_correction
        self.schema_validation_enabled = schema_validation_enabled
        self.rule_validation_enabled = rule_validation_enabled
        self.content_validation_enabled = content_validation_enabled
        self.domain_rules = {}


# Backwards-compatible LLMResponseValidator class
class LLMResponseValidator:
    """Legacy compatibility for LLMResponseValidator class"""
    def __init__(self, config=None):
        self.config = config or ValidationConfig()
        
        # Map to current implementation
        profile_name = self.config.profile.value if hasattr(self.config, 'profile') else 'standard'
        try:
            current_profile = CurrentProfile(profile_name)
            self.current_config = get_validation_config(current_profile)
        except ValueError:
            self.current_config = get_validation_config()
    
    def validate_response(self, response: Dict[str, Any], response_type: str, 
                          strict_mode: bool = None, auto_correct: bool = None) -> ValidationResult:
        """Compatibility wrapper for validate_llm_response"""
        
        # Import here to avoid circular import
        from app.validation.llm_validator import validate_llm_response
        
        # Use config values if not explicitly provided
        if strict_mode is None:
            strict_mode = True
        if auto_correct is None:
            auto_correct = getattr(self.config, 'enable_auto_correction', False)
        
        # Call current implementation
        current_result = validate_llm_response(
            response=response,
            response_type=response_type,
            strict_mode=strict_mode,
            auto_correct=auto_correct
        )
        
        # Convert to legacy result format
        issues = []
        for issue in current_result.issues:
            sev_value = issue.severity.value
            issues.append(ValidationIssue(
                field=issue.field,
                severity=ValidationSeverity(sev_value),
                message=issue.message,
                suggestion=issue.suggested_fix
            ))
        
        return ValidationResult(
            is_valid=current_result.is_valid,
            confidence_score=current_result.confidence_score,
            issues=issues,
            corrected_data=current_result.corrected_data,
            metadata={"timestamp": datetime.now().isoformat()}
        )
    
    def get_active_rules(self) -> Dict[str, Any]:
        """Get active validation rules"""
        return {
            "schema_rules": {
                "strict_mode": True,
                "min_confidence_score": getattr(self.config, "min_confidence_score", 0.7),
                "max_issues_allowed": getattr(self.config, "max_issues_allowed", 5)
            }
        }


# Function aliases for backwards compatibility
def load_validation_config(profile=None):
    """Legacy compatibility for load_validation_config"""
    if profile and isinstance(profile, ValidationProfile):
        profile = CurrentProfile(profile.value)
        
    return ValidationConfig()