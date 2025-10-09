"""
Base definitions for validation classes

This module defines the base classes and enums used across the validation system
to avoid circular imports between modules.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in LLM response"""
    field: str
    message: str
    severity: ValidationSeverity
    suggested_fix: Optional[str] = None
    raw_value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of LLM response validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    corrected_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]
    
    @property
    def has_critical_issues(self) -> bool:
        return len(self.critical_issues) > 0