"""
Comprehensive LLM Response Validation Module

This module provides validation for LLM responses across different use cases in the application.
Includes schema validation, content validation, and safety checks.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError, field_validator
from app.core.exceptions import SchemaGenerationError, ValidationError as AppValidationError

# Import base definitions
from app.validation.validation_base import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult
)

logger = logging.getLogger(__name__)


class LLMResponseValidator:
    """Main validator for LLM responses"""
    
    def __init__(self, strict_mode: bool = True, auto_correct: bool = False):
        """
        Initialize LLM Response Validator
        
        Args:
            strict_mode: If True, any validation error fails the validation
            auto_correct: If True, attempt to automatically fix common issues
        """
        self.strict_mode = strict_mode
        self.auto_correct = auto_correct
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_schema_response(self, response: Dict[str, Any]) -> ValidationResult:
        """
        Validate LLM response for schema generation
        
        Args:
            response: Raw LLM response containing schema data
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        corrected_data = response.copy() if self.auto_correct else None
        
        # Check basic structure
        if not isinstance(response, dict):
            issues.append(ValidationIssue(
                field="root",
                message="Response must be a dictionary",
                severity=ValidationSeverity.CRITICAL
            ))
            return ValidationResult(is_valid=False, issues=issues)
        
        # Validate columns field
        columns = response.get("columns", [])
        if not isinstance(columns, list):
            issues.append(ValidationIssue(
                field="columns",
                message="Columns field must be a list",
                severity=ValidationSeverity.CRITICAL,
                raw_value=columns
            ))
        elif len(columns) == 0:
            issues.append(ValidationIssue(
                field="columns",
                message="At least one column is required",
                severity=ValidationSeverity.CRITICAL
            ))
        else:
            # Validate individual columns
            valid_columns = []
            for i, column in enumerate(columns):
                column_issues, corrected_column = self._validate_column(column, i)
                issues.extend(column_issues)
                
                if self.auto_correct and corrected_column:
                    valid_columns.append(corrected_column)
                elif not any(issue.severity == ValidationSeverity.CRITICAL for issue in column_issues):
                    valid_columns.append(column)
            
            if self.auto_correct and corrected_data:
                corrected_data["columns"] = valid_columns
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(issues, len(columns))
        
        # Determine if valid
        is_valid = not self.has_blocking_issues(issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            corrected_data=corrected_data,
            confidence_score=confidence_score
        )
    
    def _validate_column(self, column: Dict[str, Any], index: int) -> Tuple[List[ValidationIssue], Optional[Dict[str, Any]]]:
        """Validate individual column definition"""
        issues = []
        corrected_column = column.copy() if self.auto_correct else None
        
        # Validate column name
        name = column.get("name", "")
        if not name:
            issues.append(ValidationIssue(
                field=f"columns[{index}].name",
                message="Column name is required",
                severity=ValidationSeverity.CRITICAL
            ))
        elif not isinstance(name, str):
            issues.append(ValidationIssue(
                field=f"columns[{index}].name",
                message="Column name must be a string",
                severity=ValidationSeverity.CRITICAL,
                raw_value=name
            ))
        elif not self._is_valid_identifier(name):
            issues.append(ValidationIssue(
                field=f"columns[{index}].name",
                message=f"Column name '{name}' is not a valid identifier",
                severity=ValidationSeverity.HIGH,
                suggested_fix=self._suggest_valid_identifier(name)
            ))
            if self.auto_correct and corrected_column:
                corrected_column["name"] = self._suggest_valid_identifier(name)
        
        # Validate column type
        col_type = column.get("type", "")
        valid_types = {"string", "integer", "float", "date", "boolean", "text"}
        if not col_type:
            issues.append(ValidationIssue(
                field=f"columns[{index}].type",
                message="Column type is required",
                severity=ValidationSeverity.CRITICAL
            ))
        elif col_type not in valid_types:
            issues.append(ValidationIssue(
                field=f"columns[{index}].type",
                message=f"Invalid column type '{col_type}'. Valid types: {valid_types}",
                severity=ValidationSeverity.HIGH,
                suggested_fix="string"  # Default fallback
            ))
            if self.auto_correct and corrected_column:
                corrected_column["type"] = "string"
        
        # Validate samples
        samples = column.get("samples", [])
        if not isinstance(samples, list):
            issues.append(ValidationIssue(
                field=f"columns[{index}].samples",
                message="Samples must be a list",
                severity=ValidationSeverity.CRITICAL,
                raw_value=samples
            ))
        elif len(samples) < 3:
            issues.append(ValidationIssue(
                field=f"columns[{index}].samples",
                message=f"At least 3 samples required, got {len(samples)}",
                severity=ValidationSeverity.HIGH
            ))
        else:
            # Validate sample consistency with type
            type_validation_issues = self._validate_samples_for_type(samples, col_type, index)
            issues.extend(type_validation_issues)
        
        return issues, corrected_column
    
    def _validate_samples_for_type(self, samples: List[str], col_type: str, column_index: int) -> List[ValidationIssue]:
        """Validate that samples match the declared column type"""
        issues = []
        
        if col_type == "integer":
            for i, sample in enumerate(samples):
                try:
                    int(sample)
                except ValueError:
                    issues.append(ValidationIssue(
                        field=f"columns[{column_index}].samples[{i}]",
                        message=f"Sample '{sample}' is not a valid integer",
                        severity=ValidationSeverity.MEDIUM,
                        raw_value=sample
                    ))
        
        elif col_type == "float":
            for i, sample in enumerate(samples):
                try:
                    float(sample)
                except ValueError:
                    issues.append(ValidationIssue(
                        field=f"columns[{column_index}].samples[{i}]",
                        message=f"Sample '{sample}' is not a valid float",
                        severity=ValidationSeverity.MEDIUM,
                        raw_value=sample
                    ))
        
        elif col_type == "boolean":
            valid_bool_values = {"true", "false", "1", "0", "yes", "no"}
            for i, sample in enumerate(samples):
                if sample.lower() not in valid_bool_values:
                    issues.append(ValidationIssue(
                        field=f"columns[{column_index}].samples[{i}]",
                        message=f"Sample '{sample}' is not a valid boolean value",
                        severity=ValidationSeverity.MEDIUM,
                        raw_value=sample
                    ))
        
        elif col_type == "date":
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
                r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
                r"\d{2}-\d{2}-\d{4}",  # MM-DD-YYYY
            ]
            for i, sample in enumerate(samples):
                if not any(re.match(pattern, sample) for pattern in date_patterns):
                    issues.append(ValidationIssue(
                        field=f"columns[{column_index}].samples[{i}]",
                        message=f"Sample '{sample}' is not a valid date format",
                        severity=ValidationSeverity.MEDIUM,
                        raw_value=sample
                    ))
        
        return issues
    
    def _is_valid_identifier(self, name: str) -> bool:
        """Check if name is a valid Python identifier"""
        return name.isidentifier()
    
    def _suggest_valid_identifier(self, name: str) -> str:
        """Suggest a valid identifier based on the given name"""
        # Remove invalid characters and replace with underscore
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure it doesn't start with a number
        if clean_name and clean_name[0].isdigit():
            clean_name = f"col_{clean_name}"
        
        # Ensure it's not empty
        if not clean_name:
            clean_name = "unnamed_column"
        
        return clean_name
    
    def _calculate_confidence_score(self, issues: List[ValidationIssue], total_columns: int) -> float:
        """Calculate confidence score based on validation issues"""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.LOW: 0.1,
            ValidationSeverity.MEDIUM: 0.3,
            ValidationSeverity.HIGH: 0.6,
            ValidationSeverity.CRITICAL: 1.0
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in issues)
        max_possible_weight = len(issues) * 1.0  # If all were critical
        
        # Calculate confidence (0-1, where 1 is perfect)
        confidence = max(0.0, 1.0 - (total_weight / max(max_possible_weight, 1.0)))
        
        # Adjust for number of columns (more columns = more confidence in valid ones)
        if total_columns > 0:
            confidence *= min(1.0, total_columns / 5.0)  # Normalize around 5 columns
        
        return round(confidence, 3)
    
    def has_blocking_issues(self, issues: List[ValidationIssue]) -> bool:
        """Check if issues contain blocking problems"""
        if self.strict_mode:
            return any(issue.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL] for issue in issues)
        else:
            return any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
    
    def validate_rule_response(self, response: Dict[str, Any]) -> ValidationResult:
        """
        Validate LLM response for rule generation
        
        Args:
            response: Raw LLM response containing rule data
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        
        # Check for required fields
        required_fields = ["rules", "explanation"]
        for field in required_fields:
            if field not in response:
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity=ValidationSeverity.CRITICAL
                ))
        
        # Validate rules structure
        rules = response.get("rules", [])
        if isinstance(rules, list):
            for i, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    issues.append(ValidationIssue(
                        field=f"rules[{i}]",
                        message="Rule must be a dictionary",
                        severity=ValidationSeverity.HIGH
                    ))
                else:
                    # Validate rule fields
                    rule_required_fields = ["condition", "action", "description"]
                    for rule_field in rule_required_fields:
                        if rule_field not in rule:
                            issues.append(ValidationIssue(
                                field=f"rules[{i}].{rule_field}",
                                message=f"Rule field '{rule_field}' is missing",
                                severity=ValidationSeverity.HIGH
                            ))
        
        confidence_score = self._calculate_confidence_score(issues, len(rules) if isinstance(rules, list) else 0)
        is_valid = not self.has_blocking_issues(issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            confidence_score=confidence_score
        )


class LLMContentValidator:
    """Validator for LLM content quality and safety"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_content_safety(self, content: str) -> ValidationResult:
        """
        Validate content for safety issues
        
        Args:
            content: Text content to validate
            
        Returns:
            ValidationResult with safety validation details
        """
        issues = []
        
        # Check for potential PII
        pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "phone": r'\b\d{3}-\d{3}-\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
        
        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, content):
                issues.append(ValidationIssue(
                    field="content",
                    message=f"Potential {pii_type} detected in content",
                    severity=ValidationSeverity.HIGH
                ))
        
        # Check for inappropriate content (basic)
        inappropriate_keywords = ["password", "secret", "key", "token"]
        for keyword in inappropriate_keywords:
            if keyword.lower() in content.lower():
                issues.append(ValidationIssue(
                    field="content",
                    message=f"Potentially sensitive keyword '{keyword}' detected",
                    severity=ValidationSeverity.MEDIUM
                ))
        
        confidence_score = 1.0 - (len(issues) * 0.2)  # Simple scoring
        is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            confidence_score=max(0.0, confidence_score)
        )
    
    def validate_content_quality(self, content: str) -> ValidationResult:
        """
        Validate content quality
        
        Args:
            content: Text content to validate
            
        Returns:
            ValidationResult with quality validation details
        """
        issues = []
        
        # Check minimum length
        if len(content.strip()) < 10:
            issues.append(ValidationIssue(
                field="content",
                message="Content is too short (minimum 10 characters)",
                severity=ValidationSeverity.MEDIUM
            ))
        
        # Check for proper structure
        if not any(char in content for char in '.!?'):
            issues.append(ValidationIssue(
                field="content",
                message="Content lacks proper punctuation",
                severity=ValidationSeverity.LOW
            ))
        
        # Check for repeated content
        words = content.lower().split()
        if len(set(words)) < len(words) * 0.5:  # More than 50% repeated words
            issues.append(ValidationIssue(
                field="content",
                message="Content has excessive repetition",
                severity=ValidationSeverity.MEDIUM
            ))
        
        confidence_score = 1.0 - (len(issues) * 0.15)
        is_valid = not any(issue.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL] for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            confidence_score=max(0.0, confidence_score)
        )


def validate_llm_response(response: Dict[str, Any], response_type: str = "schema", 
                         strict_mode: bool = True, auto_correct: bool = False) -> ValidationResult:
    """
    Main function to validate LLM responses
    
    Args:
        response: LLM response to validate
        response_type: Type of response ("schema", "rule", "content")
        strict_mode: Whether to use strict validation
        auto_correct: Whether to attempt auto-correction
        
    Returns:
        ValidationResult with validation details
    """
    validator = LLMResponseValidator(strict_mode=strict_mode, auto_correct=auto_correct)
    
    if response_type == "schema":
        return validator.validate_schema_response(response)
    elif response_type == "rule":
        return validator.validate_rule_response(response)
    else:
        # Default to schema validation
        return validator.validate_schema_response(response)


def validate_content(content: str, check_safety: bool = True, check_quality: bool = True) -> ValidationResult:
    """
    Validate text content for safety and quality
    
    Args:
        content: Text content to validate
        check_safety: Whether to check for safety issues
        check_quality: Whether to check for quality issues
        
    Returns:
        Combined ValidationResult
    """
    content_validator = LLMContentValidator()
    all_issues = []
    
    if check_safety:
        safety_result = content_validator.validate_content_safety(content)
        all_issues.extend(safety_result.issues)
    
    if check_quality:
        quality_result = content_validator.validate_content_quality(content)
        all_issues.extend(quality_result.issues)
    
    # Calculate overall confidence
    confidence_score = 1.0 - (len(all_issues) * 0.1)
    is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in all_issues)
    
    return ValidationResult(
        is_valid=is_valid,
        issues=all_issues,
        confidence_score=max(0.0, confidence_score)
    )