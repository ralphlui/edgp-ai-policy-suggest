"""
Comprehensive LLM Response Validation Module

This module provides validation for LLM responses across different use cases in the application.
Includes schema validation, content validation, safety checks, rate limiting, and input sanitization.
"""

import re
import json
import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import datetime as dt
from collections import defaultdict, deque
from enum import Enum

from pydantic import BaseModel, Field, ValidationError, field_validator
from app.exception.exceptions import SchemaGenerationError, ValidationError as AppValidationError

# Import base definitions
from app.validation.validation_base import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult
)

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for LLM validation"""
    SAFE = "safe"
    WARNING = "warning" 
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


class RateLimitManager:
    """Manages rate limiting for LLM requests"""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._minute_tracker = defaultdict(deque)
        self._hour_tracker = defaultdict(deque)
        
    def check_rate_limit(self, user_id: str) -> Tuple[bool, Dict[str, int]]:
        """
        Check if user is within rate limits
        
        Returns:
            Tuple of (is_allowed, remaining_counts)
        """
        now = datetime.now(dt.timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        minute_queue = self._minute_tracker[user_id]
        while minute_queue and minute_queue[0] <= minute_ago:
            minute_queue.popleft()
            
        hour_queue = self._hour_tracker[user_id]
        while hour_queue and hour_queue[0] <= hour_ago:
            hour_queue.popleft()
        
        # Check limits
        minute_count = len(minute_queue)
        hour_count = len(hour_queue)
        
        is_allowed = (minute_count < self.requests_per_minute and 
                     hour_count < self.requests_per_hour)
        
        if is_allowed:
            minute_queue.append(now)
            hour_queue.append(now)
            # Update counts after adding current request
            minute_count += 1
            hour_count += 1
        
        return is_allowed, {
            "minute_remaining": max(0, self.requests_per_minute - minute_count),
            "hour_remaining": max(0, self.requests_per_hour - hour_count)
        }


class InputSanitizer:
    """Sanitizes user inputs before sending to LLM"""
    
    # Dangerous patterns that should be blocked - customized for policy/schema domain
    BLOCKED_PATTERNS = [
        r'(?i)\b(hack|exploit|vulnerability|injection|malicious)\b',
        r'(?i)\b(delete|drop|truncate|destroy)\s+from\s+\w+',  # SQL delete/drop patterns
        r'(?i)\b(password|secret|token|key)\s*[:=]\s*["\']?\w+',  # Credentials
        r'(?i)\b(admin|root|sudo|elevated)\s+(access|privileges|rights)\b',
        r'(?i)\b(bypass|circumvent|disable)\s+(security|validation|auth)',
        r'(?i)(?:execute|run|eval|system)\s*\([^)]*',  # Command execution
        r'(?i)\b(virus|malware|trojan|backdoor)\b',
        r'(?i)rm\s+-rf\s+/',  # Dangerous file system commands
        # Policy domain specific patterns - be more permissive for legitimate business terms
        r'(?i)\b(illegal|harmful|dangerous|malicious)\s+(intent|purpose|use)\b',  # Only block when used with harmful intent
    ]
    
    # Warning patterns - potentially risky but not blocked (for policy/schema domain)
    WARNING_PATTERNS = [
        r'(?i)\b(modify|alter|update)\s+(critical|important|system)\b',
        r'(?i)\b(external|third-party|untrusted)\s+(source|data|input)\b',
        r'(?i)\b(temporary|quick|skip)\s+(validation|check|verification)\b',
        r'(?i)\b(override|ignore)\s+(policy|rule|constraint)\b',  # Policy domain specific
        r'(?i)\b(disable|turn.off)\s+(validation|rule|policy)\b',  # Schema validation concerns
    ]
    
    def __init__(self, max_length: int = 10000):
        self.max_length = max_length
        
    def sanitize_input(self, user_input: str) -> Tuple[str, List[ValidationIssue]]:
        """
        Sanitize user input and return cleaned version with issues found
        
        Returns:
            Tuple of (sanitized_input, validation_issues)
        """
        issues = []
        
        # Check input length
        if len(user_input) > self.max_length:
            issues.append(ValidationIssue(
                field="input",
                message=f"Input length exceeds limit ({len(user_input)} > {self.max_length})",
                severity=ValidationSeverity.CRITICAL
            ))
            return "", issues
        
        # Check for blocked patterns
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, user_input):
                issues.append(ValidationIssue(
                    field="input",
                    message=f"Blocked security pattern detected",
                    severity=ValidationSeverity.CRITICAL
                ))
                return "", issues
        
        # Check for warning patterns
        for pattern in self.WARNING_PATTERNS:
            if re.search(pattern, user_input):
                issues.append(ValidationIssue(
                    field="input", 
                    message="Potentially risky pattern detected",
                    severity=ValidationSeverity.HIGH
                ))
        
        # Basic sanitization
        sanitized = self._clean_input(user_input)
        
        return sanitized, issues
    
    def _clean_input(self, text: str) -> str:
        """Clean input of potentially harmful content"""
        # Remove HTML/XML tags
        cleaned = re.sub(r'<[^>]+>', '', text)
        
        # Escape SQL injection attempts
        cleaned = re.sub(r'([\'";])', r'\\\1', cleaned)
        
        # Remove potential script injections
        cleaned = re.sub(r'(?i)<script[^>]*>.*?</script>', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned


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
    """Enhanced validator for LLM content quality and safety"""
    
    def __init__(self, enable_advanced_safety: bool = True):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enable_advanced_safety = enable_advanced_safety
        
        # Enhanced safety patterns - customized for policy/schema domain
        self.safety_patterns = {
            "pii_email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "pii_ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "pii_phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            "pii_credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "credentials": r'(?i)(password|secret|key|token|api_key)\s*[:=]\s*["\']?\w+',
            "sql_injection": r'(?i)\b(select|insert|update|delete|drop|union|exec|execute)\b.*\b(from|where|union|join|into)\b',
            "script_injection": r'(?i)<script[^>]*>.*?</script>',
            "command_injection": r'(?i)\b(?:system|exec|eval|shell_exec|passthru)\s*\(',
            "file_inclusion": r'(?i)(\.\.\/|\.\.\\|\/etc\/|\/proc\/|file:\/\/)',
            # More permissive harmful content detection for business domain
            "harmful_content": r'(?i)\b(?:hate\s+speech|violence\s+against|harm\s+individuals|suicide\s+instructions|illegal\s+drugs|weapon\s+making)\b'
        }
        
    def validate_content_safety(self, content: str) -> ValidationResult:
        """
        Enhanced content safety validation
        
        Args:
            content: Text content to validate
            
        Returns:
            ValidationResult with comprehensive safety validation
        """
        issues = []
        
        # Check each safety pattern
        for pattern_type, pattern in self.safety_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                severity = self._get_severity_for_pattern_type(pattern_type)
                issues.append(ValidationIssue(
                    field="content",
                    message=f"Potential {pattern_type.replace('_', ' ')} detected: {len(matches)} instances",
                    severity=severity,
                    raw_value=matches[:3] if len(matches) <= 3 else matches[:3] + ["..."]  # Limit to first 3
                ))
        
        # Check content length and quality
        if len(content.strip()) == 0:
            issues.append(ValidationIssue(
                field="content",
                message="Content is empty",
                severity=ValidationSeverity.CRITICAL
            ))
        elif len(content.strip()) < 5:
            issues.append(ValidationIssue(
                field="content", 
                message="Content is too short",
                severity=ValidationSeverity.MEDIUM
            ))
        
        # Check for excessive repetition
        words = content.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                issues.append(ValidationIssue(
                    field="content",
                    message=f"Content has excessive repetition (unique ratio: {unique_ratio:.2f})",
                    severity=ValidationSeverity.MEDIUM
                ))
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(content, issues)
        
        # Determine if content is safe
        has_critical = any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        has_high = any(issue.severity == ValidationSeverity.HIGH for issue in issues)
        
        is_valid = not has_critical and (not has_high or safety_score > 0.5)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            confidence_score=safety_score
        )
    
    def _get_severity_for_pattern_type(self, pattern_type: str) -> ValidationSeverity:
        """Get appropriate severity level for different pattern types"""
        critical_patterns = ["credentials", "sql_injection", "script_injection", "command_injection"]
        high_patterns = ["pii_ssn", "pii_credit_card", "file_inclusion", "harmful_content"]
        medium_patterns = ["pii_email", "pii_phone"]
        
        if pattern_type in critical_patterns:
            return ValidationSeverity.CRITICAL
        elif pattern_type in high_patterns:
            return ValidationSeverity.HIGH
        elif pattern_type in medium_patterns:
            return ValidationSeverity.MEDIUM
        else:
            return ValidationSeverity.LOW
    
    def _calculate_safety_score(self, content: str, issues: List[ValidationIssue]) -> float:
        """Calculate overall safety score (0-1, where 1 is safest)"""
        if not issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.LOW: 0.1,
            ValidationSeverity.MEDIUM: 0.3,
            ValidationSeverity.HIGH: 0.6,
            ValidationSeverity.CRITICAL: 1.0
        }
        
        total_penalty = sum(severity_weights[issue.severity] for issue in issues)
        max_penalty = len(issues) * 1.0  # If all were critical
        
        # Base safety score
        safety_score = max(0.0, 1.0 - (total_penalty / max(max_penalty, 1.0)))
        
        # Adjust for content length (longer content generally safer if it passes basic checks)
        content_length_factor = min(1.0, len(content) / 1000)  # Normalize around 1000 chars
        safety_score = safety_score * (0.8 + 0.2 * content_length_factor)
        
        return round(safety_score, 3)
    
    def filter_unsafe_content(self, content: str) -> Tuple[str, List[str]]:
        """
        Filter out unsafe content and return cleaned version
        
        Returns:
            Tuple of (filtered_content, list_of_removed_patterns)
        """
        filtered_content = content
        removed_patterns = []
        
        # Filter out potential credentials
        credential_pattern = r'(?i)(password|secret|key|token|api_key)\s*[:=]\s*["\']?[\w\-]{8,}'
        if re.search(credential_pattern, filtered_content):
            filtered_content = re.sub(credential_pattern, r'\1: [FILTERED]', filtered_content)
            removed_patterns.append("credentials")
        
        # Filter potential PII
        pii_patterns = {
            "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_FILTERED]'),
            "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_FILTERED]'),
            "credit_card": (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD_FILTERED]')
        }
        
        for pii_type, (pattern, replacement) in pii_patterns.items():
            if re.search(pattern, filtered_content):
                filtered_content = re.sub(pattern, replacement, filtered_content)
                removed_patterns.append(pii_type)
        
        # Remove script tags
        if re.search(r'(?i)<script[^>]*>.*?</script>', filtered_content):
            filtered_content = re.sub(r'(?i)<script[^>]*>.*?</script>', '[SCRIPT_REMOVED]', filtered_content)
            removed_patterns.append("script_tags")
        
        return filtered_content, removed_patterns
    
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


class ComprehensiveLLMValidator:
    """
    Main validator that combines all validation features including:
    - Input sanitization
    - Rate limiting  
    - Content safety validation
    - Output validation
    - Schema validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.sanitizer = InputSanitizer(max_length=config.get("max_input_length", 10000))
        self.rate_limiter = RateLimitManager(
            requests_per_minute=config.get("rate_limit_per_minute", 60),
            requests_per_hour=config.get("rate_limit_per_hour", 1000)
        )
        self.content_validator = LLMContentValidator(
            enable_advanced_safety=config.get("enable_advanced_safety", True)
        )
        self.response_validator = LLMResponseValidator(
            strict_mode=config.get("strict_mode", True),
            auto_correct=config.get("auto_correct", False)
        )
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def validate_llm_request(self, user_input: str, user_id: str, 
                           context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Comprehensive validation of LLM request before processing
        
        Args:
            user_input: User's input to validate
            user_id: Unique identifier for rate limiting
            context: Additional context for validation
            
        Returns:
            ValidationResult with comprehensive validation details
        """
        all_issues = []
        
        # 1. Check rate limits
        is_allowed, rate_info = self.rate_limiter.check_rate_limit(user_id)
        if not is_allowed:
            all_issues.append(ValidationIssue(
                field="rate_limit",
                message=f"Rate limit exceeded. Remaining: {rate_info['minute_remaining']}/min, {rate_info['hour_remaining']}/hour",
                severity=ValidationSeverity.CRITICAL
            ))
            return ValidationResult(
                is_valid=False,
                issues=all_issues,
                confidence_score=0.0
            )
        
        # 2. Sanitize input
        sanitized_input, sanitization_issues = self.sanitizer.sanitize_input(user_input)
        all_issues.extend(sanitization_issues)
        
        # If sanitization found critical issues, stop here
        if any(issue.severity == ValidationSeverity.CRITICAL for issue in sanitization_issues):
            return ValidationResult(
                is_valid=False,
                issues=all_issues,
                confidence_score=0.0
            )
        
        # 3. Validate content safety
        safety_result = self.content_validator.validate_content_safety(user_input)
        all_issues.extend(safety_result.issues)
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(all_issues, safety_result.confidence_score)
        
        # Determine if request is valid
        has_critical = any(issue.severity == ValidationSeverity.CRITICAL for issue in all_issues)
        has_high = any(issue.severity == ValidationSeverity.HIGH for issue in all_issues)
        
        is_valid = not has_critical and (not has_high or confidence_score > 0.6)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            corrected_data={"sanitized_input": sanitized_input} if is_valid else None,
            confidence_score=confidence_score
        )
    
    def validate_llm_response(self, response: Union[str, Dict[str, Any]], 
                            response_type: str = "schema", 
                            expected_schema: Optional[Dict] = None) -> ValidationResult:
        """
        Comprehensive validation of LLM response
        
        Args:
            response: LLM response to validate
            response_type: Type of response ("schema", "rule", "content")
            expected_schema: Expected schema for structured responses
            
        Returns:
            ValidationResult with comprehensive validation details  
        """
        all_issues = []
        
        # 1. Content safety validation (for string responses)
        if isinstance(response, str):
            safety_result = self.content_validator.validate_content_safety(response)
            all_issues.extend(safety_result.issues)
            
            # Filter unsafe content if needed
            if not safety_result.is_valid and safety_result.confidence_score > 0.3:
                filtered_content, removed_patterns = self.content_validator.filter_unsafe_content(response)
                if removed_patterns:
                    all_issues.append(ValidationIssue(
                        field="response_filtering",
                        message=f"Filtered unsafe content: {', '.join(removed_patterns)}",
                        severity=ValidationSeverity.MEDIUM
                    ))
        
        # 2. Schema/structure validation (for dict responses)
        if isinstance(response, dict):
            if response_type == "schema":
                schema_result = self.response_validator.validate_schema_response(response)
            elif response_type == "rule":
                schema_result = self.response_validator.validate_rule_response(response)
            else:
                # Generic validation
                schema_result = self.response_validator.validate_schema_response(response)
            
            all_issues.extend(schema_result.issues)
            
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(all_issues, 1.0)
        
        # Determine if response is valid
        has_critical = any(issue.severity == ValidationSeverity.CRITICAL for issue in all_issues)
        has_high = any(issue.severity == ValidationSeverity.HIGH for issue in all_issues)
        
        is_valid = not has_critical and (not has_high or confidence_score > 0.5)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            confidence_score=confidence_score
        )
    
    def _calculate_overall_confidence(self, issues: List[ValidationIssue], 
                                    base_confidence: float) -> float:
        """Calculate overall confidence score considering all issues"""
        if not issues:
            return base_confidence
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.LOW: 0.05,
            ValidationSeverity.MEDIUM: 0.15,
            ValidationSeverity.HIGH: 0.35,
            ValidationSeverity.CRITICAL: 0.8
        }
        
        total_penalty = sum(severity_weights[issue.severity] for issue in issues)
        confidence = base_confidence * max(0.0, 1.0 - total_penalty)
        
        return round(confidence, 3)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        return {
            "rate_limiter_stats": {
                "requests_per_minute_limit": self.rate_limiter.requests_per_minute,
                "requests_per_hour_limit": self.rate_limiter.requests_per_hour,
                "active_users": len(self.rate_limiter._minute_tracker)
            },
            "sanitizer_stats": {
                "max_input_length": self.sanitizer.max_length,
                "blocked_patterns_count": len(self.sanitizer.BLOCKED_PATTERNS),
                "warning_patterns_count": len(self.sanitizer.WARNING_PATTERNS)
            },
            "content_validator_stats": {
                "safety_patterns_count": len(self.content_validator.safety_patterns),
                "advanced_safety_enabled": self.content_validator.enable_advanced_safety
            },
            "response_validator_stats": {
                "strict_mode": self.response_validator.strict_mode,
                "auto_correct": self.response_validator.auto_correct
            }
        }


# Convenience functions for easy integration
def create_llm_validator(config: Optional[Dict[str, Any]] = None) -> ComprehensiveLLMValidator:
    """Factory function to create configured LLM validator"""
    return ComprehensiveLLMValidator(config)


def validate_user_input(user_input: str, user_id: str, 
                       config: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Quick validation for user input"""
    validator = create_llm_validator(config)
    return validator.validate_llm_request(user_input, user_id)


def validate_llm_output(response: Union[str, Dict[str, Any]], 
                       response_type: str = "schema",
                       config: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Quick validation for LLM output"""
    validator = create_llm_validator(config)
    return validator.validate_llm_response(response, response_type)


# Backward compatibility functions
def validate_llm_response(response: Dict[str, Any], response_type: str = "schema", 
                         strict_mode: bool = True, auto_correct: bool = False) -> ValidationResult:
    """
    Backward compatibility function for existing code
    
    Args:
        response: LLM response to validate
        response_type: Type of response ("schema", "rule", "content")
        strict_mode: Whether to use strict validation
        auto_correct: Whether to attempt auto-correction
        
    Returns:
        ValidationResult with validation details
    """
    config = {
        "strict_mode": strict_mode,
        "auto_correct": auto_correct,
        "enable_advanced_safety": True
    }
    validator = ComprehensiveLLMValidator(config)
    return validator.validate_llm_response(response, response_type)


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
    content_validator = LLMContentValidator(enable_advanced_safety=True)
    all_issues = []
    
    if check_safety:
        safety_result = content_validator.validate_content_safety(content)
        all_issues.extend(safety_result.issues)
    
    # Calculate overall confidence
    confidence_score = 1.0 - (len(all_issues) * 0.1)
    is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in all_issues)
    
    return ValidationResult(
        is_valid=is_valid,
        issues=all_issues,
        confidence_score=max(0.0, confidence_score)
    )