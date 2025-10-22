"""
Policy-specific validation extensions for the LLM validation system.
Customized for data governance, schema design, and policy rule contexts.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import logging
from app.validation.llm_validator import (
    ValidationIssue, ValidationSeverity, ValidationResult,
    LLMContentValidator, InputSanitizer
)

# Constant pattern for valid identifiers (used for column names, table names, and rule names)
VALID_IDENTIFIER_PATTERN = r'^[a-zA-Z][a-zA-Z0-9_]*$'

logger = logging.getLogger(__name__)


@dataclass
class PolicyValidationConfig:
    """Configuration for policy domain validation"""
    
    # Allow business domain terms that might otherwise be flagged
    business_term_allowlist: List[str]
    
    # Schema-specific validation patterns
    schema_patterns: Dict[str, str]
    
    # Policy rule validation patterns  
    rule_patterns: Dict[str, str]
    
    # Domain-specific severity overrides
    severity_overrides: Dict[str, ValidationSeverity]


class PolicyContentValidator(LLMContentValidator):
    """Enhanced content validator for policy and schema contexts"""
    
    def __init__(self, enable_advanced_safety: bool = True, config: Optional[PolicyValidationConfig] = None):
        super().__init__(enable_advanced_safety)
        self.config = config or self._get_default_policy_config()
        
        # Override safety patterns with policy-aware versions
        self.safety_patterns.update({
            # Schema validation patterns
            "schema_tampering": r'(?i)\b(corrupt|destroy|manipulate)\s+(schema|metadata|structure)\b',
            "data_exfiltration": r'(?i)\b(extract|export|steal|copy)\s+(all|entire|complete)\s+(database|schema|data)\b',
            
            # Policy rule patterns  
            "policy_bypass": r'(?i)\b(bypass|circumvent|ignore|disable)\s+(all|every|entire)\s+(policy|rule|validation)\b',
            "rule_manipulation": r'(?i)\b(modify|alter|change)\s+(validation|rule|policy)\s+(to\s+allow|for\s+malicious)\b',
            
            # More specific harmful content patterns
            "malware_creation": r'(?i)\b(instructions?|tutorial|guide|how\s+to).*\b(creat|mak|build).*(malware|virus|trojan|backdoor)\b',
            "data_theft": r'(?i)\b(steal|theft).*(customer\s+data|personal\s+data|database)\b',
            "system_destruction": r'(?i)\b(destroy|damage).*(systems?|servers?)\b',
            
            # Business context patterns (more permissive)
            "business_harmful": r'(?i)\b(deliberately\s+harm|intentionally\s+damage|maliciously\s+affect)\s+(business|customers|users)\b',
        })
        
        # Add business term patterns that should be allowed
        self.business_allowlist_patterns = [
            r'(?i)\b(customer|user|client)\s+(data|information|details)\b',  # Legitimate business data terms
            r'(?i)\b(policy|rule|validation)\s+(create|update|modify|design)\b',  # Policy management terms
            r'(?i)\b(schema|column|table|field)\s+(design|structure|format)\b',  # Schema design terms
            r'(?i)\b(data\s+quality|data\s+governance|compliance|regulation)\b',  # Governance terms
        ]
    
    def _get_default_policy_config(self) -> PolicyValidationConfig:
        """Get default configuration for policy domain"""
        return PolicyValidationConfig(
            business_term_allowlist=[
                "customer_data", "user_information", "client_details",
                "policy_rule", "validation_rule", "compliance_check",
                "schema_design", "column_structure", "data_quality",
                "governance_framework", "regulatory_compliance"
            ],
            schema_patterns={
                "valid_column_name": VALID_IDENTIFIER_PATTERN,
                "valid_table_name": VALID_IDENTIFIER_PATTERN,
                "valid_domain_name": r'^[a-z0-9_-]+$'
            },
            rule_patterns={
                "gx_rule_format": r'expect_[a-z_]+',
                "valid_rule_name": VALID_IDENTIFIER_PATTERN
            },
            severity_overrides={
                "pii_email": ValidationSeverity.LOW,  # Often legitimate in business context
                "business_harmful": ValidationSeverity.MEDIUM,  # Context-dependent
            }
        )
    
    def validate_content_safety(self, content: str) -> ValidationResult:
        """Enhanced content safety validation with policy context awareness"""
        
        # First run the base validation
        base_result = super().validate_content_safety(content)
        
        # Apply policy-specific filtering
        filtered_issues = []
        
        for issue in base_result.issues:
            # Check if this issue should be allowed in business context
            if self._is_business_context_allowed(content, issue):
                # Downgrade severity or skip
                if issue.severity == ValidationSeverity.CRITICAL:
                    issue.severity = ValidationSeverity.HIGH
                elif issue.severity == ValidationSeverity.HIGH:
                    issue.severity = ValidationSeverity.MEDIUM
                # Still add it but with lower severity
                filtered_issues.append(issue)
            else:
                filtered_issues.append(issue)
        
        # Add policy-specific validations
        policy_issues = self._validate_policy_specific_patterns(content)
        filtered_issues.extend(policy_issues)
        
        # Recalculate safety score with policy context
        safety_score = self._calculate_policy_safety_score(content, filtered_issues)
        
        # Determine validity with policy context
        has_critical = any(issue.severity == ValidationSeverity.CRITICAL for issue in filtered_issues)
        has_high = any(issue.severity == ValidationSeverity.HIGH for issue in filtered_issues)
        
        is_valid = not has_critical and (not has_high or safety_score > 0.3)  # More lenient threshold
        
        return ValidationResult(
            is_valid=is_valid,
            issues=filtered_issues,
            confidence_score=safety_score
        )
    
    def _is_business_context_allowed(self, content: str, issue: ValidationIssue) -> bool:
        """Check if an issue should be allowed in business context"""
        
        # Check against business allowlist patterns
        for pattern in self.business_allowlist_patterns:
            if re.search(pattern, content):
                # If content matches business patterns, be more permissive
                if "pii" in issue.message.lower() or "email" in issue.message.lower():
                    return True
                if "business" in issue.message.lower():
                    return True
        
        # Check specific business terms
        content_lower = content.lower()
        for term in self.config.business_term_allowlist:
            if term.lower() in content_lower:
                return True
        
        return False
    
    def _validate_policy_specific_patterns(self, content: str) -> List[ValidationIssue]:
        """Validate policy-specific patterns"""
        issues = []
        
        # Check for schema naming conventions
        if self._contains_schema_references(content):
            # Validate schema naming patterns
            for name_type, pattern in self.config.schema_patterns.items():
                violations = self._find_pattern_violations(content, pattern, name_type)
                for violation in violations:
                    issues.append(ValidationIssue(
                        field="content",
                        message=f"Invalid {name_type}: {violation}",
                        severity=ValidationSeverity.MEDIUM
                    ))
        
        # Check for policy rule patterns
        if self._contains_rule_references(content):
            # Validate rule naming patterns
            for rule_type, pattern in self.config.rule_patterns.items():
                violations = self._find_pattern_violations(content, pattern, rule_type)
                for violation in violations:
                    issues.append(ValidationIssue(
                        field="content",
                        message=f"Invalid {rule_type}: {violation}",
                        severity=ValidationSeverity.LOW
                    ))
        
        return issues
    
    def _get_severity_for_pattern_type(self, pattern_type: str) -> ValidationSeverity:
        """Get appropriate severity level for different pattern types (policy-aware)"""
        critical_patterns = ["credentials", "sql_injection", "script_injection", "command_injection", 
                           "malware_creation", "data_theft", "system_destruction"]
        high_patterns = ["pii_ssn", "pii_credit_card", "file_inclusion", "harmful_content",
                        "schema_tampering", "data_exfiltration", "policy_bypass", "rule_manipulation"]
        medium_patterns = ["pii_email", "pii_phone", "business_harmful"]
        
        if pattern_type in critical_patterns:
            return ValidationSeverity.CRITICAL
        elif pattern_type in high_patterns:
            return ValidationSeverity.HIGH
        elif pattern_type in medium_patterns:
            return ValidationSeverity.MEDIUM
        else:
            return ValidationSeverity.LOW
    
    def _contains_schema_references(self, content: str) -> bool:
        """Check if content contains schema-related references"""
        schema_keywords = ["column", "table", "schema", "field", "domain"]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in schema_keywords)
    
    def _contains_rule_references(self, content: str) -> bool:
        """Check if content contains rule-related references"""
        rule_keywords = ["rule", "policy", "validation", "expect_", "gx_"]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in rule_keywords)
    
    def _find_pattern_violations(self, content: str, pattern: str, name_type: str) -> List[str]:
        """Find violations of naming patterns"""
        # This is a simplified implementation - you could make it more sophisticated
        violations = []
        
        # Extract potential names based on context
        words = re.findall(r'\b\w+\b', content)
        for word in words:
            if len(word) > 2 and not re.match(pattern, word):
                # Check if this word is in a context that suggests it should follow the pattern
                if name_type in ["column_name", "table_name"] and ("column" in content.lower() or "table" in content.lower()):
                    violations.append(word)
        
        return violations[:3]  # Limit to first 3 violations
    
    def _calculate_policy_safety_score(self, content: str, issues: List[ValidationIssue]) -> float:
        """Calculate safety score with policy context"""
        base_score = super()._calculate_safety_score(content, issues)
        
        # Boost score for business context
        business_context_bonus = 0.0
        content_lower = content.lower()
        
        # Give bonus for legitimate business terms
        business_terms = ["customer", "policy", "schema", "validation", "rule", "data quality", "governance"]
        for term in business_terms:
            if term in content_lower:
                business_context_bonus += 0.1
        
        # Cap the bonus
        business_context_bonus = min(business_context_bonus, 0.3)
        
        return min(1.0, base_score + business_context_bonus)


class PolicyInputSanitizer(InputSanitizer):
    """Enhanced input sanitizer for policy contexts"""
    
    def __init__(self, max_length: int = 10000, config: Optional[PolicyValidationConfig] = None):
        super().__init__(max_length)
        self.config = config or PolicyValidationConfig(
            business_term_allowlist=[],
            schema_patterns={},
            rule_patterns={},
            severity_overrides={}
        )
        
        # Override blocked patterns to be more permissive for business context
        self.BLOCKED_PATTERNS = [
            r'(?i)\b(hack|exploit|vulnerability|injection)\s+(attempt|attack|payload)\b',  # More specific
            r'(?i)\b(delete|drop|truncate|destroy)\s+from\s+\w+\s+(where\s+1=1|;)',  # More specific SQL injection
            r'(?i)\b(password|secret|token|key)\s*[:=]\s*["\']?\w{12,}',  # Longer credential patterns
            r'(?i)\b(bypass|circumvent|disable)\s+(all|entire|complete)\s+(security|validation|auth)',  # More specific
            r'(?i)(?:execute|run|eval|system)\s*\([^)]*(?:rm|del|format)',  # More specific command injection
            r'(?i)\b(virus|malware|trojan|backdoor|ransomware)\b',
            r'(?i)rm\s+-rf\s+/',  # Dangerous file system commands
        ]
    
    def sanitize_input(self, user_input: str) -> Tuple[str, List[ValidationIssue]]:
        """Enhanced sanitization with policy context awareness"""
        
        # Check for business context
        is_business_context = self._is_business_context(user_input)
        
        issues = []
        
        # Check input length
        if len(user_input) > self.max_length:
            issues.append(ValidationIssue(
                field="input",
                message=f"Input length exceeds limit ({len(user_input)} > {self.max_length})",
                severity=ValidationSeverity.CRITICAL
            ))
            return "", issues
        
        # Check for blocked patterns - but be more lenient in business context
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, user_input):
                # In business context, some patterns might be legitimate
                if is_business_context and self._is_legitimate_business_use(user_input, pattern):
                    issues.append(ValidationIssue(
                        field="input",
                        message="Potentially risky pattern detected in business context",
                        severity=ValidationSeverity.MEDIUM
                    ))
                else:
                    issues.append(ValidationIssue(
                        field="input",
                        message="Blocked security pattern detected",
                        severity=ValidationSeverity.CRITICAL
                    ))
                    return "", issues
        
        # Check for warning patterns
        for pattern in self.WARNING_PATTERNS:
            if re.search(pattern, user_input):
                issues.append(ValidationIssue(
                    field="input", 
                    message="Potentially risky pattern detected",
                    severity=ValidationSeverity.HIGH if not is_business_context else ValidationSeverity.MEDIUM
                ))
        
        # Basic sanitization
        sanitized = self._clean_input(user_input)
        
        return sanitized, issues
    
    def _is_business_context(self, text: str) -> bool:
        """Determine if the input is in a business/policy context"""
        business_indicators = [
            "schema", "column", "table", "domain", "policy", "rule", "validation",
            "customer", "data quality", "governance", "compliance", "business logic"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in business_indicators)
    
    def _is_legitimate_business_use(self, text: str, pattern: str) -> bool:
        """Check if a potentially dangerous pattern is legitimate in business context"""
        
        # For example, if someone is discussing SQL operations in a schema design context
        if "delete" in pattern.lower() or "drop" in pattern.lower():
            business_sql_context = [
                "schema design", "table design", "data modeling", "policy definition",
                "rule specification", "validation logic"
            ]
            text_lower = text.lower()
            return any(context in text_lower for context in business_sql_context)
        
        return False


# Factory function to create policy-aware validators
def create_policy_validator(config: Optional[PolicyValidationConfig] = None) -> PolicyContentValidator:
    """Create a policy-aware content validator"""
    return PolicyContentValidator(enable_advanced_safety=True, config=config)


def create_policy_sanitizer(max_length: int = 10000, config: Optional[PolicyValidationConfig] = None) -> PolicyInputSanitizer:
    """Create a policy-aware input sanitizer"""
    return PolicyInputSanitizer(max_length=max_length, config=config)