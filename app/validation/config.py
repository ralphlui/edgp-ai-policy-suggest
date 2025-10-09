"""
Configuration for LLM Validation System

This module provides centralized configuration for validation rules, thresholds,
and validation behavior across different LLM use cases.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum


class ValidationProfile(Enum):
    """Predefined validation profiles for different use cases"""
    STRICT = "strict"
    STANDARD = "standard" 
    LENIENT = "lenient"
    DEVELOPMENT = "development"


@dataclass
class ValidationConfig:
    """Configuration for LLM validation behavior"""
    
    # General validation settings
    strict_mode: bool = True
    auto_correct: bool = False
    min_confidence_score: float = 0.7
    
    # Schema validation settings
    min_columns: int = 3
    max_columns: int = 15
    min_samples_per_column: int = 3
    max_samples_per_column: int = 10
    
    # Allowed data types
    allowed_data_types: Set[str] = None
    
    # Content validation settings
    min_content_length: int = 10
    max_content_length: int = 10000
    check_pii: bool = True
    check_sensitive_keywords: bool = True
    
    # Performance settings
    validation_timeout: float = 30.0
    enable_caching: bool = True
    
    def __post_init__(self):
        if self.allowed_data_types is None:
            self.allowed_data_types = {
                "string", "integer", "float", "date", "boolean", "text"
            }


def get_validation_config(profile: Optional[ValidationProfile] = None) -> ValidationConfig:
    """
    Get validation configuration for specified profile
    
    Args:
        profile: Validation profile to use. If None, determines from environment
        
    Returns:
        ValidationConfig instance
    """
    
    # Determine profile from environment if not specified
    if profile is None:
        env_profile = os.getenv("VALIDATION_PROFILE", "standard").lower()
        try:
            profile = ValidationProfile(env_profile)
        except ValueError:
            profile = ValidationProfile.STANDARD
    
    # Create base config
    if profile == ValidationProfile.STRICT:
        return ValidationConfig(
            strict_mode=True,
            auto_correct=False,
            min_confidence_score=0.9,
            min_columns=5,
            max_columns=12,
            min_samples_per_column=3,
            check_pii=True,
            check_sensitive_keywords=True,
            validation_timeout=60.0
        )
    
    elif profile == ValidationProfile.LENIENT:
        return ValidationConfig(
            strict_mode=False,
            auto_correct=True,
            min_confidence_score=0.5,
            min_columns=1,
            max_columns=20,
            min_samples_per_column=2,
            check_pii=False,
            check_sensitive_keywords=False,
            validation_timeout=15.0
        )
    
    elif profile == ValidationProfile.DEVELOPMENT:
        return ValidationConfig(
            strict_mode=False,
            auto_correct=True,
            min_confidence_score=0.3,
            min_columns=1,
            max_columns=25,
            min_samples_per_column=1,
            check_pii=False,
            check_sensitive_keywords=False,
            validation_timeout=10.0,
            enable_caching=False
        )
    
    else:  # STANDARD
        return ValidationConfig(
            strict_mode=True,
            auto_correct=True,
            min_confidence_score=0.7,
            min_columns=3,
            max_columns=15,
            min_samples_per_column=3,
            check_pii=True,
            check_sensitive_keywords=True,
            validation_timeout=30.0
        )


@dataclass 
class ValidationRules:
    """Custom validation rules for specific domains"""
    
    domain: str
    required_columns: List[str] = None
    forbidden_columns: List[str] = None
    column_patterns: Dict[str, str] = None  # column_name -> regex pattern
    custom_validators: Dict[str, callable] = None
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = []
        if self.forbidden_columns is None:
            self.forbidden_columns = []
        if self.column_patterns is None:
            self.column_patterns = {}
        if self.custom_validators is None:
            self.custom_validators = {}


# Domain-specific validation rules
DOMAIN_VALIDATION_RULES = {
    "customer": ValidationRules(
        domain="customer",
        required_columns=["customer_id", "name"],
        forbidden_columns=["password", "secret"],
        column_patterns={
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?1?-?\.?\s?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$"
        }
    ),
    
    "product": ValidationRules(
        domain="product",
        required_columns=["product_id", "name", "price"],
        forbidden_columns=["internal_cost", "margin"],
        column_patterns={
            "sku": r"^[A-Z0-9]{3,}-[A-Z0-9]{3,}$",
            "price": r"^\d+\.?\d{0,2}$"
        }
    ),
    
    "order": ValidationRules(
        domain="order", 
        required_columns=["order_id", "customer_id", "total"],
        forbidden_columns=["internal_notes"],
        column_patterns={
            "order_id": r"^ORD-\d{6,}$"
        }
    ),
    
    "transaction": ValidationRules(
        domain="transaction",
        required_columns=["transaction_id", "amount", "timestamp"],
        forbidden_columns=["account_number", "routing_number"],
        column_patterns={
            "transaction_id": r"^TXN-[A-Z0-9]{8,}$",
            "amount": r"^\d+\.?\d{0,2}$"
        }
    )
}


def get_domain_validation_rules(domain: str) -> Optional[ValidationRules]:
    """
    Get validation rules for a specific domain
    
    Args:
        domain: Domain name (e.g., 'customer', 'product')
        
    Returns:
        ValidationRules for the domain, or None if not found
    """
    return DOMAIN_VALIDATION_RULES.get(domain.lower())


def register_domain_validation_rules(domain: str, rules: ValidationRules) -> None:
    """
    Register custom validation rules for a domain
    
    Args:
        domain: Domain name
        rules: ValidationRules instance
    """
    DOMAIN_VALIDATION_RULES[domain.lower()] = rules


# Environment-based configuration
def load_validation_config_from_env() -> ValidationConfig:
    """Load validation configuration from environment variables"""
    
    config = get_validation_config()
    
    # Override with environment variables if present
    if os.getenv("VALIDATION_STRICT_MODE"):
        config.strict_mode = os.getenv("VALIDATION_STRICT_MODE", "").lower() == "true"
    
    if os.getenv("VALIDATION_AUTO_CORRECT"):
        config.auto_correct = os.getenv("VALIDATION_AUTO_CORRECT", "").lower() == "true"
    
    if os.getenv("VALIDATION_MIN_CONFIDENCE"):
        try:
            config.min_confidence_score = float(os.getenv("VALIDATION_MIN_CONFIDENCE"))
        except ValueError:
            pass
    
    if os.getenv("VALIDATION_MIN_COLUMNS"):
        try:
            config.min_columns = int(os.getenv("VALIDATION_MIN_COLUMNS"))
        except ValueError:
            pass
    
    if os.getenv("VALIDATION_CHECK_PII"):
        config.check_pii = os.getenv("VALIDATION_CHECK_PII", "").lower() == "true"
    
    return config


def load_validation_config(profile: Optional[ValidationProfile] = None) -> ValidationConfig:
    """
    Alias for get_validation_config for backward compatibility
    
    Args:
        profile: Validation profile to use
        
    Returns:
        ValidationConfig instance
    """
    return get_validation_config(profile)