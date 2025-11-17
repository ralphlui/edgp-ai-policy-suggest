"""
LLM Validation Middleware

This module provides middleware components for integrating LLM validation
into the agent workflow and API endpoints.
"""

import logging
import functools
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
import datetime as dt

from app.validation.llm_validator import (
    ComprehensiveLLMValidator, ValidationResult, ValidationIssue, ValidationSeverity
)
from app.validation.policy_validator import create_policy_validator, create_policy_sanitizer
from app.exception.exceptions import ValidationError

logger = logging.getLogger(__name__)


class LLMValidationMiddleware:
    """
    Middleware for LLM validation in agent workflows
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM Validation Middleware with policy-aware validators
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.validator = ComprehensiveLLMValidator(self.config)  # Pass config directly
        # Use policy-aware validators for better domain compatibility
        self.content_validator = create_policy_validator()
        self.input_sanitizer = create_policy_sanitizer()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Track validation metrics
        self.validation_metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "filtered_responses": 0,
            "last_reset": datetime.now(dt.timezone.utc)
        }
    
    def validate_input(self, user_input: str, user_id: str, 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate user input before sending to LLM
        
        Args:
            user_input: User's input to validate
            user_id: User identifier for rate limiting
            context: Additional context
            
        Returns:
            Dict containing validation result and sanitized input
            
        Raises:
            ValidationError: If input is invalid and cannot be processed
        """
        # Enhanced logging for complete observability
        self.logger.info(f" Starting input validation for user {user_id}")
        self.logger.debug(f" Input length: {len(user_input)} chars, context: {bool(context)}")
        
        self.validation_metrics["total_requests"] += 1
        
        try:
            # Log validation process start
            self.logger.info(f" Running comprehensive input validation")
            result = self.validator.validate_llm_request(user_input, user_id, context)
            
            # Log validation results with detailed metrics
            self.logger.info(f" Input validation completed - Valid: {result.is_valid}, "
                           f"Confidence: {result.confidence_score:.2f}, Issues: {len(result.issues)}")
            
            if not result.is_valid:
                self.validation_metrics["blocked_requests"] += 1
                
                # Enhanced logging for validation failures
                self.logger.warning(f" Input validation BLOCKED for user {user_id}: "
                                  f"{len(result.issues)} security issues detected")
                
                # Log specific issues for debugging
                for issue in result.issues:
                    self.logger.warning(f"    {issue.severity.value.upper()}: {issue.field} - {issue.message}")
                
                # Log validation failure
                self.logger.warning(
                    f"Input validation failed for user {user_id}: "
                    f"{len(result.issues)} issues found"
                )
                
                # Check if this is a critical failure that should block the request
                critical_issues = [issue for issue in result.issues if issue.severity == ValidationSeverity.CRITICAL]
                if critical_issues:
                    error_message = f"Input validation failed: {'; '.join(issue.message for issue in critical_issues)}"
                    raise ValidationError(error_message)
            
            # Return validation result with sanitized input
            return {
                "is_valid": result.is_valid,
                "sanitized_input": result.corrected_data.get("sanitized_input", user_input) if result.corrected_data else user_input,
                "confidence_score": result.confidence_score,
                "issues": [
                    {
                        "field": issue.field,
                        "message": issue.message,
                        "severity": issue.severity.value
                    } for issue in result.issues
                ],
                "validation_metadata": {
                    "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                    "user_id": user_id,
                    "input_length": len(user_input)
                }
            }
            
        except Exception as e:
            self.logger.error(f" Critical error during input validation for user {user_id}: {e}")
            self.logger.debug(f" Validation metrics - Total: {self.validation_metrics['total_requests']}, "
                            f"Blocked: {self.validation_metrics['blocked_requests']}")
            raise ValidationError(f"Validation failed: {str(e)}")
    
    def validate_output(self, llm_response: Union[str, Dict[str, Any]], 
                       response_type: str = "schema",
                       expected_schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate LLM output before returning to user
        
        Args:
            llm_response: LLM response to validate
            response_type: Type of response ("schema", "rule", "content")
            expected_schema: Expected schema for validation
            
        Returns:
            Dict containing validation result and filtered response
        """
        # Enhanced logging for output validation
        self.logger.info(f" Starting output validation for response_type: {response_type}")
        self.logger.debug(f" Response size: {len(str(llm_response))} chars, has_schema: {bool(expected_schema)}")
        
        try:
            # Log validation process start
            self.logger.info(f" Running comprehensive output validation")
            result = self.validator.validate_llm_response(
                llm_response, response_type, expected_schema
            )
            
            # Log validation results with detailed metrics
            self.logger.info(f" Output validation completed - Valid: {result.is_valid}, "
                           f"Confidence: {result.confidence_score:.2f}, Issues: {len(result.issues)}")
            
            if not result.is_valid:
                self.validation_metrics["filtered_responses"] += 1
                
                # Enhanced logging for output validation failures
                self.logger.warning(f" Output validation FILTERED: {len(result.issues)} quality issues detected")
                
                # Log specific output issues
                for issue in result.issues:
                    self.logger.warning(f"    {issue.severity.value.upper()}: {issue.field} - {issue.message}")
                
                self.logger.warning(
                    f"Output validation found issues: {len(result.issues)} issues"
                )
            else:
                # Log successful validation
                self.logger.info(f" Output validation PASSED - Quality score: {result.confidence_score:.2f}")
            
            return {
                "is_valid": result.is_valid,
                "filtered_response": llm_response,  # Could add filtering logic here
                "confidence_score": result.confidence_score,
                "issues": [
                    {
                        "field": issue.field,
                        "message": issue.message,
                        "severity": issue.severity.value
                    } for issue in result.issues
                ],
                "validation_metadata": {
                    "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                    "response_type": response_type,
                    "response_length": len(str(llm_response))
                }
            }
            
        except Exception as e:
            self.logger.error(f" Critical error during output validation: {e}")
            self.logger.debug(f" Validation metrics - Total: {self.validation_metrics['total_requests']}, "
                            f"Filtered: {self.validation_metrics['filtered_responses']}")
            return {
                "is_valid": False,
                "filtered_response": llm_response,
                "confidence_score": 0.0,
                "issues": [{"field": "validation", "message": str(e), "severity": "critical"}],
                "validation_metadata": {"error": str(e)}
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics"""
        return {
            **self.validation_metrics,
            "validator_stats": self.validator.get_validation_stats()
        }
    
    def reset_metrics(self):
        """Reset validation metrics"""
        self.validation_metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "filtered_responses": 0,
            "last_reset": datetime.now(dt.timezone.utc)
        }


def llm_input_validator(config: Optional[Dict[str, Any]] = None):
    """
    Decorator for validating LLM inputs in functions
    
    Usage:
        @llm_input_validator()
        def my_llm_function(user_input: str, user_id: str, **kwargs):
            # Function will receive validated input
            pass
    """
    middleware = LLMValidationMiddleware(config)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user_input and user_id from arguments
            user_input = kwargs.get('user_input') or (args[0] if args else None)
            user_id = kwargs.get('user_id') or (args[1] if len(args) > 1 else 'unknown')
            
            if not user_input:
                raise ValueError("user_input is required for LLM validation")
            
            # Validate input
            validation_result = middleware.validate_input(user_input, user_id)
            
            if not validation_result["is_valid"]:
                # Optionally block or allow with warnings based on severity
                critical_issues = [
                    issue for issue in validation_result["issues"] 
                    if issue["severity"] == "critical"
                ]
                if critical_issues:
                    raise ValidationError("Input validation failed with critical issues")
            
            # Replace user_input with sanitized version
            if 'user_input' in kwargs:
                kwargs['user_input'] = validation_result["sanitized_input"]
            elif args:
                args = list(args)
                args[0] = validation_result["sanitized_input"]
                args = tuple(args)
            
            # Add validation info to kwargs
            kwargs['_validation_info'] = validation_result
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def llm_output_validator(response_type: str = "schema", 
                        expected_schema: Optional[Dict] = None,
                        config: Optional[Dict[str, Any]] = None):
    """
    Decorator for validating LLM outputs from functions
    
    Usage:
        @llm_output_validator("schema")
        def my_llm_function() -> dict:
            # Function return value will be validated
            return llm_response
    """
    middleware = LLMValidationMiddleware(config)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Validate output
            validation_result = middleware.validate_output(
                result, response_type, expected_schema
            )
            
            # Add validation metadata to result if it's a dict
            if isinstance(result, dict):
                result['_validation'] = validation_result
            
            return result
        
        return wrapper
    return decorator


class AgentValidationContext:
    """
    Context manager for agent workflow validation
    
    Usage:
        with AgentValidationContext(user_id="user123") as validator:
            sanitized_input = validator.validate_input(user_input)
            # ... process with LLM ...
            validated_output = validator.validate_output(llm_response)
    """
    
    def __init__(self, user_id: str, config: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.middleware = LLMValidationMiddleware(config)
        self.validation_log = []
        self.session_start = datetime.now(dt.timezone.utc)
        self.logger = logging.getLogger(f"{__name__}.AgentValidationContext")
    
    def __enter__(self):
        # Enhanced context entry logging
        self.logger.info(f" Starting agent validation session for user {self.user_id}")
        self.logger.debug(f" Session started at: {self.session_start.isoformat()}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Enhanced context exit logging with session summary
        session_duration = (datetime.now(dt.timezone.utc) - self.session_start).total_seconds()
        
        if exc_type:
            self.logger.error(f" Validation session ended with exception: {exc_type.__name__}: {exc_val}")
        
        # Comprehensive session summary
        total_validations = len(self.validation_log)
        successful_validations = len([v for v in self.validation_log if v["valid"]])
        input_validations = len([v for v in self.validation_log if v["type"] == "input"])
        output_validations = len([v for v in self.validation_log if v["type"] == "output"])
        
        self.logger.info(f" VALIDATION SESSION SUMMARY for user {self.user_id}:")
        self.logger.info(f"    Duration: {session_duration:.2f}s")
        self.logger.info(f"    Total validations: {total_validations}")
        self.logger.info(f"    Successful: {successful_validations}/{total_validations}")
        self.logger.info(f"    Input validations: {input_validations}")
        self.logger.info(f"    Output validations: {output_validations}")
        
        if total_validations > 0:
            success_rate = (successful_validations / total_validations) * 100
            self.logger.info(f"    Success rate: {success_rate:.1f}%")
    
    def validate_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Validate input and return sanitized version"""
        # Enhanced input validation logging
        self.logger.info(f" [CONTEXT] Validating input for user {self.user_id}")
        self.logger.debug(f" Input preview: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        
        result = self.middleware.validate_input(user_input, self.user_id, context)
        
        # Log validation result in context
        validation_entry = {
            "type": "input",
            "valid": result["is_valid"],
            "confidence": result["confidence_score"],
            "issues_count": len(result["issues"]),
            "timestamp": datetime.now(dt.timezone.utc).isoformat()
        }
        self.validation_log.append(validation_entry)
        
        self.logger.info(f" [CONTEXT] Input validation result: valid={result['is_valid']}, "
                        f"confidence={result['confidence_score']:.2f}, issues={len(result['issues'])}")
        
        if not result["is_valid"]:
            critical_issues = [
                issue for issue in result["issues"] 
                if issue["severity"] == "critical"
            ]
            if critical_issues:
                self.logger.error(f" [CONTEXT] Critical validation failure - blocking request")
                raise ValidationError("Input validation failed")
            else:
                self.logger.warning(f" [CONTEXT] Non-critical issues found but allowing request to proceed")
        
        return result["sanitized_input"]
    
    def validate_output(self, llm_response: Union[str, Dict[str, Any]], 
                       response_type: str = "schema") -> Union[str, Dict[str, Any]]:
        """Validate output and return filtered version"""
        # Enhanced output validation logging
        self.logger.info(f" [CONTEXT] Validating output for user {self.user_id}, type: {response_type}")
        self.logger.debug(f" Output size: {len(str(llm_response))} chars")
        
        result = self.middleware.validate_output(llm_response, response_type)
        
        # Log validation result in context
        validation_entry = {
            "type": "output",
            "valid": result["is_valid"],
            "confidence": result["confidence_score"],
            "issues_count": len(result["issues"]),
            "response_type": response_type,
            "timestamp": datetime.now(dt.timezone.utc).isoformat()
        }
        self.validation_log.append(validation_entry)
        
        self.logger.info(f" [CONTEXT] Output validation result: valid={result['is_valid']}, "
                        f"confidence={result['confidence_score']:.2f}, issues={len(result['issues'])}")
        
        if not result["is_valid"]:
            self.logger.warning(f" [CONTEXT] Output quality issues detected but proceeding with response")
        
        return result["filtered_response"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics for this context"""
        return {
            "user_id": self.user_id,
            "validations_performed": len(self.validation_log),
            "input_validations": len([v for v in self.validation_log if v["type"] == "input"]),
            "output_validations": len([v for v in self.validation_log if v["type"] == "output"]),
            "failed_validations": len([v for v in self.validation_log if not v["valid"]])
        }


# Global middleware instance for easy access
_global_middleware = None

def get_global_middleware(config: Optional[Dict[str, Any]] = None) -> LLMValidationMiddleware:
    """Get or create global middleware instance"""
    global _global_middleware
    if _global_middleware is None:
        _global_middleware = LLMValidationMiddleware(config)
    return _global_middleware


def validate_input_quick(user_input: str, user_id: str) -> str:
    """Quick input validation function"""
    middleware = get_global_middleware()
    result = middleware.validate_input(user_input, user_id)
    if not result["is_valid"]:
        critical_issues = [issue for issue in result["issues"] if issue["severity"] == "critical"]
        if critical_issues:
            raise ValidationError("Input validation failed")
    return result["sanitized_input"]


def validate_output_quick(llm_response: Union[str, Dict[str, Any]], 
                         response_type: str = "schema") -> Union[str, Dict[str, Any]]:
    """Quick output validation function"""
    middleware = get_global_middleware()
    result = middleware.validate_output(llm_response, response_type)
    return result["filtered_response"]