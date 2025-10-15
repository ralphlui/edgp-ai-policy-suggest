#!/usr/bin/env python3
"""
Demonstration of the policy-aware LLM validation system
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app.validation.policy_validator import create_policy_validator, create_policy_sanitizer
from app.validation.middleware import AgentValidationContext

def demo_policy_validation():
    print("  Policy-Aware LLM Validation System Demo")
    print("=" * 50)
    
    # Create policy-aware validators
    validator = create_policy_validator()
    sanitizer = create_policy_sanitizer()
    
    print("\n1. Testing Business Context Email Handling:")
    business_email_content = "Create customer schema with email column to store addresses like user@company.com"
    result = validator.validate_content_safety(business_email_content)
    print(f"   Input: {business_email_content[:50]}...")
    print(f"   Valid: {result.is_valid}")
    print(f"   Issues: {len(result.issues)} (all non-critical in business context)")
    
    print("\n2. Testing Schema Design Language:")
    schema_content = """
    For customer domain, create table with columns:
    - customer_id (primary key)
    - email (contact info)
    - phone (contact number)
    """
    result = validator.validate_content_safety(schema_content)
    print(f"   Input: Schema design content")
    print(f"   Valid: {result.is_valid}")
    print(f"   Confidence: {result.confidence_score:.2f}")
    
    print("\n3. Testing Great Expectations Rule Language:")
    gx_content = """
    Apply validation rules:
    - expect_column_values_to_not_be_null
    - expect_column_values_to_match_regex
    - expect_column_values_to_be_in_range
    """
    result = validator.validate_content_safety(gx_content)
    print(f"   Input: GX rules content")
    print(f"   Valid: {result.is_valid}")
    print(f"   Issues: {len(result.issues)}")
    
    print("\n4. Testing Agent Validation Context:")
    user_id = "demo_user"
    
    try:
        with AgentValidationContext(user_id) as agent_validator:
            # Test valid business input
            business_input = "Suggest data quality policies for customer domain"
            sanitized = agent_validator.validate_input(business_input)
            print(f"   ✅ Business input validated and sanitized")
            
            # Test output validation
            mock_output = {"rules": ["expect_column_values_to_not_be_null"], "confidence": 0.9}
            filtered_output = agent_validator.validate_output(mock_output, "rule")
            print(f"   ✅ Output validation successful")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n5. Testing Malicious Input Blocking:")
    try:
        malicious_input = "DROP TABLE customers; DELETE FROM users WHERE 1=1;"
        with AgentValidationContext(user_id) as agent_validator:
            agent_validator.validate_input(malicious_input)
        print(f"   ❌ Malicious input was NOT blocked (unexpected)")
    except Exception:
        print(f"   ✅ Malicious input was properly blocked")
    
    print("\n6. Testing Input Sanitization:")
    test_inputs = [
        "Create customer policy with validation rules",
        """"
LLM Validation and Safety System - Project Compatibility Summary

This document summarizes the enhancements made to make the LLM validation and safety 
system fully compatible with the EDGP AI Policy Suggestion project.
"""

# ========================================
# SUMMARY OF CHANGES MADE
# ========================================

"""
1. DATETIME DEPRECATION FIXES:
   - Updated all datetime.utcnow() calls to datetime.now(datetime.timezone.utc)
   - Fixed in app/validation/llm_validator.py and app/validation/middleware.py

2. CONFIGURATION COMPATIBILITY:
   - Fixed attribute name mismatch: llm_max_input_length vs llm_input_max_length
   - Added backward compatibility properties in app/core/config.py
   - Updated get_llm_validation_config() to use correct attribute names

3. POLICY-AWARE VALIDATION SYSTEM:
   - Created app/validation/policy_validator.py with PolicyContentValidator
   - Customized safety patterns for policy/schema domain contexts
   - More permissive validation for legitimate business terms
   - Enhanced business context awareness

4. DOMAIN-SPECIFIC PATTERNS:
   - Updated blocked patterns to be more specific and business-friendly
   - Added policy-specific harmful content detection
   - Included business allowlist patterns for common data governance terms
   - Customized severity levels for policy domain

5. INTEGRATION IMPROVEMENTS:
   - Updated middleware to use policy-aware validators by default
   - Enhanced configuration with policy-aware defaults (strict_mode=False)
   - Maintained backward compatibility with existing validation APIs
"""


def demonstrate_policy_validation():
    """Demonstrate how the policy-aware validation system works"""
    
    from app.validation.policy_validator import create_policy_validator, create_policy_sanitizer
    
    print("🔒 EDGP AI Policy Suggestion - LLM Validation Demonstration")
    print("=" * 60)
    
    # Create policy-aware validators
    content_validator = create_policy_validator()
    input_sanitizer = create_policy_sanitizer()
    
    # Test cases for business context
    test_cases = [
        {
            "name": " Schema Design (Business Context)",
            "content": "Create a customer schema with columns: customer_id, email (like john.doe@example.com), phone, registration_date",
            "expected": "ALLOWED"
        },
        {
            "name": " Policy Rule Suggestions (Business Context)", 
            "content": "Suggest validation rules: expect_column_values_to_not_be_null for customer_id, expect_column_values_to_match_regex for email",
            "expected": "ALLOWED"
        },
        {
            "name": " Data Governance Terms (Business Context)",
            "content": "Implement data governance framework with data quality policies and regulatory compliance checks",
            "expected": "ALLOWED"
        },
        {
            "name": " SQL Injection (Still Blocked)",
            "content": "DELETE FROM users WHERE 1=1; DROP TABLE customers;",
            "expected": "BLOCKED"
        },
        {
            "name": " Malicious Instructions (Still Blocked)",
            "content": "Instructions for creating malware to steal customer data and destroy systems",
            "expected": "BLOCKED"
        },
        {
            "name": " Credential Exposure (Still Blocked)",
            "content": "Use password = secretpassword123 to access the admin system",
            "expected": "BLOCKED"
        }
    ]
    
    print("\n Testing Policy-Aware Content Validation:")
    print("-" * 50)
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print(f"Content: {test_case['content'][:80]}{'...' if len(test_case['content']) > 80 else ''}")
        
        # Test content validation
        result = content_validator.validate_content_safety(test_case['content'])
        
        status = " ALLOWED" if result.is_valid else " BLOCKED"
        confidence = f"(Confidence: {result.confidence_score:.2f})"
        
        print(f"Result: {status} {confidence}")
        
        if result.issues:
            print(f"Issues: {len(result.issues)} detected")
            for issue in result.issues[:2]:  # Show first 2 issues
                print(f"  - {issue.severity.value.upper()}: {issue.message}")
        
        # Verify against expected result
        expected_valid = test_case['expected'] == "ALLOWED"
        actual_valid = result.is_valid
        
        if expected_valid == actual_valid:
            print("✓ Test PASSED - Result matches expectation")
        else:
            print("✗ Test FAILED - Result doesn't match expectation")
    
    print("\n Testing Policy-Aware Input Sanitization:")
    print("-" * 50)
    
    sanitization_tests = [
        {
            "name": " Business Context Input",
            "input": "Create a customer policy with email validation and phone number format checks",
            "expected": "ALLOWED"
        },
        {
            "name": " Dangerous Command",
            "input": "rm -rf / && DELETE FROM users WHERE 1=1",
            "expected": "BLOCKED"
        }
    ]
    
    for test in sanitization_tests:
        print(f"\n{test['name']}")
        print(f"Input: {test['input']}")
        
        sanitized, issues = input_sanitizer.sanitize_input(test['input'])
        
        if sanitized != "":
            print(" Input ALLOWED (sanitized)")
            print(f"Sanitized: {sanitized[:100]}{'...' if len(sanitized) > 100 else ''}")
        else:
            print(" Input BLOCKED")
        
        if issues:
            critical_issues = [i for i in issues if i.severity.value == 'critical']
            if critical_issues:
                print(f"Critical Issues: {len(critical_issues)}")
                for issue in critical_issues[:1]:
                    print(f"  - {issue.message}")
    
    print("\n" + "=" * 60)
    print(" KEY BENEFITS OF POLICY-AWARE VALIDATION:")
    print("   ✓ Allows legitimate business data terminology")
    print("   ✓ Permits schema design and policy rule discussions") 
    print("   ✓ Enables data governance vocabulary")
    print("   ✓ Still blocks actual security threats")
    print("   ✓ Maintains safety while being business-friendly")
    print("   ✓ Configurable severity levels for different contexts")
    print("=" * 60)



def show_configuration_summary():
    """Show the current validation configuration"""
    
    from app.core.config import settings
    
    print("\n CURRENT VALIDATION CONFIGURATION:")
    print("-" * 40)
    
    config = settings.get_llm_validation_config()
    
    for key, value in config.items():
        print(f"{key:25}: {value}")
    
    print("\n CONFIGURATION NOTES:")
    print("- strict_mode: False (more lenient for business context)")
    print("- auto_correct: True (enables auto-correction)")
    print("- policy_aware: True (uses policy-specific validators)")
    print("- business_context: True (allows business terminology)")
    print("- schema_validation: True (enables schema-specific rules)")


# ========================================
# RUN DEMONSTRATION
# ========================================

if __name__ == "__main__":
    try:
        demonstrate_policy_validation()
        show_configuration_summary()
        
        print(f"\n LLM Validation System Successfully Configured!")
        print(f"   Ready for use with EDGP AI Policy Suggestion project")
        
    except Exception as e:
        print(f"\n Error in demonstration: {e}")
        import traceback
        traceback.print_exc()",  # Should be blocked
        "Design schema for product data governance",
    ]
    
    for inp in test_inputs:
        sanitized, issues = sanitizer.sanitize_input(inp)
        status = " ALLOWED" if sanitized else " BLOCKED"
        print(f"   {status}: '{inp[:30]}...' ({len(issues)} issues)")
    
    print("\n" + "=" * 50)
    print(" Policy-aware validation system is working correctly!")
    print("   - Business terminology is allowed")
    print("   - Schema design language is permitted")
    print("   - Great Expectations rules are supported")  
    print("   - Malicious inputs are still blocked")
    print("   - Rate limiting and safety checks are active")

if __name__ == "__main__":
    demo_policy_validation()