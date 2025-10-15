#!/usr/bin/env python3
"""
Test what the actual rule/suggest API will return with confidence scores
"""

def simulate_api_response():
    """Simulate what your /api/aips/rules/suggest endpoint will return"""
    
    print(" Your Enhanced /api/aips/rules/suggest API Response")
    print("=" * 60)
    
    print("\n REQUEST:")
    print("POST /api/aips/rules/suggest")
    print('{"domain": "customer"}')
    
    print("\n RESPONSE (with default insights + confidence):")
    
    # This is what your API will actually return
    api_response = {
        "rule_suggestions": [
            {
                "column": "customer_id",
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {}
            },
            {
                "column": "customer_id", 
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {}
            },
            {
                "column": "email",
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {
                    "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                }
            },
            {
                "column": "phone",
                "expectation_type": "expect_column_values_to_match_regex", 
                "kwargs": {
                    "regex": r"^\+?1?-?\(?[0-9]{3}\)?-?[0-9]{3}-?[0-9]{4}$"
                }
            },
            {
                "column": "age",
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {
                    "min_value": 18,
                    "max_value": 120
                }
            }
        ],
        
        #  NEW: Confidence section (always included by default)
        "confidence": {
            "overall": 0.85,
            "level": "high",
            "factors": {
                "rule_generation": {
                    "rules_generated": 5,
                    "score": 0.85,
                    "status": "good"
                },
                "error_handling": {
                    "errors_encountered": 0,
                    "score": 1.0,
                    "status": "good"
                },
                "execution_performance": {
                    "duration_seconds": 4.23,
                    "score": 0.88,
                    "status": "normal"
                },
                "reasoning_depth": {
                    "thoughts_generated": 8,
                    "observations_made": 8,
                    "reflections_completed": 3,
                    "score": 0.75,
                    "status": "thorough"
                }
            }
        },
        
        #  Enhanced: Agent insights (always included by default)
        "agent_insights": {
            "reasoning_steps": 8,
            "observations": 8,
            "reflections": 3,
            "execution_time": 4.23,
            "detailed_confidence_scores": {
                "llm_generation": 0.87,
                "schema_parsing": 0.95,
                "rule_validation": 0.82
            },
            "key_thoughts": [
                "I need to generate validation rules for 'customer' domain with 5 columns including PII fields",
                "Successfully applied PII detection patterns for email and phone columns",
                "All rules have been normalized to Great Expectations format successfully"
            ],
            "final_reflection": "High-quality rule generation completed with excellent coverage for all column types including proper PII handling"
        }
    }
    
    import json
    print(json.dumps(api_response, indent=2))
    
    print("\n" + "=" * 60)
    print(" CONFIDENCE SCORES ARE INCLUDED BY DEFAULT!")
    print(" Users get full transparency into AI reasoning")
    print(" Production-ready quality metrics")
    print(" Backwards compatible (can opt-out with include_insights: false)")

if __name__ == "__main__":
    simulate_api_response()