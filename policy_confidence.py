#!/usr/bin/env python3
"""
Example of Enhanced Confidence Score Response
Shows what your API will return with the new confidence scoring system
"""

def show_confidence_example():
    """Show example of enhanced confidence response"""
    
    # Example response with new confidence structure
    enhanced_response = {
        "rule_suggestions": [
            {
                "column": "email",
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {"regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
            },
            {
                "column": "phone",
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {"regex": r"^\+?1?-?\(?[0-9]{3}\)?-?[0-9]{3}-?[0-9]{4}$"}
            },
            {
                "column": "age",
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"min_value": 18, "max_value": 120}
            },
            {
                "column": "customer_id",
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {}
            },
            {
                "column": "customer_id",
                "expectation_type": "expect_column_values_to_be_unique",
                "kwargs": {}
            }
        ],
        
        # 🎯 NEW: User-friendly confidence section
        "confidence": {
            "overall": 0.85,  # Single score for quick assessment
            "level": "high",  # Human-readable: high/medium/low/very_low
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
        
        # 🔍 ENHANCED: Agent insights for debugging
        "agent_insights": {
            "reasoning_steps": 8,
            "observations": 8,
            "reflections": 3,
            "execution_time": 4.23,
            "detailed_confidence_scores": {  # Raw internal scores
                "llm_generation": 0.87,
                "schema_parsing": 0.95,
                "rule_validation": 0.82
            },
            "key_thoughts": [
                "Generated 5 comprehensive validation rules covering all column types",
                "Successfully applied PII detection patterns for email and phone",
                "All rules normalized to Great Expectations format successfully"
            ],
            "final_reflection": "High-quality rule generation completed with excellent coverage and accuracy"
        }
    }
    
    return enhanced_response

def show_confidence_usage_patterns():
    """Show how different users would use confidence scores"""
    
    print("🎯 CONFIDENCE SCORE USAGE PATTERNS:")
    print("=" * 60)
    
    print("\n1️⃣ QUICK DECISION MAKING:")
    print("   if response['confidence']['level'] == 'high':")
    print("       auto_apply_rules(response['rule_suggestions'])")
    print("   else:")
    print("       send_for_human_review(response)")
    
    print("\n2️⃣ THRESHOLD-BASED AUTOMATION:")
    print("   overall_confidence = response['confidence']['overall']")
    print("   if overall_confidence >= 0.8:")
    print("       deploy_to_production()")
    print("   elif overall_confidence >= 0.6:")
    print("       stage_for_testing()")
    print("   else:")
    print("       flag_for_manual_review()")
    
    print("\n3️⃣ QUALITY MONITORING:")
    print("   factors = response['confidence']['factors']")
    print("   if factors['execution_performance']['status'] == 'slow':")
    print("       alert_performance_team()")
    print("   if factors['rule_generation']['rules_generated'] < 3:")
    print("       alert_quality_team()")
    
    print("\n4️⃣ USER INTERFACE DISPLAY:")
    print("   level = response['confidence']['level']")
    print("   if level == 'high':")
    print("       show_green_checkmark()")
    print("   elif level == 'medium':")
    print("       show_yellow_warning()")
    print("   else:")
    print("       show_red_alert()")

def main():
    """Run confidence score examples"""
    print("🤖 Enhanced Confidence Score System")
    print("=" * 50)
    
    print("\n📊 EXAMPLE API RESPONSE:")
    print("-" * 30)
    
    import json
    response = show_confidence_example()
    
    # Show simplified version first
    simplified = {
        "rule_suggestions": f"[{len(response['rule_suggestions'])} rules generated]",
        "confidence": response['confidence']
    }
    
    print(json.dumps(simplified, indent=2))
    
    print("\n" + "=" * 50)
    show_confidence_usage_patterns()
    
    print("\n🎉 KEY BENEFITS:")
    print("✅ Simple overall score (0.0-1.0) for quick decisions")
    print("✅ Human-readable level (high/medium/low/very_low)")
    print("✅ Detailed factors breakdown for analysis")
    print("✅ Clear status indicators for each factor")
    print("✅ Backwards compatible with existing insights")

if __name__ == "__main__":
    main()