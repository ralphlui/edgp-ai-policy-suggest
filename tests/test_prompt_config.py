import pytest
from app.prompt.prompt_config import (
    PromptConfig,
    PromptComplexity,
    IndustryDomain,
    EnhancedPromptManager,
    get_prompt_manager,
    get_enhanced_rule_prompt,
    get_enhanced_schema_prompt,
    get_enhanced_column_prompt,
)


def test_rule_prompt_includes_domain_and_counts_and_compliance():
    config = PromptConfig(include_compliance=True, include_reasoning=True)
    mgr = EnhancedPromptManager(config)
    schema = {
        "domain": "customer",
        "id": {"dtype": "integer", "sample_values": [1, 2, 3]},
        "email": {"dtype": "string", "sample_values": ["a@b.com"]},
    }
    gx_rules = [{"rule_name": "rule1", "description": "desc"}]

    prompt = mgr.get_rule_generation_prompt("customer", schema, gx_rules)

    assert "Domain: customer" in prompt
    # Includes total keys in schema dict (including 'domain')
    assert "3 columns detected" in prompt
    assert "1 Great Expectations patterns" in prompt
    # Compliance block included
    assert "COMPLIANCE REQUIREMENTS" in prompt
    # Reasoning framework included
    assert "REASONING FRAMEWORK" in prompt


def test_rule_prompt_without_compliance_or_reasoning():
    config = PromptConfig(include_compliance=False, include_reasoning=False)
    mgr = EnhancedPromptManager(config)
    schema = {"domain": "general", "name": {"dtype": "string", "sample_values": ["x"]}}

    prompt = mgr.get_rule_generation_prompt("general", schema, [])

    # No compliance block text present
    assert "COMPLIANCE REQUIREMENTS" not in prompt
    # Reasoning collapsed into brief text
    assert "Apply standard validation patterns" in prompt
    # When no rules, include default message
    assert "Using default Great Expectations rule set" in prompt


def test_schema_prompt_respects_config_params_and_patterns():
    config_params = {
        "min_columns": 3,
        "max_columns": 7,
        "supported_types": ["string", "integer"],
        "min_samples": 2,
        "format_instructions": "JSON only"
    }

    prompt = get_enhanced_schema_prompt("product", config_params)

    assert "DOMAIN FOCUS:** product" in prompt
    assert "3-7 columns" in prompt
    assert "string, integer" in prompt
    assert "2 realistic sample values" in prompt
    assert "JSON only" in prompt
    # Should include some domain patterns section
    assert "COLUMN DESIGN PATTERNS" in prompt


def test_column_prompt_includes_domain_and_naming_standards():
    prompt = get_enhanced_column_prompt("finance")
    assert "DOMAIN:** finance" in prompt
    assert "NAMING CONVENTIONS" in prompt
    assert "Return ONLY a JSON array" in prompt
