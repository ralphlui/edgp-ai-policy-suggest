from langchain.agents import tool
from langchain_community.chat_models import ChatOpenAI
from app.core.config import settings
from app.aws.aws_secrets_service import require_openai_api_key
from app.prompt.prompt_config import get_enhanced_rule_prompt, get_enhanced_column_prompt
from app.validation.middleware import AgentValidationContext, validate_input_quick, validate_output_quick
from app.validation.policy_validator import create_policy_validator, create_policy_sanitizer
from app.exception.exceptions import ValidationError
import json, re, requests
import logging
import os

logger = logging.getLogger(__name__)

@tool
def fetch_gx_rules(query: str = "") -> list:
    """Fetch GX rules from Rule Microservice."""
    try:
        rule_url = settings.rule_api_url or os.getenv("RULE_URL")
        if not rule_url or rule_url in ["{RULE_URL}", "RULE_URL"]:
            logger.warning("Rule Microservice URL not configured. Using default rules.")
            return _get_default_rules()
            
        resp = requests.get(rule_url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Rule Microservice not available. Using default rules. Error: {e}")
        return _get_default_rules()

def _get_default_rules() -> list:
    """Return default GX rules when service is unavailable"""
    return [
        {
            "rule_name": "expect_column_values_to_not_be_null",
            "description": "Validate that column values are not null",
            "applies_to": ["all"]
        },
        {
            "rule_name": "expect_column_values_to_be_of_type",
            "description": "Validate column data type",
            "applies_to": ["all"]
        },
        {
            "rule_name": "expect_column_values_to_be_in_range",
            "description": "Validate column values are within specified range",
            "applies_to": ["numeric"]
        }
    ]

@tool
def suggest_column_rules(data_schema: dict, gx_rules: list) -> str:
    """Use LLM to suggest GX rules per column with expertise and reasoning. Includes validation and safety checks."""
    
    openai_key = require_openai_api_key()
    llm = ChatOpenAI(model=settings.rules_llm_model, openai_api_key=openai_key, temperature=settings.llm_temperature)
    
    domain = data_schema.get('domain', 'unknown')
    
    # Use enhanced prompt from configuration system
    prompt = get_enhanced_rule_prompt(domain, data_schema, gx_rules)

    # Get validation config and create user context
    validation_config = None
    if settings.llm_validation_enabled:
        validation_config = settings.get_llm_validation_config()
    
    # Create a unique user ID for this request (in real app, this would come from authentication)
    user_id = f"agent_{domain}_{hash(str(data_schema)) % 10000}"
    
    try:
        # Validate input if validation is enabled
        if validation_config:
            logger.info(" Validating LLM input for safety and compliance")
            sanitized_prompt = validate_input_quick(prompt, user_id)
            logger.info(" Input validation passed")
        else:
            sanitized_prompt = prompt
            logger.info(" LLM validation disabled - proceeding without safety checks")

        logger.info("LLM Prompt:\n%s", sanitized_prompt)
        
        # Make LLM call
        response = llm.invoke(sanitized_prompt)
        raw_response = response.content.strip()
        
        logger.info("Raw LLM output:\n%s", raw_response)
        
        # Validate output if validation is enabled
        if validation_config:
            logger.info(" Validating LLM output for safety and quality")
            validated_response = validate_output_quick(raw_response, "content")
            logger.info(" Output validation passed")
            return validated_response
        else:
            logger.info(" LLM validation disabled - returning raw output")
            return raw_response
            
    except ValidationError as e:
        logger.error(f" LLM validation failed: {e}")
        # Return a safe fallback response
        return json.dumps({
            "error": "Validation failed - using safe fallback",
            "message": "The request could not be processed due to safety restrictions",
            "fallback_rules": [
                {
                    "column": col,
                    "expectations": [{"expectation": "expect_column_values_to_not_be_null"}]
                } for col in data_schema.keys() if col != "domain"
            ]
        })
    except Exception as e:
        logger.error(f" Unexpected error in LLM call: {e}")
        raise


@tool
def suggest_column_names_only(domain: str) -> list:
    """Use LLM to suggest CSV column names with business intelligence expertise. Includes validation and safety checks."""
    
    openai_key = require_openai_api_key()
    llm = ChatOpenAI(model=settings.rules_llm_model, openai_api_key=openai_key, temperature=settings.llm_temperature)

    # Use enhanced prompt from configuration system  
    prompt = get_enhanced_column_prompt(domain)

    # Get validation config
    validation_config = None
    if settings.llm_validation_enabled:
        validation_config = settings.get_llm_validation_config()
    
    # Create a unique user ID for this request
    user_id = f"agent_columns_{domain}_{hash(domain) % 10000}"
    
    try:
        # Validate input if validation is enabled
        if validation_config:
            logger.info(" Validating column suggestion input")
            sanitized_prompt = validate_input_quick(prompt, user_id)
            logger.info(" Input validation passed")
        else:
            sanitized_prompt = prompt
            logger.info(" LLM validation disabled")

        logger.info("LLM Prompt for column names:\n%s", sanitized_prompt)
        
        # Make LLM call
        response = llm.invoke(sanitized_prompt)
        raw_response = response.content.strip()
        
        logger.info("Raw LLM column names output:\n%s", raw_response)
        
        # Validate output if validation is enabled
        if validation_config:
            logger.info(" Validating column suggestion output")
            validated_response = validate_output_quick(raw_response, "content")
            logger.info(" Output validation passed")
            return validated_response
        else:
            logger.info(" LLM validation disabled")
            return raw_response
            
    except ValidationError as e:
        logger.error(f" Column suggestion validation failed: {e}")
        # Return safe fallback column names
        return json.dumps([
            f"{domain}_id",
            f"{domain}_name", 
            f"{domain}_description",
            "created_date",
            "updated_date"
        ])
    except Exception as e:
        logger.error(f" Unexpected error in column suggestion: {e}")
        raise


@tool
def format_gx_rules(raw_text: str) -> list:
    """Parse LLM output into structured rule list. Handles malformed JSON gracefully."""
    import json, re, logging
    logger = logging.getLogger(__name__)

    # Log the raw output for debugging
    logger.info("Raw LLM output:\n%s", raw_text)

    # Try direct JSON parsing first
    try:
        parsed = json.loads(raw_text)
        logger.info(" Successfully parsed JSON directly")
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as e:
        logger.warning(" Direct JSON parse failed: %s", e)

    # Fix regex escape issues comprehensively
    try:
        def fix_regex_patterns(text):
            """Fix regex patterns in JSON by properly escaping backslashes"""
            def escape_regex(match):
                regex_value = match.group(2)
                # Replace single backslashes with double backslashes
                escaped_regex = regex_value.replace('\\', '\\\\')
                return f'{match.group(1)}"{escaped_regex}"{match.group(3)}'
            
            # Pattern to match: "regex": "pattern"
            return re.sub(r'("regex"\s*:\s*)"([^"]*)"(\s*[,}])', escape_regex, text)
        
        cleaned_text = fix_regex_patterns(raw_text)
        parsed = json.loads(cleaned_text)
        logger.info(" Successfully parsed JSON after fixing regex patterns")
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as e:
        logger.warning(" JSON parse failed after regex fixing: %s", e)

    # Final fallback: extract individual column objects manually
    try:
        # Extract complete column objects that have both "column" and "expectations"
        pattern = r'\{\s*"column"\s*:\s*"[^"]+"\s*,\s*"expectations"\s*:\s*\[[^\]]*(?:\{[^}]*\}[^\]]*)*\]\s*\}'
        matches = re.findall(pattern, raw_text, re.DOTALL)
        
        if not matches:
            # Broader pattern for any object containing "column"
            pattern = r'\{[^{}]*"column"[^{}]*(?:\{[^}]*\}[^{}]*)*\}'
            matches = re.findall(pattern, raw_text, re.DOTALL)
        
        parsed_objects = []
        for i, match in enumerate(matches):
            try:
                # Apply regex fixing to each match
                clean_match = fix_regex_patterns(match)
                obj = json.loads(clean_match)
                
                # Verify it's a proper column object
                if isinstance(obj, dict) and "column" in obj:
                    parsed_objects.append(obj)
                    logger.info(f" Parsed object {i+1}: {obj.get('column')}")
                    
            except json.JSONDecodeError as e:
                logger.warning(f" Failed to parse object {i+1}: {e}")
        
        if parsed_objects:
            logger.info(f" Successfully extracted {len(parsed_objects)} column objects")
            return parsed_objects
        else:
            logger.error(" No valid column objects found")
            return [{"error": "No valid rules parsed", "raw": raw_text}]
            
    except Exception as e:
        logger.error(f" Fallback parsing failed: {e}")
        return [{"error": f"Could not parse rules: {e}", "raw": raw_text}]

@tool
def normalize_rule_suggestions(rule_input: dict) -> dict:
    """
    Expects: { "raw": [...] }
    Returns: { "ColumnName": { "expectations": [...] }, ... }
    """
    import logging
    logger = logging.getLogger(__name__)

    raw = rule_input.get("raw", [])
    if not isinstance(raw, list):
        logger.warning(" Expected list under 'raw', got: %s", type(raw))
        return {"error": "Invalid input type", "raw": raw}

    logger.info(f"🔍 Normalizing {len(raw)} raw rule objects")
    result = {}
    
    for i, item in enumerate(raw):
        try:
            # Debug logging
            logger.info(f" Processing item {i+1}: {type(item)} - {item}")
            
            if not isinstance(item, dict):
                logger.warning(f" Item {i+1} is not a dict: {type(item)}")
                continue
                
            if "column" not in item:
                logger.warning(f" Item {i+1} missing 'column' key. Keys: {list(item.keys())}")
                continue
                
            column = item["column"]
            expectations = item.get("expectations", [])
            result[column] = {"expectations": expectations}
            logger.info(f" Successfully processed column '{column}' with {len(expectations)} expectations")
            
        except Exception as e:
            logger.warning(" Exception processing item %d: %s", i+1, e)

    logger.info(f"🎯 Normalization complete: {len(result)} columns processed")
    return result

@tool
def convert_to_rule_ms_format(rule_input: dict) -> list:
    """
    Convert normalized suggestions to Rule Microservice format.
    Expects: { "suggestions": { column: { expectations: [...] } } }
    """
    suggestions = rule_input.get("suggestions", {})
    result = []

    for column, data in suggestions.items():
        if not isinstance(data, dict):
            continue
        for rule in data.get("expectations", []):
            rule_name = rule.get("expectation_type")
            kwargs = rule.get("kwargs", {})

            gx_rule_name = "".join(
                word.capitalize() for word in rule_name.replace("expect_", "").split("_")
            )
            gx_rule_name = f"Expect{gx_rule_name}"

            result.append({
                "rule_name": gx_rule_name,
                "column_name": column,
                "value": kwargs if kwargs else None
            })

    return result
