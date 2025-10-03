from langchain.agents import tool
from langchain_community.chat_models import ChatOpenAI
from app.core.config import settings
from app.core.aws_secrets_service import require_openai_api_key
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
    """Use LLM to suggest GX rules per column."""
    
    openai_key = require_openai_api_key()
    llm = ChatOpenAI(model=settings.rules_llm_model, openai_api_key=openai_key, temperature=settings.llm_temperature)
    
    prompt = f"""
    You are a data governance expert. Given this schema:
    {json.dumps(data_schema, indent=2)}

    And these available GX rules:
    {json.dumps(gx_rules, indent=2)}

    Suggest the best validation rule(s) for each column.

    ⚠️ Important:
    - Return ONLY a valid JSON array.
    - Do NOT include markdown, explanation, or extra formatting.
    - Do NOT wrap the output in ```json or any other code block.
    - Do NOT include comments or trailing commas.

    Example format:
    [
    {{
        "column": "Email",
        "expectations": [
        {{
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {{ "regex": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$" }}
        }}
        ]
    }}
    ]
    """

    logger.info("LLM Prompt:\n%s", prompt)
    response = llm.invoke(prompt)
    logger.info("Raw LLM output:\n%s", response.content)
    return response.content.strip()


@tool
def suggest_column_names_only(domain: str) -> list:
    """Use LLM to suggest CSV column names only (no data types) for a domain not found in vector DB."""
    
    openai_key = require_openai_api_key()
    llm = ChatOpenAI(model=settings.rules_llm_model, openai_api_key=openai_key, temperature=settings.llm_temperature)

    prompt = f"""
    You are a data architect helping suggest CSV column names for a new domain called '{domain}'.
    
    Suggest 5-11 plausible CSV column names that would be commonly found in this domain.
    Focus on the most essential and representative columns for this domain.
    
    ⚠️ Important:
    - Return ONLY a JSON array of column names (strings).
    - Do NOT include data types, sample values, or any other metadata.
    - Do NOT include markdown, explanation, or extra formatting.
    - Do NOT wrap the output in ```json or any other code block.
    
    Example format:
    ["column1", "column2", "column3"]
    """

    logger.info("LLM Prompt for column names:\n%s", prompt)
    response = llm.invoke(prompt)
    logger.info("Raw LLM column names output:\n%s", response.content)
    return response.content.strip()


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
        logger.info("✅ Successfully parsed JSON directly")
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as e:
        logger.warning("⚠️ Direct JSON parse failed: %s", e)

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
        logger.info("✅ Successfully parsed JSON after fixing regex patterns")
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as e:
        logger.warning("⚠️ JSON parse failed after regex fixing: %s", e)

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
                    logger.info(f"✅ Parsed object {i+1}: {obj.get('column')}")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Failed to parse object {i+1}: {e}")
        
        if parsed_objects:
            logger.info(f"✅ Successfully extracted {len(parsed_objects)} column objects")
            return parsed_objects
        else:
            logger.error("❌ No valid column objects found")
            return [{"error": "No valid rules parsed", "raw": raw_text}]
            
    except Exception as e:
        logger.error(f"❌ Fallback parsing failed: {e}")
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
        logger.warning("⚠️ Expected list under 'raw', got: %s", type(raw))
        return {"error": "Invalid input type", "raw": raw}

    logger.info(f"🔍 Normalizing {len(raw)} raw rule objects")
    result = {}
    
    for i, item in enumerate(raw):
        try:
            # Debug logging
            logger.info(f"🔍 Processing item {i+1}: {type(item)} - {item}")
            
            if not isinstance(item, dict):
                logger.warning(f"⚠️ Item {i+1} is not a dict: {type(item)}")
                continue
                
            if "column" not in item:
                logger.warning(f"⚠️ Item {i+1} missing 'column' key. Keys: {list(item.keys())}")
                continue
                
            column = item["column"]
            expectations = item.get("expectations", [])
            result[column] = {"expectations": expectations}
            logger.info(f"✅ Successfully processed column '{column}' with {len(expectations)} expectations")
            
        except Exception as e:
            logger.warning("⚠️ Exception processing item %d: %s", i+1, e)

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
