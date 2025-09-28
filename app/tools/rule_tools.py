from langchain.agents import tool
from langchain_community.chat_models import ChatOpenAI
from app.core.config import OPENAI_API_KEY, RULE_MICROSERVICE_URL
import json, re, requests
import logging

logger = logging.getLogger(__name__)

@tool
def fetch_gx_rules(query: str = "") -> list:
    """Fetch GX rules from Rule Microservice."""
    try:
        resp = requests.get(RULE_MICROSERVICE_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning(f"Rule Microservice not available at {RULE_MICROSERVICE_URL}. Using default rules.")
        # Return a basic set of GX rules for common data types
        return [
            {
                "rule_name": "ExpectColumnValuesToNotBeNull",
                "description": "Expect column values to not be null",
                "applies_to": ["string", "integer", "float", "date", "boolean"]
            },
            {
                "rule_name": "ExpectColumnValuesToMatchRegex", 
                "description": "Expect column values to match a regular expression",
                "applies_to": ["string"]
            },
            {
                "rule_name": "ExpectColumnValuesToBeBetween",
                "description": "Expect column values to be between min and max",
                "applies_to": ["integer", "float"]
            },
            {
                "rule_name": "ExpectColumnValuesToBeOfType",
                "description": "Expect column values to be of a specific type",
                "applies_to": ["string", "integer", "float", "date", "boolean"]
            }
        ]
    except Exception as e:
        logger.error(f"Unexpected error fetching GX rules: {e}")
        return []

@tool
def suggest_column_rules(data_schema: dict, gx_rules: list) -> str:
    """Use LLM to suggest GX rules per column."""

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.2)

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
    
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.2)

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

    # Try direct JSON parsing
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as e:
        logger.warning("⚠️ Direct JSON parse failed: %s", e)

    # Fallback: extract JSON-like objects
    try:
        objs = re.findall(r"\{.*?\}", raw_text, re.DOTALL)
        parsed_objs = []
        for o in objs:
            try:
                parsed_objs.append(json.loads(o))
            except json.JSONDecodeError as e:
                logger.warning("⚠️ Skipping malformed object: %s", e)
        return parsed_objs if parsed_objs else [{"error": "No valid rules parsed", "raw": raw_text}]
    except Exception as e:
        logger.error("Failed to parse LLM output: %s", e)
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

    result = {}
    for item in raw:
        try:
            column = item["column"]
            expectations = item.get("expectations", [])
            result[column] = {"expectations": expectations}
        except Exception as e:
            logger.warning("⚠️ Skipping malformed item: %s", e)

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
