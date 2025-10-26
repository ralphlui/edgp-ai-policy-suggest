from langchain.agents import tool
from langchain_openai import ChatOpenAI
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
            return _get_default_rules()  # Return all default rules
            
        resp = requests.get(rule_url, timeout=3)  # Keep fast timeout
        resp.raise_for_status()
        rules = resp.json()
        
        # Group rules by type for efficient processing
        if isinstance(rules, list):
            rules_by_type = {}
            for rule in rules:
                rule_type = rule.get("applies_to", ["all"])[0]
                if rule_type not in rules_by_type:
                    rules_by_type[rule_type] = []
                rules_by_type[rule_type].append(rule)
            return rules  # Return all rules with type grouping
        return rules
    except Exception as e:
        logger.warning(f"Rule Microservice not available. Using default rules. Error: {e}")
        return _get_default_rules()

def _get_default_rules() -> list:
    """Return default GX rules when service is unavailable"""
    return [
        {
            "rule_name": "ExpectColumnDistinctValuesToBeInSet",
            "column_name": "test",
            "value": [1, 2, 3, 4, 5]
        },
        {
            "rule_name": "ExpectColumnValuesToBeInSet",
            "column_name": "test",
            "value": ["value1", "value2"]
        },
        {
            "rule_name": "ExpectColumnValuesToNotBeInSet",
            "column_name": "test",
            "value": ["invalid1", "invalid2"]
        },
        {
            "rule_name": "ExpectColumnValuesToBeBetween",
            "column_name": "age",
            "value": {
                "min_value": 0,
                "max_value": 100
            }
        },
        {
            "rule_name": "ExpectColumnValueLengthsToBeBetween",
            "column_name": "description",
            "value": {
                "min_value": 5,
                "max_value": 500
            }
        },
        {
            "rule_name": "ExpectColumnValuesToMatchRegex",
            "column_name": "phone",
            "value": {
                "regex": "^\\d{3}-\\d{3}-\\d{4}$"
            }
        },
        {
            "rule_name": "ExpectColumnValuesToNotMatchRegex",
            "column_name": "ssn",
            "value": {
                "regex": "\\d{3}-\\d{2}-\\d{4}"
            }
        },
        {
            "rule_name": "ExpectColumnValuesToMatchStrftimeFormat", 
            "column_name": "date",
            "value": {
                "strftime_format": "%Y-%m-%d"
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeUnique",
            "column_name": "id",
            "value": None
        },
        {
            "rule_name": "ExpectCompoundColumnsToBeUnique",
            "column_name": None,
            "value": {
                "column_list": ["first_name", "last_name", "dob"]
            }
        },
        {
            "rule_name": "ExpectColumnMeanToBeBetween",
            "column_name": "salary",
            "value": {
                "min_value": 50000,
                "max_value": 100000
            }
        },
        {
            "rule_name": "ExpectColumnMedianToBeBetween",
            "column_name": "price",
            "value": {
                "min_value": 20,
                "max_value": 50
            }
        },
        {
            "rule_name": "ExpectColumnSumToBeBetween",
            "column_name": "inventory",
            "value": {
                "min_value": 1000,
                "max_value": 2000
            }
        },
        {
            "rule_name": "ExpectColumnMinToBeBetween",
            "column_name": "temperature",
            "value": {
                "min_value": -10,
                "max_value": 0
            }
        },
        {
            "rule_name": "ExpectColumnMaxToBeBetween",
            "column_name": "score",
            "value": {
                "min_value": 90,
                "max_value": 100
            }
        },
        {
            "rule_name": "ExpectColumnProportionOfUniqueValuesToBeBetween",
            "column_name": "user_id",
            "value": {
                "min_value": 0.95,
                "max_value": 1.0
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeOfType",
            "column_name": "count",
            "value": {
                "type_": "INTEGER"
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeInTypeList",
            "column_name": "comment",
            "value": {
                "type_list": ["VARCHAR", "TEXT"]
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeDateutilParseable",
            "column_name": "event_time",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToMatchLikePattern",
            "column_name": "product_code",
            "value": {
                "like_pattern": "PROD_%"
            }
        },
        {
            "rule_name": "ExpectColumnValuesToNotMatchLikePattern",
            "column_name": "internal_code",
            "value": {
                "like_pattern": "TEST_%"
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeBoolean",
            "column_name": "is_active",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToBeNone",
            "column_name": "optional_field",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToNotBeNone",
            "column_name": "required_field",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValueLengthsToEqual",
            "column_name": "zip_code",
            "value": {
                "value": 5
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBePositive",
            "column_name": "quantity",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToBeLessThan",
            "column_name": "age",
            "value": {
                "value": 120
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeGreaterThan",
            "column_name": "salary",
            "value": {
                "value": 0
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeIncreasing",
            "column_name": "timestamp",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToBeDecreasing",
            "column_name": "depreciation",
            "value": None
        },
        {
            "rule_name": "ExpectColumnPairValuesAToBeGreaterThanB",
            "column_name": None,
            "value": {
                "column_A": "revenue",
                "column_B": "cost",
                "or_equal": True
            }
        },
        {
            "rule_name": "ExpectColumnPairValuesToBeEqual",
            "column_name": None,
            "value": {
                "column_A": "id",
                "column_B": "reference_id"
            }
        },
        {
            "rule_name": "ExpectTableColumnsToMatchOrderedList",
            "column_name": None,
            "value": {
                "column_list": ["id", "name", "date"]
            }
        },
        {
            "rule_name": "ExpectTableColumnCountToBeBetween",
            "column_name": None,
            "value": {
                "min_value": 5,
                "max_value": 10
            }
        },
        {
            "rule_name": "ExpectTableRowCountToEqual",
            "column_name": None,
            "value": {
                "value": 10000
            }
        },
        {
            "rule_name": "ExpectTableRowCountToBeBetween",
            "column_name": None,
            "value": {
                "min_value": 9000,
                "max_value": 11000
            }
        },
        {
            "rule_name": "ExpectTableCustomQueryToReturnNoRows",
            "column_name": None,
            "value": {
                "query": "SELECT * FROM table WHERE total < 0"
            }
        },
        {
            "rule_name": "ExpectColumnToExist",
            "column_name": "required_column",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToBeValidEmail",
            "column_name": "email",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToBeValidUrl",
            "column_name": "website",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToBeValidIPv4",
            "column_name": "ip_address",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToBeValidCreditCardNumber",
            "column_name": "credit_card",
            "value": None
        },
        {
            "rule_name": "ExpectColumnValuesToBeAfter",
            "column_name": "event_date",
            "value": {
                "min_date": "2023-01-01"
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeBefore",
            "column_name": "expiry_date",
            "value": {
                "max_date": "2025-12-31"
            }
        },
        {
            "rule_name": "ExpectColumnValuesToBeBetweenDates",
            "column_name": "order_date",
            "value": {
                "min_date": "2023-01-01",
                "max_date": "2023-12-31"
            }
        }
    ]

@tool
def suggest_column_rules(data_schema: dict, gx_rules: list) -> str:
    """Use LLM to suggest GX rules for all columns in a single call with optimized batching."""
    
    openai_key = require_openai_api_key()
    llm = ChatOpenAI(
        model=settings.schema_llm_model,
        openai_api_key=openai_key,
        temperature=0.1,
        max_tokens=4000,
        request_timeout=45,  # Increased for larger batches
        max_retries=3,  # More retries for reliability
        streaming=False,
        seed=42
    )
    
    # Extract domain and columns 
    domain = data_schema.get('domain', 'unknown')
    columns = [k for k in data_schema.keys() if k != 'domain']
    
    if not columns:
        logger.warning("No columns found in schema")
        return "[]"
    
    # Group columns by inferred type for batched processing
    type_groups = {}
    
    for col in columns:
        # Get type from schema or infer it
        col_type = data_schema[col].get('type', 'unknown').lower()
        if col_type == 'unknown':
            col_name = col.upper()
            if '_DATE' in col_name or 'DATE' in col_name:
                col_type = 'date'
            elif '_NUM' in col_name or 'NUMBER' in col_name or col_name.endswith('_ID'):
                col_type = 'number'
            elif '_FLAG' in col_name or col_name.endswith('_YN'):
                col_type = 'boolean'
            else:
                col_type = 'string'
                
        if col_type not in type_groups:
            type_groups[col_type] = []
        type_groups[col_type].append(col)
    
    # Process each type group in a single batch
    results = []
    
    for data_type, group_columns in type_groups.items():
        # Get rules applicable to this type
        type_rules = [r for r in gx_rules if 
                     data_type in r.get('applies_to', []) or 
                     'all' in r.get('applies_to', [])]
        
        # Prepare focused schema for this group
        group_schema = {
            'domain': domain,
            'columns': {}
        }
        
        for col in group_columns:
            group_schema['columns'][col] = {
                'type': data_type,
                'name': col,
                'description': data_schema[col].get('description', ''),
                'constraints': data_schema[col].get('constraints', {}),
                'format': data_schema[col].get('format', ''),
                'sample_values': data_schema[col].get('sample_values', [])
            }
        
        # Generate prompt for all columns of this type
        column_info = [f"{col}({data_type})" for col in group_columns]
        prompt = get_enhanced_rule_prompt(domain, group_schema, type_rules)
        
        # Add explicit JSON formatting instructions
        prompt += f"""
Generate rules for these {data_type} columns: {', '.join(column_info)}

CRITICAL: Respond ONLY with a JSON array. No markdown, no extra text.

Required structure:
[
  {{
    "column": "column_name",
    "expectations": [
      {{
        "expectation_type": "rule_name",
        "kwargs": {{
          "key": "value"
        }}
      }}
    ]
  }}
]

Rules should be relevant for {data_type} data type. Include validation for:
- Data type consistency 
- Value ranges if applicable
- Format validation
- Business logic constraints
- NULL handling
"""

        try:
            # Make single LLM call for all columns of this type
            result = _process_llm_request(llm, prompt)
            
            if result and result != "[]":
                results.append(result)
            else:
                # Generate type-specific fallbacks
                fallbacks = []
                for col in group_columns:
                    fallbacks.append(generate_type_specific_fallback(col, data_type))
                results.append(json.dumps(fallbacks))
                
        except Exception as e:
            logger.error(f"Error processing {data_type} columns: {e}")
            # Add fallback rules
            fallbacks = []
            for col in group_columns:
                fallbacks.append(generate_type_specific_fallback(col, data_type))
            results.append(json.dumps(fallbacks))

    # Combine all results
    if not results:
        # Global fallback
        fallbacks = []
        for col in columns:
            fallbacks.append({
                "column": col,
                "expectations": [
                    {"expectation_type": "expect_column_values_to_not_be_null"}
                ]
            })
        return json.dumps(fallbacks)
    
    # Combine valid results
    combined = "[\n" + ",\n".join(
        result.strip('[]') for result in results if result and result.strip('[]')
    ) + "\n]"
    
    return combined
        
def generate_type_specific_fallback(column: str, data_type: str) -> dict:
    """Generate type-specific fallback rules for a column"""
    rules = [{"expectation_type": "expect_column_values_to_not_be_null"}]
    
    if data_type in ['number', 'integer', 'float']:
        rules.extend([
            {
                "expectation_type": "expect_column_values_to_be_in_type_list",
                "kwargs": {"type_list": ["number"]},
                "meta": {"reasoning": "Ensure numeric data type consistency"}
            },
            {
                "expectation_type": "expect_column_values_to_be_in_range",
                "kwargs": {"min_value": None, "max_value": None},
                "meta": {"reasoning": "Validate numeric range"}
            }
        ])
    elif data_type in ['string', 'text']:
        rules.extend([
            {
                "expectation_type": "expect_column_values_to_be_in_type_list",
                "kwargs": {"type_list": ["string"]},
                "meta": {"reasoning": "Ensure string data type consistency"}
            },
            {
                "expectation_type": "expect_column_values_to_match_regex",
                "kwargs": {"regex": "^.+$"},
                "meta": {"reasoning": "Validate non-empty string content"}
            }
        ])
    elif data_type in ['date', 'datetime']:
        rules.extend([
            {
                "expectation_type": "expect_column_values_to_be_dateutil_parseable",
                "meta": {"reasoning": "Ensure valid date/time format"}
            },
            {
                "expectation_type": "expect_column_values_to_be_in_type_list",
                "kwargs": {"type_list": ["datetime", "string"]},
                "meta": {"reasoning": "Allow both datetime and string representations"}
            }
        ])
    elif data_type in ['boolean']:
        rules.extend([
            {
                "expectation_type": "expect_column_values_to_be_in_type_list",
                "kwargs": {"type_list": ["boolean"]},
                "meta": {"reasoning": "Ensure boolean data type"}
            },
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "kwargs": {"value_set": [True, False]},
                "meta": {"reasoning": "Validate boolean values"}
            }
        ])
    elif data_type in ['array', 'list']:
        rules.extend([
            {
                "expectation_type": "expect_column_values_to_be_in_type_list",
                "kwargs": {"type_list": ["array"]},
                "meta": {"reasoning": "Ensure array data type"}
            }
        ])
    elif data_type in ['object', 'json']:
        rules.extend([
            {
                "expectation_type": "expect_column_values_to_be_json_parseable",
                "meta": {"reasoning": "Validate JSON structure"}
            }
        ])
        
    return {
        "column": column,
        "expectations": rules
    }

def _process_llm_request(llm, prompt: str) -> str:
    """Process a single LLM request and return the response"""
    try:
        # Make LLM call
        response = llm.invoke([{"role": "user", "content": prompt}])
        result = response.content.strip()
        
        # Log raw response for debugging
        logger.debug(f"Raw LLM response: {result}")
        
        # Extract JSON if wrapped in markdown code blocks
        if result.startswith("```json"):
            result = result[7:]  # Remove ```json
        if result.startswith("```"):
            result = result[3:]  # Remove ```
        if result.endswith("```"):
            result = result[:-3]  # Remove trailing ```
        
        # Clean up any remaining markdown or text
        result = result.strip()
        
        # Validate JSON format
        try:
            # Try to parse as JSON object first
            parsed = json.loads(result)
            
            # Ensure we have an array of objects
            if isinstance(parsed, dict):
                # If we got a single object, wrap it in an array
                if "column" in parsed and "expectations" in parsed:
                    parsed = [parsed]
                # If we got a JSON object with a rules array
                elif "rules" in parsed and isinstance(parsed["rules"], list):
                    parsed = parsed["rules"]
            
            # Validate structure
            if isinstance(parsed, list):
                # Ensure each item has required fields
                for item in parsed:
                    if not isinstance(item, dict) or "column" not in item or "expectations" not in item:
                        logger.warning("Invalid rule structure in response")
                        return "[]"
                return json.dumps(parsed, indent=2)
            else:
                logger.warning("Response is not a list of rules")
                return "[]"
                
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON response from LLM: {str(e)}")
            return "[]"
            
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return "[]"


@tool
def suggest_column_names_only(domain: str) -> list:
    """Use LLM to suggest CSV column names with business intelligence expertise. Includes validation and safety checks."""
    
    openai_key = require_openai_api_key()
    llm = ChatOpenAI(
        model=settings.schema_llm_model,
        openai_api_key=openai_key,
        temperature=settings.llm_temperature,
        seed=42,  # Consistent outputs
        model_kwargs={
            'response_format': {'type': 'json_object'}  # Force JSON output
        }
    )

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

    logger.info(f" Normalizing {len(raw)} raw rule objects")
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

    logger.info(f" Normalization complete: {len(result)} columns processed")
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

        # Track if we've added type-specific rules for this column
        has_type_specific_rules = False
            
        for rule in data.get("expectations", []):
            rule_name = rule.get("expectation_type")
            kwargs = rule.get("kwargs", {})
            meta = rule.get("meta", {})

            # Convert rule name to proper format
            gx_rule_name = "".join(
                word.capitalize() for word in rule_name.replace("expect_", "").split("_")
            )
            gx_rule_name = f"Expect{gx_rule_name}"

            # Prepare rule value based on rule type
            rule_value = None
            if kwargs:
                if "type_list" in kwargs:
                    rule_value = {"type_list": kwargs["type_list"]}
                    has_type_specific_rules = True
                elif "regex" in kwargs:
                    rule_value = {"regex": kwargs["regex"]}
                    has_type_specific_rules = True
                elif "min_value" in kwargs or "max_value" in kwargs:
                    rule_value = {
                        "min_value": kwargs.get("min_value"),
                        "max_value": kwargs.get("max_value")
                    }
                    has_type_specific_rules = True
                elif "value_set" in kwargs:
                    rule_value = {"value_set": kwargs["value_set"]}
                    has_type_specific_rules = True
                else:
                    rule_value = kwargs

            # Add the rule without meta tag
            result.append({
                "rule_name": gx_rule_name,
                "column_name": column,
                "value": rule_value
            })

        # If no type-specific rules were added, add appropriate ones based on data type
        if not has_type_specific_rules:
            # Get column info from the original data schema
            col_info = data.get("column_info", {})
            col_type = col_info.get("type", "").lower()

            if col_type in ["number", "integer", "float"]:
                result.append({
                    "rule_name": "ExpectColumnValuesToBeInTypeList",
                    "column_name": column,
                    "value": {"type_list": ["number"]}
                })
            elif col_type == "date" or column.upper().endswith("_DATE") or "DATE" in column.upper():
                result.append({
                    "rule_name": "ExpectColumnValuesToBeDateutilParseable",
                    "column_name": column,
                    "value": None
                })
            elif col_type == "string" or True:  # Default to string type rules if unknown
                result.append({
                    "rule_name": "ExpectColumnValuesToBeInTypeList",
                    "column_name": column,
                    "value": {"type_list": ["string"]}
                })

    return result
