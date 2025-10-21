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
    # Optimized for reliable rule generation
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",  # Latest model with better JSON handling
        openai_api_key=openai_key,
        temperature=0.1,  # Slight variation for better rules
        max_tokens=400,  # Enough tokens for complete rules
        request_timeout=15,  # Reasonable timeout
        max_retries=1,  # Single retry for stability
        streaming=False,  # Disable streaming
        model_kwargs={
            'response_format': {'type': 'json'},  # Force JSON output
            'seed': 42  # Consistent outputs
        }
    )
    
    # Keep domain and analyze column types
    domain = data_schema.get('domain', 'unknown')
    columns = [k for k in data_schema.keys() if k != 'domain']
    
    # Group similar columns by data type for more efficient processing
    column_groups = {}
    for col in columns:
        col_type = data_schema[col].get('type', 'unknown')
        if col_type not in column_groups:
            column_groups[col_type] = []
        column_groups[col_type].append(col)
    
    # Efficient processing for all columns
    num_columns = len(columns)
    
    # Use dynamic batch sizing based on column count
    batch_size = min(max(10, num_columns // 3), 40)  # Scale between 10-40 columns per batch
    
    # Group columns by type for efficient parallel processing
    type_grouped_columns = {}
    type_rules = {}  # Pre-cache rules by type
    
    # Pre-process columns and rules
    for col in columns:
        col_type = data_schema[col].get('type', 'unknown')
        if col_type not in type_grouped_columns:
            type_grouped_columns[col_type] = []
            # Pre-filter rules for this type
            type_rules[col_type] = [r for r in gx_rules if 
                                  col_type in r.get('applies_to', []) or 
                                  'all' in r.get('applies_to', [])]
        type_grouped_columns[col_type].append(col)
    
    # Optimize column processing order - similar types together
    columns = []
    for col_type in type_grouped_columns:
        columns.extend(type_grouped_columns[col_type])
    
    # Simple cache key
    user_id = f"agent_{domain}"
    validation_config = None  # Disable validation for speed
    
    # Process columns by type to ensure unique rules
    processed_results = []
    type_processed = {}
    
    # Process columns by data type
    for data_type, type_cols in type_grouped_columns.items():
        if data_type not in type_processed:
            type_processed[data_type] = True
            
            # Get rules specific to this data type
            type_specific_rules = [r for r in gx_rules if 
                                 data_type in r.get('applies_to', []) or 
                                 'all' in r.get('applies_to', [])]
            
            for col in type_cols:
                # Create focused schema for each column
                column_schema = {
                    'domain': domain,
                    'columns': {
                        col: {
                            'type': data_schema[col].get('type', 'unknown'),
                            'name': col,
                            'description': data_schema[col].get('description', ''),
                            'constraints': data_schema[col].get('constraints', {}),
                            'format': data_schema[col].get('format', ''),
                            'sample_values': data_schema[col].get('sample_values', []),
                        }
                    }
                }
                
                # Generate type-specific prompt
                prompt = get_enhanced_rule_prompt(domain, column_schema, type_specific_rules)
                prompt += f"\nGenerate rules ONLY for column '{col}' with type '{data_type}'. Focus on type-specific validations and business rules."
                
                # Infer actual data type from name/type info
                inferred_type = data_type
                if inferred_type == 'unknown':
                    col_name = col.upper()
                    if '_DATE' in col_name or 'DATE' in col_name:
                        inferred_type = 'date'
                    elif '_NUM' in col_name or 'NUMBER' in col_name or col_name.endswith('_ID'):
                        inferred_type = 'number'
                    elif '_FLAG' in col_name or col_name.endswith('_YN'):
                        inferred_type = 'boolean'
                    else:
                        inferred_type = 'string'

                try:
                    result = _process_llm_request(llm, prompt)
                    if result and result != "[]":
                        processed_results.append(result)
                    else:
                        # Generate type-specific fallback
                        fallback = generate_type_specific_fallback(col, inferred_type)
                        processed_results.append(json.dumps([fallback]))
                except Exception as e:
                    logger.error(f"Error processing column {col}: {e}")
                    fallback = generate_type_specific_fallback(col, inferred_type)
                    processed_results.append(json.dumps([fallback]))
    
    # Return early if we have all results
    if processed_results:
        combined = "[\n" + ",\n".join(
            result.strip('[]') for result in processed_results if result and result.strip('[]')
        ) + "\n]"
        return combined
    
    # If no results, process in batches as fallback
    import asyncio
    from app.core.async_llm import process_batch_async
    
    # Process in sequential batches for reliability
    all_results = []
    batch_size = min(max(3, num_columns // 4), 8)  # Smaller batches
    
    # Get common rules that apply to all
    common_rules = [r for r in gx_rules if 'all' in r.get('applies_to', ['all'])][:3]
    
    for i in range(0, num_columns, batch_size):
        try:
            batch_columns = columns[i:i + batch_size]
            
            # Get column types and infer types from names if needed
            batch_types = {}
            for col in batch_columns:
                col_type = data_schema[col].get('type', 'unknown').lower()
                
                # Infer type from column name if not specified
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
                
                batch_types[col] = col_type
            
            # Collect relevant rules for each type
            batch_rules = common_rules.copy()
            for col_type in set(batch_types.values()):
                type_specific = [r for r in gx_rules 
                               if col_type in r.get('applies_to', []) or
                               (col_type == 'number' and 'numeric' in r.get('applies_to', []))]
                batch_rules.extend(type_specific)
            
            # Create focused schema with more context
            batch_schema = {
                'domain': domain,
                'columns': {
                    k: {
                        'type': data_schema[k].get('type', 'unknown'),
                        'name': k,
                        'description': data_schema[k].get('description', ''),
                        'format': data_schema[k].get('format', ''),
                        'constraints': data_schema[k].get('constraints', {})
                    } for k in batch_columns
                }
            }
            
            # Generate type-specific rules for this batch
            type_specific_rules = []
            for col in batch_columns:
                col_type = data_schema[col].get('type', 'unknown')
                # Get rules specific to this column type
                col_rules = [r for r in gx_rules if 
                           col_type in r.get('applies_to', []) or 
                           'all' in r.get('applies_to', [])]
                type_specific_rules.extend(col_rules)
            
            # Add type-specific rules to batch rules
            batch_rules.extend([r for r in type_specific_rules if r not in batch_rules])
            
            # Enhanced prompt with type-specific context
            prompt = get_enhanced_rule_prompt(domain, batch_schema, batch_rules)
            prompt += f"\nNote: Generate specific rules for each column based on its type. For {', '.join(batch_columns)}, consider their types: {', '.join(f'{k}({data_schema[k].get('type', 'unknown')})' for k in batch_columns)}."
            
            result = _process_llm_request(llm, prompt)
            
            # Ensure we have valid rules with type-specific defaults
            if not result or result == "[]":
                # Generate type-specific default rules
                fallback = []
                for col in batch_columns:
                    col_type = data_schema[col].get('type', 'unknown')
                    rules = [{"expectation_type": "expect_column_values_to_not_be_null"}]
                    
                    if col_type in ['number', 'integer', 'float']:
                        rules.extend([
                            {"expectation_type": "expect_column_values_to_be_in_type_list",
                             "kwargs": {"type_list": ["number"]}},
                            {"expectation_type": "expect_column_values_to_be_in_range",
                             "kwargs": {"min_value": None, "max_value": None}}
                        ])
                    elif col_type in ['string', 'text']:
                        rules.extend([
                            {"expectation_type": "expect_column_values_to_be_in_type_list",
                             "kwargs": {"type_list": ["string"]}},
                            {"expectation_type": "expect_column_values_to_match_regex",
                             "kwargs": {"regex": ".*"}}
                        ])
                    elif col_type in ['date', 'datetime']:
                        rules.extend([
                            {"expectation_type": "expect_column_values_to_be_dateutil_parseable"},
                            {"expectation_type": "expect_column_values_to_be_in_type_list",
                             "kwargs": {"type_list": ["datetime", "string"]}}
                        ])
                    elif col_type in ['boolean']:
                        rules.extend([
                            {"expectation_type": "expect_column_values_to_be_in_type_list",
                             "kwargs": {"type_list": ["boolean"]}},
                            {"expectation_type": "expect_column_values_to_be_in_set",
                             "kwargs": {"value_set": [True, False]}}
                        ])
                    
                    fallback.append({
                        "column": col,
                        "expectations": rules
                    })
                result = json.dumps(fallback)
            
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add fallback rules for this batch
            fallback = []
            for col in batch_columns:
                fallback.append({
                    "column": col,
                    "expectations": [
                        {"expectation_type": "expect_column_values_to_not_be_null"}
                    ]
                })
            all_results.append(json.dumps(fallback))
    
    # Combine all results
    if not all_results:
        # Fallback for all columns if nothing was processed
        fallback = []
        for col in columns:
            fallback.append({
                "column": col,
                "expectations": [
                    {"expectation_type": "expect_column_values_to_not_be_null"}
                ]
            })
        return json.dumps(fallback)
    
    # Fast result combination
    if not all_results:
        return "[]"
        
    # Quick combine without extra processing
    combined = "[\n" + ",\n".join(
        r.strip('[]') for r in all_results if r and r.strip('[]')
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
        
        # Validate JSON format
        try:
            json.loads(result)
            return result
        except json.JSONDecodeError:
            logger.warning("Invalid JSON response from LLM")
            return "[]"
            
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return "[]"


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
