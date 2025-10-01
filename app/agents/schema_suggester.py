from typing import Dict, Any, List, Optional, Union
from functools import lru_cache
from dataclasses import dataclass
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.vector_db.schema_loader import validate_column_schema
from app.core.config import OPENAI_API_KEY, settings
from app.core.exceptions import SchemaGenerationError
import logging
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class SchemaGenerationConfig:
    """Configuration for schema generation"""
    min_columns: int = 5
    max_columns: int = 11
    min_samples: int = 3
    max_retries: int = 3
    timeout_seconds: int = 30
    supported_types: List[str] = None
    
    def __post_init__(self):
        if self.supported_types is None:
            self.supported_types = ["string", "integer", "float", "date", "boolean", "text"]


class ColumnSchema(BaseModel):
    """Pydantic model for column schema validation"""
    name: str = Field(..., description="Column name as valid identifier")
    type: str = Field(..., description="Data type")
    samples: List[str] = Field(..., min_length=3, description="Sample values")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.isidentifier():
            raise ValueError(f"Column name '{v}' is not a valid identifier")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        allowed_types = ["string", "integer", "float", "date", "boolean", "text"]
        if v not in allowed_types:
            raise ValueError(f"Type '{v}' not in allowed types: {allowed_types}")
        return v


class SchemaResponse(BaseModel):
    """Pydantic model for LLM schema response"""
    columns: List[ColumnSchema] = Field(..., min_length=5, max_length=11)


# Cache for model chains to avoid recreation
_model_chain_cache = {}


@lru_cache(maxsize=1)
def get_schema_generation_config() -> SchemaGenerationConfig:
    """Get schema generation configuration with caching"""
    return SchemaGenerationConfig()


def get_model_chain(use_structured_output: bool = True) -> Runnable:
    """Get or create model chain with caching and error handling"""
    cache_key = f"{settings.schema_llm_model}_{settings.llm_temperature}_{use_structured_output}"
    
    if cache_key not in _model_chain_cache:
        try:
            # Validate API key exists
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured")
            
            model = ChatOpenAI(
                model=settings.schema_llm_model,
                temperature=settings.llm_temperature,
                openai_api_key=OPENAI_API_KEY,
                timeout=get_schema_generation_config().timeout_seconds,
                max_retries=2
            )
            
            # Choose parser based on configuration
            if use_structured_output:
                parser = PydanticOutputParser(pydantic_object=SchemaResponse)
                format_instructions = parser.get_format_instructions()
            else:
                parser = JsonOutputParser()
                format_instructions = """
Respond in JSON format:
{
  "columns": [
    {
      "name": "column_name",
      "type": "data_type",
      "samples": ["sample1", "sample2", "sample3"]
    }
  ]
}"""
            
            config = get_schema_generation_config()
            
            # Enhanced prompt template with better instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are an expert data architect specializing in CSV schema design. "
                 "You create realistic, production-ready column definitions for business domains. "
                 "Focus on commonly used columns that would appear in real-world datasets."),
                ("human", 
                 f"""Design a CSV schema for the domain: '{{"domain"}}'.

Requirements:
- Generate {config.min_columns}-{config.max_columns} relevant columns
- Each column must have a valid identifier name (no spaces, special chars)
- Use data types: {', '.join(config.supported_types)}
- Provide exactly {config.min_samples} realistic sample values per column
- Include common business fields (IDs, names, dates, status, etc.)
- Ensure samples are diverse and realistic

{format_instructions}""")
            ])
            
            # Create and cache the chain
            chain = prompt | model | parser
            _model_chain_cache[cache_key] = chain
            logger.info(f"Created new model chain for {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to create model chain: {e}")
            raise SchemaGenerationError(f"Model initialization failed: {e}")
    
    return _model_chain_cache[cache_key]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError))
)
def call_llm(domain: str, use_structured_output: bool = True) -> Dict[str, Any]:
    """
    Invoke LLM to suggest schema for a domain with retry logic and error handling.
    
    Args:
        domain: The domain name to generate schema for
        use_structured_output: Whether to use Pydantic structured output
        
    Returns:
        Dict containing the schema response
        
    Raises:
        SchemaGenerationError: If schema generation fails after retries
    """
    if not domain or not isinstance(domain, str):
        raise ValueError("Domain must be a non-empty string")
    
    domain = domain.strip().lower()
    if not domain:
        raise ValueError("Domain cannot be empty after normalization")
    
    start_time = time.time()
    logger.info(f"🔄 Calling LLM for domain: '{domain}' (structured={use_structured_output})")
    
    try:
        chain = get_model_chain(use_structured_output=use_structured_output)
        result = chain.invoke({"domain": domain})
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ LLM response received in {elapsed_time:.2f}s for domain '{domain}'")
        logger.debug(f"LLM raw response: {result}")
        
        # Validate response structure
        if use_structured_output:
            if not isinstance(result, SchemaResponse):
                raise ValueError(f"Expected SchemaResponse, got {type(result)}")
            # Convert Pydantic model to dict
            return result.dict()
        else:
            if not isinstance(result, dict) or "columns" not in result:
                raise ValueError("Response missing 'columns' key")
            return result
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ LLM call failed after {elapsed_time:.2f}s for domain '{domain}': {e}")
        
        # If structured output fails, try fallback to JSON parser
        if use_structured_output:
            logger.warning("🔄 Retrying with JSON parser fallback")
            return call_llm(domain, use_structured_output=False)
        
        raise SchemaGenerationError(f"LLM call failed for domain '{domain}': {e}")


def format_llm_schema(raw: Dict[str, Any], strict_validation: bool = True) -> Dict[str, Any]:
    """
    Convert LLM output into agent-compatible schema format with enhanced validation.
    
    Args:
        raw: Raw LLM response containing columns
        strict_validation: Whether to apply strict validation rules
        
    Returns:
        Formatted schema dictionary
        
    Raises:
        SchemaGenerationError: If no valid columns found
    """
    logger.info(f"📋 Formatting LLM response with {len(raw.get('columns', []))} columns")
    
    columns = raw.get("columns", [])
    if not columns:
        raise SchemaGenerationError("No columns found in LLM response")
    
    formatted = {}
    validation_errors = []
    
    for i, col in enumerate(columns):
        col_name = col.get("name", f"unnamed_column_{i}")
        
        try:
            logger.debug(f"🔍 Validating column {i+1}/{len(columns)}: {col_name}")
            
            # Enhanced validation
            if not validate_column_schema(col):
                error_msg = f"Column '{col_name}' failed basic validation"
                validation_errors.append(error_msg)
                logger.warning(f"⚠️ {error_msg}")
                if strict_validation:
                    continue
            
            # Additional validation checks
            col_type = col.get("type", "string").lower()
            samples = col.get("samples", [])
            
            # Validate sample values match type
            if not _validate_samples_for_type(samples, col_type):
                error_msg = f"Column '{col_name}': samples don't match type '{col_type}'"
                validation_errors.append(error_msg)
                logger.warning(f"⚠️ {error_msg}")
                if strict_validation:
                    continue
            
            # Check for duplicate column names
            if col_name in formatted:
                error_msg = f"Duplicate column name: '{col_name}'"
                validation_errors.append(error_msg)
                logger.warning(f"⚠️ {error_msg}")
                col_name = f"{col_name}_{i}"  # Make unique
            
            # Format the column
            formatted[col_name] = {
                "dtype": _normalize_data_type(col_type),
                "sample_values": samples[:3]  # Ensure max 3 samples
            }
            
            logger.debug(f"✅ Column '{col_name}' validated and added")
            
        except Exception as e:
            error_msg = f"Error processing column '{col_name}': {e}"
            validation_errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            if strict_validation:
                continue
    
    # Check minimum column requirements
    config = get_schema_generation_config()
    if len(formatted) < config.min_columns:
        error_summary = f"Only {len(formatted)} valid columns found (minimum {config.min_columns}). Errors: {validation_errors}"
        logger.error(f"❌ {error_summary}")
        raise SchemaGenerationError(error_summary)
    
    logger.info(f"✅ Successfully formatted schema with {len(formatted)} columns")
    if validation_errors:
        logger.warning(f"⚠️ {len(validation_errors)} validation issues encountered")
    
    return formatted


def _validate_samples_for_type(samples: List[str], col_type: str) -> bool:
    """
    Validate that sample values are reasonable for the given data type.
    """
    if not samples or len(samples) < 3:
        return False
    
    try:
        if col_type in ["integer", "int"]:
            return all(str(sample).strip().lstrip('-').isdigit() for sample in samples)
        elif col_type in ["float", "decimal", "number"]:
            return all(_is_valid_float(str(sample)) for sample in samples)
        elif col_type in ["boolean", "bool"]:
            return all(str(sample).lower() in ["true", "false", "0", "1", "yes", "no"] for sample in samples)
        elif col_type == "date":
            return all(len(str(sample)) >= 8 for sample in samples)  # Basic date length check
        else:  # string, text, etc.
            return all(isinstance(sample, (str, int, float)) for sample in samples)
    except Exception:
        return False


def _is_valid_float(value: str) -> bool:
    """Check if string can be converted to float"""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _normalize_data_type(col_type: str) -> str:
    """
    Normalize data type names to standard format.
    """
    type_mapping = {
        "int": "integer",
        "bool": "boolean",
        "str": "string",
        "text": "string",
        "number": "float",
        "decimal": "float"
    }
    return type_mapping.get(col_type.lower(), col_type.lower())


def bootstrap_schema_for_domain(
    domain: str, 
    use_structured_output: bool = True,
    strict_validation: bool = True,
    fallback_on_error: bool = True
) -> Dict[str, Any]:
    """
    Public entry point: generate and format schema for a new domain.
    
    Args:
        domain: Domain name to generate schema for
        use_structured_output: Whether to use Pydantic structured output
        strict_validation: Whether to apply strict validation rules
        fallback_on_error: Whether to retry with relaxed settings on failure
        
    Returns:
        Dictionary containing formatted schema
        
    Raises:
        SchemaGenerationError: If schema generation fails
    """
    if not domain:
        raise ValueError("Domain cannot be empty")
    
    start_time = time.time()
    logger.info(f"🚀 Bootstrapping schema for domain: '{domain}'")
    
    try:
        # Generate schema with LLM
        raw = call_llm(domain, use_structured_output=use_structured_output)
        
        # Format and validate schema
        result = format_llm_schema(raw, strict_validation=strict_validation)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Successfully bootstrapped schema for '{domain}' in {elapsed_time:.2f}s")
        logger.info(f"📊 Generated {len(result)} columns: {list(result.keys())}")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Schema bootstrapping failed for '{domain}' after {elapsed_time:.2f}s: {e}")
        
        # Fallback with relaxed validation
        if fallback_on_error and strict_validation:
            logger.warning("🔄 Retrying with relaxed validation...")
            try:
                return bootstrap_schema_for_domain(
                    domain, 
                    use_structured_output=False,
                    strict_validation=False,
                    fallback_on_error=False
                )
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
        
        raise SchemaGenerationError(f"Failed to bootstrap schema for domain '{domain}': {e}")


def bootstrap_schema(domain: str) -> Dict[str, Any]:
    """
    Alias for bootstrap_schema_for_domain - for backward compatibility with tests.
    Uses default settings for compatibility.
    """
    return bootstrap_schema_for_domain(domain)


def validate_schema_completeness(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that a generated schema meets completeness requirements.
    
    Args:
        schema: Generated schema dictionary
        
    Returns:
        Validation report with status and issues
    """
    config = get_schema_generation_config()
    issues = []
    
    # Check column count
    column_count = len(schema)
    if column_count < config.min_columns:
        issues.append(f"Too few columns: {column_count} < {config.min_columns}")
    elif column_count > config.max_columns:
        issues.append(f"Too many columns: {column_count} > {config.max_columns}")
    
    # Check for essential columns (common business patterns)
    essential_patterns = ["id", "name", "email", "date", "status", "created"]
    found_patterns = sum(1 for pattern in essential_patterns 
                        if any(pattern in col_name.lower() for col_name in schema.keys()))
    
    if found_patterns < 2:
        issues.append(f"Missing common business patterns (found {found_patterns}/6)")
    
    # Check data type diversity
    types_used = set(col_info.get("dtype", "unknown") for col_info in schema.values())
    if len(types_used) < 3:
        issues.append(f"Limited data type diversity: {types_used}")
    
    return {
        "is_valid": len(issues) == 0,
        "column_count": column_count,
        "types_used": list(types_used),
        "issues": issues
    }


def clear_model_cache() -> None:
    """
    Clear the model chain cache. Useful for testing or configuration changes.
    """
    global _model_chain_cache
    _model_chain_cache.clear()
    get_schema_generation_config.cache_clear()
    logger.info("🧹 Model cache cleared")
