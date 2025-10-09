"""
Comprehensive Schema Suggester with both basic and enhanced functionality.

Combines:
1. Basic schema generation with Pydantic validation (original functionality)
2. Enhanced AI-powered schema generation with user preferences
3. Iterative refinement and style customization
4. Smart column conflict resolution and duplicate detection

Features:
- Basic domain schema generation for backward compatibility
- Enhanced schema generation with user preferences
- Style-based generation (minimal, standard, comprehensive)
- Iteration tracking and progressive refinement
- User preference filtering (exclude columns, include keywords)
- Fallback mechanisms and robust error handling
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from functools import lru_cache
from dataclasses import dataclass

# Pydantic and LangChain imports for structured output
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

# Retry and error handling
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# OpenAI async client for enhanced features
from openai import AsyncOpenAI

# App imports
from app.vector_db.schema_loader import validate_column_schema
from app.core.config import settings
from app.core.aws_secrets_service import require_openai_api_key
from app.core.exceptions import SchemaGenerationError
from app.validation.llm_validator import validate_llm_response, ValidationSeverity
from app.validation.metrics import record_validation_metric

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION AND DATA MODELS
# =============================================================================

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


# =============================================================================
# CACHING AND CONFIGURATION
# =============================================================================

# Cache for model chains to avoid recreation
_model_chain_cache = {}

@lru_cache(maxsize=1)
def get_schema_generation_config() -> SchemaGenerationConfig:
    """Get schema generation configuration with caching"""
    return SchemaGenerationConfig()


# =============================================================================
# BASIC SCHEMA GENERATION (ORIGINAL FUNCTIONALITY)
# =============================================================================

def get_model_chain(use_structured_output: bool = True) -> Runnable:
    """Get or create model chain with caching and error handling"""
    cache_key = f"{settings.schema_llm_model}_{settings.llm_temperature}_{use_structured_output}"
    
    if cache_key not in _model_chain_cache:
        try:
            # Get API key from AWS Secrets Manager
            openai_key = require_openai_api_key()
            
            model = ChatOpenAI(
                model=settings.schema_llm_model,
                temperature=settings.llm_temperature,
                openai_api_key=openai_key,
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
                 """Design a CSV schema for the domain: '{domain}'.

Requirements:
- Generate """ + f"{config.min_columns}-{config.max_columns}" + """ relevant columns
- Each column must have a valid identifier name (no spaces, special chars)
- Use data types: """ + f"{', '.join(config.supported_types)}" + """
- Provide exactly """ + f"{config.min_samples}" + """ realistic sample values per column
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
    logger.info(f" Calling LLM for domain: '{domain}' (structured={use_structured_output})")
    
    try:
        chain = get_model_chain(use_structured_output=use_structured_output)
        
        # Get format instructions for the chain
        if use_structured_output:
            from langchain.output_parsers import PydanticOutputParser
            parser = PydanticOutputParser(pydantic_object=SchemaResponse)
            format_instructions = parser.get_format_instructions()
        else:
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
        
        result = chain.invoke({
            "domain": domain,
            "format_instructions": format_instructions
        })
        
        elapsed_time = time.time() - start_time
        logger.info(f" LLM response received in {elapsed_time:.2f}s for domain '{domain}'")
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
        logger.error(f" LLM call failed after {elapsed_time:.2f}s for domain '{domain}': {e}")
        
        # If structured output fails, try fallback to JSON parser
        if use_structured_output:
            logger.warning(" Retrying with JSON parser fallback")
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
    logger.info(f" Formatting LLM response with {len(raw.get('columns', []))} columns")
    
    # Track validation timing for metrics
    validation_start_time = time.time()
    
    # First, validate the entire response using the new validation system
    validation_result = validate_llm_response(
        response=raw, 
        response_type="schema", 
        strict_mode=strict_validation,
        auto_correct=True
    )
    
    # Record validation metrics
    validation_time_ms = (time.time() - validation_start_time) * 1000
    domain = raw.get("domain", "unknown")
    try:
        record_validation_metric(
            domain=domain,
            response_type="schema",
            validation_result=validation_result,
            validation_time_ms=validation_time_ms
        )
    except Exception as e:
        logger.warning(f"Failed to record validation metrics: {e}")
    
    # Log validation results
    if validation_result.issues:
        logger.warning(f" Validation found {len(validation_result.issues)} issues:")
        for issue in validation_result.issues:
            logger.warning(f"   {issue.severity.value.upper()}: {issue.field} - {issue.message}")
    
    logger.info(f" Validation confidence score: {validation_result.confidence_score}")
    
    # Use corrected data if available and auto-correction was enabled
    working_data = validation_result.corrected_data if validation_result.corrected_data else raw
    
    # Check if validation failed critically
    if not validation_result.is_valid and strict_validation:
        critical_issues = [issue for issue in validation_result.issues if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            error_msg = f"Critical validation errors: {[issue.message for issue in critical_issues]}"
            logger.error(f" {error_msg}")
            raise SchemaGenerationError(error_msg)
    
    columns = working_data.get("columns", [])
    if not columns:
        raise SchemaGenerationError("No columns found in LLM response")
    
    formatted = {}
    validation_errors = []
    
    for i, col in enumerate(columns):
        col_name = col.get("name", f"unnamed_column_{i}")
        
        try:
            logger.debug(f" Validating column {i+1}/{len(columns)}: {col_name}")
            
            # Enhanced validation
            if not validate_column_schema(col):
                error_msg = f"Column '{col_name}' failed basic validation"
                validation_errors.append(error_msg)
                logger.warning(f" {error_msg}")
                if strict_validation:
                    continue
            
            # Additional validation checks
            col_type = col.get("type", "string").lower()
            samples = col.get("samples", [])
            
            # Validate sample values match type
            if not _validate_samples_for_type(samples, col_type):
                error_msg = f"Column '{col_name}': samples don't match type '{col_type}'"
                validation_errors.append(error_msg)
                logger.warning(f" {error_msg}")
                if strict_validation:
                    continue
            
            # Check for duplicate column names
            if col_name in formatted:
                error_msg = f"Duplicate column name: '{col_name}'"
                validation_errors.append(error_msg)
                logger.warning(f" {error_msg}")
                col_name = f"{col_name}_{i}"  # Make unique
            
            # Format the column
            formatted[col_name] = {
                "dtype": _normalize_data_type(col_type),
                "sample_values": samples[:3]  # Ensure max 3 samples
            }
            
            logger.debug(f" Column '{col_name}' validated and added")
            
        except Exception as e:
            error_msg = f"Error processing column '{col_name}': {e}"
            validation_errors.append(error_msg)
            logger.error(f" {error_msg}")
            if strict_validation:
                continue
    
    # Check minimum column requirements
    config = get_schema_generation_config()
    if len(formatted) < config.min_columns:
        error_summary = f"Only {len(formatted)} valid columns found (minimum {config.min_columns}). Errors: {validation_errors}"
        logger.error(f" {error_summary}")
        raise SchemaGenerationError(error_summary)
    
    logger.info(f" Successfully formatted schema with {len(formatted)} columns")
    if validation_errors:
        logger.warning(f" {len(validation_errors)} validation issues encountered")
    
    return formatted


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
        logger.info(f" Successfully bootstrapped schema for '{domain}' in {elapsed_time:.2f}s")
        logger.info(f" Generated {len(result)} columns: {list(result.keys())}")
        
        return result
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f" Schema bootstrapping failed for '{domain}' after {elapsed_time:.2f}s: {e}")
        
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
                logger.error(f" Fallback also failed: {fallback_error}")
        
        raise SchemaGenerationError(f"Failed to bootstrap schema for domain '{domain}': {e}")


# =============================================================================
# ENHANCED SCHEMA GENERATION WITH USER PREFERENCES
# =============================================================================

class SchemaSuggesterEnhanced:
    """Enhanced schema suggester with user preferences and iterative improvements."""
    
    def __init__(self):
        try:
            # Get API key from AWS Secrets Manager for enhanced features
            openai_key = require_openai_api_key()
            self.client = AsyncOpenAI(api_key=openai_key)
        except Exception:
            # Fall back to settings if AWS Secrets Manager fails
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"
    
    async def bootstrap_schema_with_preferences(
        self, 
        business_description: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate schema with enhanced user preferences and style options.
        
        Args:
            business_description: Description of the business domain
            user_preferences: Dictionary containing user customization options
                - style: "minimal", "standard", "comprehensive"
                - exclude_columns: List of column names to avoid
                - include_keywords: List of keywords to prioritize
                - column_count: Preferred number of columns
                - iteration: Iteration number for progressive refinement
        
        Returns:
            Enhanced schema with metadata about preferences applied
        """
        if user_preferences is None:
            user_preferences = {}
        
        # Extract preferences with defaults
        style = user_preferences.get("style", "standard")
        exclude_columns = user_preferences.get("exclude_columns", [])
        include_keywords = user_preferences.get("include_keywords", [])
        column_count = user_preferences.get("column_count", self._get_default_column_count(style))
        iteration = user_preferences.get("iteration", 1)
        
        # Build enhanced prompt based on preferences
        enhanced_prompt = self._build_enhanced_prompt(
            business_description=business_description,
            style=style,
            exclude_columns=exclude_columns,
            include_keywords=include_keywords,
            column_count=column_count,
            iteration=iteration
        )
        
        # Generate schema with dynamic temperature based on iteration
        temperature = self._calculate_temperature(iteration, style)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert business analyst and database designer. Generate JSON schema suggestions that are practical, well-structured, and aligned with business needs."
                    },
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                temperature=temperature,
                max_tokens=2000
            )
            
            # Parse and enhance the response
            raw_schema = self._parse_openai_response(response.choices[0].message.content)
            
            # Apply user preferences and enhancements
            enhanced_schema = self._apply_user_preferences(
                raw_schema=raw_schema,
                user_preferences=user_preferences,
                business_description=business_description
            )
            
            return enhanced_schema
            
        except Exception as e:
            logger.error(f"Error generating enhanced schema: {str(e)}")
            # Fallback to basic schema generation
            return await self._fallback_schema_generation(business_description, user_preferences)
    
    def _build_enhanced_prompt(
        self,
        business_description: str,
        style: str,
        exclude_columns: List[str],
        include_keywords: List[str],
        column_count: int,
        iteration: int
    ) -> str:
        """Build an enhanced prompt based on user preferences."""
        
        style_instructions = {
            "minimal": f"Generate exactly {column_count} essential columns focusing on core business data only. Prioritize simplicity and efficiency.",
            "standard": f"Generate {column_count} well-balanced columns covering essential business needs with reasonable detail.",
            "comprehensive": f"Generate {column_count} detailed columns providing extensive business insights and analytics capabilities."
        }
        
        prompt_parts = [
            f"Business Domain: {business_description}",
            "",
            f"Style: {style_instructions.get(style, style_instructions['standard'])}",
            ""
        ]
        
        if exclude_columns:
            prompt_parts.extend([
                f"EXCLUDE these column names or similar concepts: {', '.join(exclude_columns)}",
                ""
            ])
        
        if include_keywords:
            prompt_parts.extend([
                f"PRIORITIZE these concepts/keywords: {', '.join(include_keywords)}",
                ""
            ])
        
        if iteration > 1:
            prompt_parts.extend([
                f"This is iteration #{iteration}. Previous suggestions were not satisfactory.",
                "Focus on creative alternatives and fresh perspectives.",
                ""
            ])
        
        prompt_parts.extend([
            "Return a JSON object with this exact structure:",
            "{",
            '  "columns": [',
            '    {',
            '      "column_name": "string",',
            '      "type": "string|integer|float|boolean|date|datetime|text",',
            '      "description": "detailed business purpose",',
            '      "sample_values": ["value1", "value2", "value3"],',
            '      "business_relevance": "why this matters for the business"',
            '    }',
            '  ]',
            '}',
            "",
            "Ensure column names are business-friendly, descriptive, and follow snake_case convention."
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_default_column_count(self, style: str) -> int:
        """Get default column count based on style."""
        defaults = {
            "minimal": 5,
            "standard": 8,
            "comprehensive": 12
        }
        return defaults.get(style, 8)
    
    def _calculate_temperature(self, iteration: int, style: str) -> float:
        """Calculate OpenAI temperature based on iteration and style."""
        base_temperatures = {
            "minimal": 0.3,
            "standard": 0.5,
            "comprehensive": 0.7
        }
        
        base_temp = base_temperatures.get(style, 0.5)
        
        # Increase creativity for higher iterations
        iteration_boost = min((iteration - 1) * 0.1, 0.3)
        
        return min(base_temp + iteration_boost, 0.9)
    
    def _parse_openai_response(self, content: str) -> Dict[str, Any]:
        """Parse OpenAI response, handling various JSON formats."""
        try:
            # Clean the content
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Parse JSON
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Content: {content}")
            
            # Try to extract JSON from text
            return self._extract_json_from_text(content)
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON structure from mixed text content."""
        try:
            # Find JSON-like structure
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            
            # Fallback: return empty structure
            return {"columns": []}
            
        except Exception as e:
            logger.error(f"JSON extraction error: {str(e)}")
            return {"columns": []}
    
    def _apply_user_preferences(
        self,
        raw_schema: Dict[str, Any],
        user_preferences: Dict[str, Any],
        business_description: str
    ) -> Dict[str, Any]:
        """Apply user preferences and add enhancement metadata."""
        
        columns = raw_schema.get("columns", [])
        exclude_columns = user_preferences.get("exclude_columns", [])
        include_keywords = user_preferences.get("include_keywords", [])
        style = user_preferences.get("style", "standard")
        iteration = user_preferences.get("iteration", 1)
        
        # Filter out excluded columns
        filtered_columns = []
        for column in columns:
            column_name = column.get("column_name", "").lower()
            
            # Check if column should be excluded
            should_exclude = any(
                excluded.lower() in column_name or column_name in excluded.lower()
                for excluded in exclude_columns
            )
            
            if not should_exclude:
                filtered_columns.append(column)
        
        # Score and sort columns based on keyword preferences
        if include_keywords:
            for column in filtered_columns:
                column["preference_score"] = self._calculate_preference_score(
                    column, include_keywords
                )
            
            # Sort by preference score (highest first)
            filtered_columns.sort(
                key=lambda x: x.get("preference_score", 0), 
                reverse=True
            )
        
        # Apply column count limit
        target_count = user_preferences.get("column_count")
        if target_count and len(filtered_columns) > target_count:
            filtered_columns = filtered_columns[:target_count]
        
        # Generate enhancement tips
        next_iteration_tips = self._generate_iteration_tips(
            columns=filtered_columns,
            user_preferences=user_preferences,
            business_description=business_description
        )
        
        style_recommendations = self._generate_style_recommendations(
            current_style=style,
            column_count=len(filtered_columns)
        )
        
        # Build enhanced response
        enhanced_schema = {
            "columns": filtered_columns,
            "total_columns": len(filtered_columns),
            "style": style,
            "iteration": iteration,
            "preferences_applied": {
                "excluded_columns": len(columns) - len(filtered_columns),
                "keyword_filtering": bool(include_keywords),
                "style_customization": style != "standard",
                "iteration_enhancement": iteration > 1
            },
            "next_iteration_tips": next_iteration_tips,
            "style_recommendations": style_recommendations,
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "temperature_used": self._calculate_temperature(iteration, style),
                "business_description_hash": hash(business_description) % 100000
            }
        }
        
        return enhanced_schema
    
    def _calculate_preference_score(self, column: Dict[str, Any], keywords: List[str]) -> float:
        """Calculate preference score based on keyword matching."""
        score = 0.0
        
        column_text = " ".join([
            column.get("column_name", ""),
            column.get("description", ""),
            column.get("business_relevance", ""),
            " ".join(str(v) for v in column.get("sample_values", []))
        ]).lower()
        
        for keyword in keywords:
            keyword = keyword.lower()
            if keyword in column_text:
                # Higher score for exact matches in column name
                if keyword in column.get("column_name", "").lower():
                    score += 3.0
                # Medium score for description/relevance matches
                elif keyword in column.get("description", "").lower():
                    score += 2.0
                # Lower score for sample value matches
                else:
                    score += 1.0
        
        return score
    
    def _generate_iteration_tips(
        self,
        columns: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
        business_description: str
    ) -> List[str]:
        """Generate tips for the next iteration."""
        tips = []
        
        current_style = user_preferences.get("style", "standard")
        exclude_columns = user_preferences.get("exclude_columns", [])
        
        # Style-specific tips
        if current_style == "minimal":
            tips.append("Try 'standard' style for more detailed business insights")
        elif current_style == "comprehensive":
            tips.append("Try 'minimal' style for essential columns only")
        else:
            tips.append("Try 'comprehensive' style for advanced analytics columns")
        
        # Column exclusion tips
        if len(exclude_columns) > 3:
            tips.append("Consider reducing exclusions to get more diverse suggestions")
        elif not exclude_columns:
            tips.append("Add specific column exclusions to avoid unwanted suggestions")
        
        # Business focus tips
        if "analytics" not in business_description.lower():
            tips.append("Include 'analytics' keywords for better reporting columns")
        
        if "compliance" not in business_description.lower():
            tips.append("Add 'compliance' focus for regulatory tracking columns")
        
        return tips[:3]  # Limit to 3 tips
    
    def _generate_style_recommendations(self, current_style: str, column_count: int) -> List[str]:
        """Generate style recommendations based on current selection."""
        recommendations = []
        
        if current_style == "minimal" and column_count < 7:
            recommendations.append("Current minimal style works well for focused domains")
        elif current_style == "comprehensive" and column_count > 10:
            recommendations.append("Comprehensive style provides extensive business coverage")
        else:
            recommendations.append("Standard style offers balanced business insights")
        
        # Add suggestions for other styles
        if current_style != "minimal":
            recommendations.append("Try minimal style for core business essentials")
        
        if current_style != "comprehensive":
            recommendations.append("Try comprehensive style for advanced analytics")
        
        return recommendations
    
    async def _fallback_schema_generation(
        self,
        business_description: str,
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback to basic schema generation if enhanced generation fails."""
        try:
            # Use the existing basic schema generator
            basic_schema = bootstrap_schema_for_domain(business_description)
            
            # Convert to enhanced format
            columns = []
            for col_name, col_info in basic_schema.items():
                columns.append({
                    "column_name": col_name,
                    "type": col_info.get("dtype", "string"),
                    "description": f"Basic schema column: {col_name}",
                    "sample_values": col_info.get("sample_values", []),
                    "business_relevance": f"Standard business field for {business_description}"
                })
            
            # Add basic enhancement metadata
            return {
                "columns": columns,
                "total_columns": len(columns),
                "style": user_preferences.get("style", "standard"),
                "iteration": user_preferences.get("iteration", 1),
                "preferences_applied": {
                    "fallback_used": True,
                    "enhanced_generation": False
                },
                "next_iteration_tips": ["Try again with simpler preferences"],
                "style_recommendations": ["Standard style recommended for fallback"],
                "metadata": {
                    "fallback_mode": True,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback schema generation failed: {str(e)}")
            return {
                "columns": [],
                "total_columns": 0,
                "style": "standard",
                "iteration": 1,
                "preferences_applied": {"error": True},
                "next_iteration_tips": ["Please try again with a simpler business description"],
                "style_recommendations": [],
                "metadata": {"error": True}
            }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

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
    logger.info(" Model cache cleared")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def generate_enhanced_schema(
    business_description: str,
    style: str = "standard",
    column_count: Optional[int] = None,
    exclude_columns: Optional[List[str]] = None,
    include_keywords: Optional[List[str]] = None,
    iteration: int = 1
) -> Dict[str, Any]:
    """
    Convenience function for generating enhanced schemas with common parameters.
    
    Args:
        business_description: Description of the business domain
        style: "minimal", "standard", or "comprehensive"
        column_count: Number of columns to generate (optional)
        exclude_columns: Column names to avoid (optional)
        include_keywords: Keywords to prioritize (optional) 
        iteration: Iteration number for refinement
        
    Returns:
        Enhanced schema with metadata
    """
    suggester = SchemaSuggesterEnhanced()
    
    user_preferences = {
        "style": style,
        "iteration": iteration
    }
    
    if column_count is not None:
        user_preferences["column_count"] = column_count
    if exclude_columns:
        user_preferences["exclude_columns"] = exclude_columns
    if include_keywords:
        user_preferences["include_keywords"] = include_keywords
    
    return await suggester.bootstrap_schema_with_preferences(
        business_description=business_description,
        user_preferences=user_preferences
    )