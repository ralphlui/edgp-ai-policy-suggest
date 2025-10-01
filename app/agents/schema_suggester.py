from typing import Dict, Any
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from app.vector_db.schema_loader import validate_column_schema
from app.core.config import OPENAI_API_KEY
import logging

logger = logging.getLogger(__name__)

# Initialize model and parser lazily
def get_model_chain() -> Runnable:
    """Lazy initialization of ChatOpenAI model and chain"""
    model = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    parser = JsonOutputParser()
    
    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data architect helping define CSV schemas for new domains."),
        ("human", """Suggest 5–11 plausible CSV columns for a domain called '{domain}'. 
For each column, include:
- name
- data type (e.g., string, integer, date)
- 3 sample values

Respond in JSON format:
{{
  "columns": [
    {{
      "name": "email",
      "type": "string",
      "samples": ["alice@example.com", "bob@example.com", "carol@example.com"]
    }},
    ...
  ]
}}""")
    ])
    
    # Chain: prompt → model → parser
    return prompt | model | parser

def call_llm(domain: str) -> Dict[str, Any]:
    """
    Invoke LLM to suggest schema for a domain.
    """
    logger.info(f"Calling LLM for domain: {domain}")
    chain = get_model_chain()  # Lazy initialization
    result = chain.invoke({ "domain": domain })
    logger.info(f"LLM raw response: {result}")
    return result


def format_llm_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert LLM output into agent-compatible schema format.
    Filters out invalid columns using validation layer.
    """
    logger.info(f"Formatting LLM response: {raw}")
    columns = raw.get("columns", [])
    logger.info(f"Found {len(columns)} columns to validate")
    
    formatted = {}
    for col in columns:
        logger.info(f"Validating column: {col}")
        if validate_column_schema(col):
            formatted[col["name"]] = {
                "dtype": col["type"],
                "sample_values": col["samples"]
            }
            logger.info(f"Column {col['name']} validated successfully")
        else:
            logger.warning(f"Column {col} failed validation")
    
    logger.info(f"Final formatted schema: {formatted}")
    return formatted


def bootstrap_schema_for_domain(domain: str) -> Dict[str, Any]:
    """
    Public entry point: generate and format schema for a new domain.
    """
    logger.info(f"Bootstrapping schema for domain: {domain}")
    raw = call_llm(domain)
    result = format_llm_schema(raw)
    logger.info(f"Bootstrap result: {result}")
    return result

def bootstrap_schema(domain: str) -> Dict[str, Any]:
    """
    Alias for bootstrap_schema_for_domain - for backward compatibility with tests
    """
    return bootstrap_schema_for_domain(domain)
