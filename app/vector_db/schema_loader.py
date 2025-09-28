from app.aoss.column_store import OpenSearchColumnStore
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Lazy initialization to prevent startup failures
_store = None

def get_store():
    """Get OpenSearch store with lazy initialization"""
    global _store
    if _store is None:
        try:
            _store = OpenSearchColumnStore(index_name=settings.column_index_name)
            logger.info(" Schema loader: OpenSearch store initialized")
        except Exception as e:
            logger.warning(f" Schema loader: Failed to initialize OpenSearch store: {e}")
            return None
    return _store

def get_schema_by_domain(domain: str) -> dict:
    """
    Retrieve column metadata for a given domain and format it for agent input.
    """
    store = get_store()
    if store is None:
        logger.warning("OpenSearch store not available, returning empty schema")
        return {}
    
    try:
        # Use the new direct domain search method instead of semantic search
        results = store.get_columns_by_domain(
            domain=domain,
            return_fields=["column_name", "metadata", "sample_values"]
        )

        if not results:
            return {}

        return {
            col["column_name"]: {
                "dtype": col["metadata"].get("type", "unknown"),
                "sample_values": col.get("sample_values", [])
            }
            for col in results
        }
    except Exception as e:
        logger.error(f"Error querying OpenSearch for domain {domain}: {e}")
        return {}

def validate_column_schema(col: dict) -> bool:
    """
    Validates a single column schema suggestion from LLM.
    Ensures:
    - name is a valid identifier
    - type is one of the allowed types
    - samples is a non-empty list
    """
    name = col.get("name")
    dtype = col.get("type")
    samples = col.get("samples")

    return (
        isinstance(name, str) and name.isidentifier() and
        dtype in {"string", "integer", "float", "date", "boolean"} and
        isinstance(samples, list) and len(samples) >= 3
    )
