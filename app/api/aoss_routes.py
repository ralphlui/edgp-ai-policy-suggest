# Vector DB status endpoint for test and API use
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.aoss.column_store import OpenSearchColumnStore
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Lazy initialization of OpenSearch store to prevent startup failures
_store = None

def get_store() -> OpenSearchColumnStore:
    """Get OpenSearch store with lazy initialization and error handling"""
    global _store
    if _store is None:
        try:
            _store = OpenSearchColumnStore(index_name=settings.opensearch_index)
            logger.info(" OpenSearch store initialized successfully")
        except Exception as e:
            logger.error(f" Failed to initialize OpenSearch store: {e}")
            logger.error("This is likely due to AWS permission issues. See AWS_ADMIN_SETUP_GUIDE.md")
            return None
    return _store

async def check_vectordb_status():
    """Check vector database connection and index status with validation metrics"""
    try:
        store = get_store()
        if store is None:
            return JSONResponse({
                "status": "error",
                "message": "OpenSearch store not available",
                "connection": "failed",
                "validation_status": "unavailable"
            }, status_code=503)
        
        client = store.client
        index_name = store.index_name
        index_exists = client.indices.exists(index=index_name)
        
        result = {
            "status": "connected",
            "index_name": index_name,
            "index_exists": index_exists,
            "validation_status": "available"
        }
        
        if index_exists:
            try:
                stats = client.indices.stats(index=index_name)
                doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
                result["document_count"] = doc_count
            except Exception as stats_error:
                result["document_count"] = "unknown"
                result["stats_error"] = str(stats_error)
        
        # Add validation system status
        try:
            from app.validation.metrics import ValidationMetrics
            metrics = ValidationMetrics.get_current_metrics()
            result["validation_metrics"] = {
                "total_validations": metrics.total_validations,
                "success_rate": metrics.success_rate,
                "last_validation": metrics.last_updated.isoformat() if metrics.last_updated else None
            }
        except Exception as validation_error:
            result["validation_metrics"] = {"error": str(validation_error)}
        
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "connection": "failed",
            "validation_status": "error"
        }, status_code=500)

# Explicitly export direct route functions for test imports
check_vectordb_status = check_vectordb_status
router = APIRouter(prefix="/api/aips/vector", tags=["aoss"])

@router.get("/status")
async def get_vectordb_status():
    """Endpoint for vector database status"""
    return await check_vectordb_status()

# Re-export endpoint functions for test imports
from app.api.domain_schema_routes import (
    create_domain, get_domains, verify_domain_exists,
    list_domains_in_vectordb, get_domain_from_vectordb,
    download_csv_file, regenerate_suggestions, extend_domain, suggest_extensions, get_store
)
try:
    from app.api.rule_suggestion_routes import suggest_rules
except ImportError:
    pass

