from fastapi import APIRouter, HTTPException, Body, Depends, Request
from fastapi.responses import JSONResponse
from app.agents.agent_runner import run_agent
from app.agents.schema_suggester import bootstrap_schema_for_domain
from app.vector_db.schema_loader import get_schema_by_domain
from app.aoss.column_store import OpenSearchColumnStore
from app.core.config import settings
from app.auth.authentication import verify_any_scope_token, UserInfo
import traceback
import logging
import time
from app.aoss.column_store import ColumnDoc
from app.embedding.embedder import embed_column_names_batched_async
from fastapi.responses import StreamingResponse
from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd, io

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

router = APIRouter()


@router.get("/api/aips/vector/status")
async def check_vectordb_status():
    """Check vector database connection and index status."""
    try:
        store = get_store()
        if store is None:
            return JSONResponse({
                "status": "error",
                "message": "OpenSearch store not available",
                "connection": "failed"
            }, status_code=503)
        
        # Test connection and get index info
        client = store.client
        index_name = store.index_name
        
        # Check if index exists
        index_exists = client.indices.exists(index=index_name)
        
        result = {
            "status": "connected",
            "index_name": index_name,
            "index_exists": index_exists,
            "connection": "success"
        }
        
        if index_exists:
            # Get index stats
            try:
                stats = client.indices.stats(index=index_name)
                result["document_count"] = stats["indices"][index_name]["total"]["docs"]["count"]
                result["index_size"] = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]
            except Exception as stats_error:
                logger.warning(f"Could not get index stats: {stats_error}")
                result["document_count"] = "unknown"
                result["index_size"] = "unknown"
                result["stats_error"] = str(stats_error)
        else:
            result["document_count"] = 0
            result["index_size"] = 0
            result["note"] = "Index does not exist yet. It will be created when first data is added."
        
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse({
            "status": "error", 
            "message": str(e),
            "connection": "failed"
        }, status_code=500)


