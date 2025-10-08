# Vector DB status endpoint for test and API use
from fastapi import Depends
from fastapi.responses import JSONResponse

async def check_vectordb_status():
    """Check vector database connection and index status"""
    try:
        store = get_store()
        if store is None:
            return JSONResponse({
                "status": "error",
                "message": "OpenSearch store not available",
                "connection": "failed"
            }, status_code=503)
        client = store.client
        index_name = store.index_name
        index_exists = client.indices.exists(index=index_name)
        result = {
            "status": "connected",
            "index_name": index_name,
            "index_exists": index_exists
        }
        if index_exists:
            try:
                stats = client.indices.stats(index=index_name)
                doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
                result["document_count"] = doc_count
            except Exception as stats_error:
                result["document_count"] = "unknown"
                result["stats_error"] = str(stats_error)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "connection": "failed"
        }, status_code=500)

# Explicitly export direct route functions for test imports
check_vectordb_status = check_vectordb_status
# Explicitly export direct route functions for test imports
check_vectordb_status = check_vectordb_status
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
        

        # Test connection and get index info

