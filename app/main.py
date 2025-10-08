from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.api.domain_schema_routes import router as domain_schema_router
from app.api.rule_suggestion_routes import router as rule_suggestion_router
from app.api.routes import router as vector_router
from app.core.exceptions import (
    authentication_exception_handler,
    general_exception_handler,
    validation_exception_handler,
    internal_server_error_handler
)
import time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EDGP AI Policy Suggest Microservice", 
    version="1.0",
    description="AI-powered data quality policy and rule suggestion microservice"
)

# Add exception handlers for standardized responses
app.add_exception_handler(HTTPException, authentication_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, internal_server_error_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"{request.method} {request.url} - {response.status_code} - {time.time() - start:.2f}s")
    return response

@app.get("/api/aips/health")
def health():
    """Enhanced health check with AWS connection status"""
    from app.api.routes import get_store
    
    health_status = {
        "service_name": "EDGP AI Policy Suggest Microservice",
        "version": "1.0",
        "status": "ok",
        "timestamp": time.time(),
        "services": {
            "fastapi": "healthy",
            "opensearch": "unknown"
        }
    }
    
    # Test OpenSearch connection
    try:
        store = get_store()
        if store is None:
            health_status["services"]["opensearch"] = "unavailable"
            health_status["opensearch_message"] = "Store initialization failed - likely AWS permission issues"
        else:
            # Try a simple operation
            try:
                store.client.info()
                health_status["services"]["opensearch"] = "healthy"
            except Exception as e:
                health_status["services"]["opensearch"] = "error"
                health_status["opensearch_error"] = str(e)[:100]  # Truncate error
    except Exception as e:
        health_status["services"]["opensearch"] = "error"
        health_status["opensearch_error"] = str(e)[:100]
    
    # Overall status
    if health_status["services"]["opensearch"] != "healthy":
        health_status["status"] = "degraded"
        health_status["message"] = "Some services are unavailable"
    
    return health_status

@app.get("/api/aips/info")
def service_info():
    """Service information endpoint"""
    from app.api.routes import get_store
    info = {
        "service_name": "EDGP AI Policy Suggest Microservice",
        "version": "1.0",
        "description": "AI-powered data quality policy and rule suggestion microservice",
        "endpoints": {
            "health": {
                "method": "GET",
                "path": "/api/aips/health",
                "description": "Health check with OpenSearch status"
            },
            "info": {
                "method": "GET",
                "path": "/api/aips/info",
                "description": "Service information and live vector DB status"
            },
            "suggest_rules": {
                "method": "POST",
                "path": "/api/aips/suggest-rules",
                "description": "Suggest validation rules for a domain"
            },
            "create_domain": {
                "method": "POST",
                "path": "/api/aips/domain/create",
                "description": "Create a new domain with columns"
            },
            "extend_domain": {
                "method": "PUT",
                "path": "/api/aips/domain/extend-schema",
                "description": "Extend an existing domain with new columns"
            },
            "suggest_extend_schema": {
                "method": "POST",
                "path": "/api/aips/domain/suggest-extend-schema/{domain_name}",
                "description": "Suggest additional columns for an existing domain"
            },
            "suggest_schema": {
                "method": "POST",
                "path": "/api/aips/domain/suggest-schema",
                "description": "AI-powered domain schema suggestions"
            },
            "vector_status": {
                "method": "GET",
                "path": "/api/aips/vector/status",
                "description": "Check vector database connection and index status"
            },
            "domains": {
                "method": "GET",
                "path": "/api/aips/domains",
                "description": "Get all available domains"
            },
            "domain_details": {
                "method": "GET",
                "path": "/api/aips/domain/{domain_name}",
                "description": "Get details for a specific domain"
            }
        },
        "repository": "edgp-ai-policy-suggest",
        "branch": "task/llm-validation"
    }
    # Live vector DB status
    try:
        store = get_store()
        if store is not None:
            client = store.client
            index_name = store.index_name
            index_exists = client.indices.exists(index=index_name)
            info["vector_db"] = {
                "index_name": index_name,
                "index_exists": index_exists
            }
            if index_exists:
                try:
                    stats = client.indices.stats(index=index_name)
                    doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
                    info["vector_db"]["document_count"] = doc_count
                except Exception as stats_error:
                    info["vector_db"]["document_count"] = "unknown"
                    info["vector_db"]["stats_error"] = str(stats_error)
            # Get domain list
            try:
                domains = store.get_all_domains_realtime(force_refresh=True)
                info["domain_count"] = len(domains)
                info["domains"] = domains
            except Exception as e:
                info["domain_count"] = "unknown"
                info["domains_error"] = str(e)
        else:
            info["vector_db"] = {"status": "unavailable"}
    except Exception as e:
        info["vector_db"] = {"status": "error", "error": str(e)}
    return info

app.include_router(domain_schema_router)
app.include_router(rule_suggestion_router)
app.include_router(vector_router)

if __name__ == "__main__":
    import uvicorn
    import os
    from app.core.config import settings
    
    # Get configuration from settings (which reads from .env files)
    host = settings.host  # Use host from settings (.env files)
    port = settings.port  # Use port from settings (.env files)
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(" Starting EDGP AI Policy Suggest Microservice...")
    logger.info(f" Server will run on http://{host}:{port}")
    logger.info(f" Server accessible from all IP addresses")
    logger.info(f" Log level: {log_level}")
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
        use_colors=True
    )
