from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from app.api.domain_schema_routes import router as domain_schema_router
from app.api.rule_suggestion_routes import router as rule_suggestion_router
from app.api.aoss_routes import router as vector_router
from app.exception.exceptions import (
    authentication_exception_handler,
    general_exception_handler,
    validation_exception_handler,
    internal_server_error_handler
)
from app.aws.audit_middleware import add_audit_middleware
from app.aws.audit_service import audit_system_health
import time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EDGP AI Policy Suggest Microservice", 
    version="1.0",
    description="AI-powered data quality policy and rule suggestion microservice"
)

# Add exception handlers for standardized responses
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, internal_server_error_handler)

# Specific HTTP exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions based on status code"""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    # Use authentication handler for 401 and 403
    if exc.status_code in {401, 403}:
        return await authentication_exception_handler(request, exc)
    
    # Use general handler for all other status codes
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": exc.detail,
            "success": False,
            "status": exc.status_code
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add audit logging middleware - captures ALL API transactions
add_audit_middleware(
    app, 
    excluded_paths=[
        "/health", "/metrics", "/docs", "/openapi.json", "/favicon.ico", 
        "/api/aips/health", "/api/aips/info"
    ],  # Only exclude system monitoring endpoints
    log_request_body=True,  # Capture request bodies for complete transaction tracking
    log_response_body=False,  # Keep response logging disabled for performance
    max_body_size=10000  # Increase size limit for better transaction details
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"{request.method} {request.url} - {response.status_code} - {time.time() - start:.2f}s")
    return response

# Check if validation system is available
try:
    from app.validation.llm_validator import LLMResponseValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

@app.get("/api/aips/health")
def health():
    """Enhanced health check with AWS connection status and audit system"""
    from app.api.aoss_routes import get_store
    
    health_status = {
        "service_name": "EDGP AI Policy Suggest Microservice",
        "version": "1.0",
        "status": "ok",
        "timestamp": time.time(),
        "services": {
            "fastapi": "healthy",
            "opensearch": "unknown",
            "validation": "unknown",
            "audit_system": "unknown"
        }
    }
    
    # Test OpenSearch connection
    try:
        store = get_store()
        if store is None:
            health_status["services"]["opensearch"] = "unavailable"
            health_status["opensearch_message"] = "Store initialization failed - likely AWS permission issues"
            health_status["status"] = "ok"  # Still return 200 even if OpenSearch is unavailable
        else:
            # Try a simple operation
            try:
                store.client.info()
                health_status["services"]["opensearch"] = "healthy"
            except Exception as e:
                health_status["services"]["opensearch"] = "error"
                health_status["opensearch_error"] = str(e)[:100]  # Truncate error
                health_status["status"] = "ok"  # Still return 200 even if OpenSearch has error
    except Exception as e:
        health_status["services"]["opensearch"] = "error"
        health_status["opensearch_error"] = str(e)[:100]
        health_status["status"] = "ok"  # Still return 200 even if OpenSearch has error
    
    # Test validation system
    if VALIDATION_AVAILABLE:
        try:
            from app.validation.llm_validator import LLMResponseValidator
            validator = LLMResponseValidator()
            health_status["services"]["validation"] = "healthy"
        except Exception as e:
            health_status["services"]["validation"] = "error"
            health_status["validation_error"] = str(e)[:100]
    else:
        health_status["services"]["validation"] = "unavailable"
        health_status["validation_message"] = "Validation system not installed"
    
    # Test audit system
    try:
        audit_health = audit_system_health()
        if audit_health["sqs_configured"] and audit_health["sqs_client_initialized"]:
            if audit_health["connection_test"]:
                health_status["services"]["audit_system"] = "healthy"
            else:
                health_status["services"]["audit_system"] = "degraded"
                health_status["audit_message"] = "SQS connection test failed"
        else:
            health_status["services"]["audit_system"] = "unavailable"
            health_status["audit_message"] = "SQS not configured - audit logs will be written locally"
        health_status["audit_details"] = audit_health
    except Exception as e:
        health_status["services"]["audit_system"] = "error"
        health_status["audit_error"] = str(e)[:100]
    
    # Overall status
    unhealthy_services = [
        service for service, status in health_status["services"].items() 
        if status not in ["healthy", "unknown"]
    ]
    
    if unhealthy_services:
        health_status["status"] = "degraded"
        health_status["message"] = f"Some services are unavailable: {', '.join(unhealthy_services)}"
    
    return health_status

@app.get("/api/aips/info")
def service_info():
    """Service information endpoint"""
    from app.api.aoss_routes import get_store
    info = {
        "service_name": "EDGP AI Policy Suggest Microservice",
        "version": "1.0",
        "description": "AI-powered data quality policy and rule suggestion microservice",
        "endpoints": {
            "health": {
                "method": "GET",
                "path": "/api/aips/health",
                "description": "Health check with OpenSearch and audit system status"
            },
            "info": {
                "method": "GET",
                "path": "/api/aips/info",
                "description": "Service information and live vector DB status"
            },
            "suggest_rules": {
                "method": "POST",
                "path": "/api/aips/rules/suggest",
                "description": "Suggest validation rules for a domain with agent insights by default"
            },
            "create_domain": {
                "method": "POST",
                "path": "/api/aips/domains/create",
                "description": "Create a new domain with columns"
            },
            "extend_domain": {
                "method": "PUT",
                "path": "/api/aips/domains/extend-schema",
                "description": "Extend an existing domain with new columns"
            },
            "suggest_extend_schema": {
                "method": "POST",
                "path": "/api/aips/domains/suggest-extend-schema/{domain_name}",
                "description": "Suggest additional columns for an existing domain"
            },
            "suggest_schema": {
                "method": "POST",
                "path": "/api/aips/domains/suggest-schema",
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
                "path": "/api/aips/domains/{domain_name}",
                "description": "Get details for a specific domain"
            }
        },
        "repository": "edgp-ai-policy-suggest",
        "branch": "task/llm-validation"
    }
    
    # Add validation endpoints if available
    if VALIDATION_AVAILABLE:
        info["endpoints"].update({
            "validation_metrics": {
                "method": "GET",
                "path": "/api/aips/validation/metrics",
                "description": "Get LLM validation metrics and statistics"
            },
            "validate_schema": {
                "method": "POST",
                "path": "/api/aips/validation/validate-schema",
                "description": "Validate an LLM-generated schema response"
            },
            "validate_rules": {
                "method": "POST",
                "path": "/api/aips/validation/validate-rules",
                "description": "Validate LLM-generated rules"
            },
            "validation_health": {
                "method": "GET",
                "path": "/api/aips/validation/health",
                "description": "Health check for validation system"
            },
            "validation_test": {
                "method": "POST",
                "path": "/api/aips/validation/test",
                "description": "Test endpoint for validation system"
            }
        })
        info["validation_system"] = "enabled"
    else:
        info["validation_system"] = "disabled"
    
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
