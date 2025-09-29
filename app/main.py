from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EDGP AI Policy Suggest Microservice", 
    version="1.0",
    description="AI-powered data quality policy and rule suggestion microservice"
)

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
    return {
        "service_name": "EDGP AI Policy Suggest Microservice",
        "version": "1.0",
        "description": "AI-powered data quality policy and rule suggestion microservice",
        "endpoints": {
            "health": "GET /api/aips/health",
            "info": "GET /api/aips/info", 
            "suggest_rules": "POST /api/aips/suggest-rules",
            "create_domain": "POST /api/aips/create/domain",
            "vectordb_status": "GET /api/aips/vectordb/status",
            "vectordb_domains": "GET /api/aips/vectordb/domains",
            "vectordb_domain": "GET /api/aips/vectordb/domain/{domain_name}"
        },
        "repository": "edgp-ai-policy-suggest",
        "branch": "feature/opensearch-column-store"
    }

app.include_router(router)
