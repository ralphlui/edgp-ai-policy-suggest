import time, logging, uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes_opensearch import router as opensearch_router
from app.core.config import settings
from app.core.state import STORES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(" Starting application...")
    logger.info(" Ready for lazy domain index creation")
    # Don't pre-create stores - let them be created on-demand to avoid startup issues
    
    yield
    
    # Shutdown
    logger.info(" Shutting down application...")
    STORES.clear()

app = FastAPI(
    title="Policy Suggestion Agent", 
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()
    logger.info(f"[{request_id}] {request.method} {request.url}")
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"[{request_id}] {request.method} {request.url} - {response.status_code} - {duration:.2f}s")
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(opensearch_router)
