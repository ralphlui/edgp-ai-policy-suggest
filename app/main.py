import time, logging, uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_opensearch import router as opensearch_router
from app.core.config import settings
from app.core.state import STORES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Policy Suggestion Agent", version="1.0")

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

@app.on_event("startup")
async def startup_event():
    logger.info(" Starting application...")
    logger.info(" Ready for lazy domain index creation")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")

app.include_router(opensearch_router)
