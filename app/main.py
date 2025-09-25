from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time, logging, uuid

from app.api.routes_opensearch import router as opensearch_router
from app.core.config import settings
from app.aoss.column_store import OpenSearchColumnStore

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

# --- OpenSearch stores per domain ---
STORES: dict[str, OpenSearchColumnStore] = {}

@app.on_event("startup")
async def startup_event():
    for domain in settings.domains:
        index_name = f"columns_{domain}"
        store = OpenSearchColumnStore(index_name=index_name, embedding_dim=1536)
        store.ensure_index()
        STORES[domain] = store
        logger.info(f"✅ OpenSearch index ensured for domain: {domain}")

# Routers
app.include_router(opensearch_router)
