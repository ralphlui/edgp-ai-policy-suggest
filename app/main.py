from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import time, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Policy Agent", version="1.0")

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

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(router)
