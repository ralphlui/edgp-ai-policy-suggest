# app/api/routes_opensearch.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import logging

from app.aoss.column_store import OpenSearchColumnStore, ColumnDoc
from app.tools.embeddings import embed_text
from app.core.state import STORES   # shared global dict
from app.utils.validators import validate_domain_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/opensearch", tags=["opensearch"])

# ---------- Schemas ----------
class IngestColumn(BaseModel):
    column_id: str
    column_name: str
    sample_values: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any]

class IngestPayload(BaseModel):
    domain: str
    docs: List[IngestColumn]

    @field_validator("domain")
    @classmethod
    def check_domain(cls, v: str) -> str:
        return validate_domain_name(v)

class QueryPayload(BaseModel):
    domain: str
    query_text: str
    top_k: int = 10
    table: Optional[str] = None
    pii_only: Optional[bool] = None

    @field_validator("domain")
    @classmethod
    def check_domain(cls, v: str) -> str:
        return validate_domain_name(v)

# ---------- Helpers ----------
def get_or_create_store(domain: str) -> OpenSearchColumnStore:
    try:
        validate_domain_name(domain)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if domain not in STORES:
        index_name = f"columns_{domain}"
        store = OpenSearchColumnStore(index_name=index_name, embedding_dim=1536)
        store.ensure_index()
        STORES[domain] = store
        logger.info(f"⚙️ Auto-created new domain index '{index_name}'")
    return STORES[domain]

# ---------- Routes ----------
@router.post("/ingest")
def ingest(payload: IngestPayload):
    try:
        store = get_or_create_store(payload.domain)
        docs = []
        for c in payload.docs:
            emb = embed_text(
                f"Column: {c.column_name} | Meta: {c.metadata} | Samples: {', '.join(c.sample_values)}"
            )
            docs.append(ColumnDoc(
                column_id=c.column_id,
                column_name=c.column_name,
                embedding=emb,
                sample_values=c.sample_values,
                metadata=c.metadata,
            ))
        store.upsert_columns(docs)
        return {"status": "ok", "count": len(docs), "domain": payload.domain}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
def search(payload: QueryPayload):
    try:
        store = get_or_create_store(payload.domain)
        qvec = embed_text(payload.query_text)
        results = store.semantic_search(
            query_embedding=qvec,
            top_k=payload.top_k,
            domain=payload.domain,
            table=payload.table,
            pii_only=payload.pii_only
        )
        return {"status": "ok", "domain": payload.domain, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health():
    return {"status": "ok"}
