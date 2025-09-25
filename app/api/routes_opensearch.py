from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import logging

from app.aoss.column_store import OpenSearchColumnStore, ColumnDoc
from app.tools.embeddings import embed_text
from app.main import STORES
from app.utils.validators import validate_domain_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/opensearch", tags=["opensearch"])

class IngestColumn(BaseModel):
    column_id: str
    column_name: str
    sample_values: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any]

class IngestPayload(BaseModel):
    domain: str
    docs: List[IngestColumn]

    # Pydantic v2 style validator
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
