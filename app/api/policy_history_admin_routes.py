from fastapi import APIRouter, Depends, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import logging

from app.auth.authentication import verify_any_scope_token, UserInfo
from app.aoss.policy_history_store import PolicyHistoryStore
from app.embedding.embedder import embed_column_names_batched_async

router = APIRouter(tags=["policy-history-admin"])
logger = logging.getLogger(__name__)


def _schema_to_string(schema: Dict[str, Any]) -> str:
    """Convert schema map to a compact string for embedding."""
    parts: List[str] = []
    for col_name, col_info in schema.items():
        dtype = col_info.get("dtype", "unknown") if isinstance(col_info, dict) else "unknown"
        parts.append(f"{col_name}:{dtype}")
    return ",".join(parts)


@router.post("/api/aips/policy-history/retrieve")
async def retrieve_policy_history(
    domain: str = Body(..., embed=True, description="Domain name to search"),
    schema_map: Optional[Dict[str, Any]] = Body(
        None,
        embed=True,
        alias="schema",
        description="Optional schema map for better intent"
    ),
    org_id: Optional[str] = Body(None, embed=True, description="Optional org filter (multi-tenant)"),
    top_k: int = Body(5, embed=True),
    min_success_score: Optional[float] = Body(0.0, embed=True),
    user: UserInfo = Depends(verify_any_scope_token)
):
    """Retrieve historical policies using hybrid search (BM25 + kNN) with auth.

    Builds a compact intent query from domain and optional schema, embeds it, and performs a
    hybrid retrieval. Results are re-ranked by success_score and recency.
    """
    try:
        # Build query text
        cols_part = _schema_to_string(schema_map) if schema_map else ""
        query_text = f"domain:{domain} {cols_part}".strip()

        # Embed the intent text
        embedding = await embed_column_names_batched_async([query_text])
        query_embedding = embedding[0]

        store = PolicyHistoryStore()
        results = await store.retrieve_policies_hybrid(
            query_text=query_text,
            query_embedding=query_embedding,
            org_id=org_id,
            domain=domain,
            top_k=top_k,
            min_success_score=min_success_score,
        )

        # Light response shaping
        items = []
        for r in results:
            items.append({
                "score": r.get("score"),
                "history_id": r.get("history_id"),
                "policy_id": r.get("policy_id"),
                "org_id": r.get("org_id"),
                "domain": r.get("domain"),
                "success_score": r.get("success_score") or r.get("performance_metrics", {}).get("success_rate"),
                "last_used": r.get("last_used") or r.get("updated_at"),
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
                "rules": r.get("rules", []),
            })

        return {"success": True, "count": len(items), "items": items}
    except Exception as e:
        logger.error(f"Policy history retrieve failed: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
