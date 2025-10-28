from fastapi import APIRouter
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/aips/rules/refresh")
async def refresh_gx_rules():
    """Trigger an immediate refresh of GX rules cache.

    Fetches from Rule Microservice (RULE_URL) and updates the in-memory cache.
    Falls back to default rules if Rule MS is unavailable.
    """
    try:
        from app.core.gx_rules_store import get_rules_store

        store = await get_rules_store()
        refreshed = await store.refresh_rules()

        # Gather status details
        try:
            current_hash = await store._get_stored_hash()  # type: ignore[attr-defined]
        except Exception:
            current_hash = None

        rule_count = len(store._cache_rules) if getattr(store, "_cache_rules", None) else 0  # type: ignore[attr-defined]
        last_update = getattr(store, "_last_update", None)  # type: ignore[attr-defined]
        last_update_iso = last_update.isoformat() if isinstance(last_update, datetime) else None

        return {
            "success": True,
            "refreshed": bool(refreshed),
            "rule_count": rule_count,
            "current_hash": current_hash,
            "last_update": last_update_iso,
        }
    except Exception as e:
        logger.error(f"Rules refresh failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
