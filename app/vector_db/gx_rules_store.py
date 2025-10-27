from app.core.config import settings
import asyncio
import logging
import json
import os
import requests
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class GXRulesStore:
    def __init__(self):
        # In-memory cache only; no vector DB
        self._cache_rules: list | None = None
        self._current_hash: str | None = None
        self._last_update: datetime | None = None
        self._update_lock = asyncio.Lock()
        self._refresh_interval = 300  # 5 minutes in seconds

    async def initialize(self):
        """No-op initializer for in-memory cache."""
        return

    async def refresh_rules(self) -> bool:
        """Fetch rules from Rule Microservice and update in-memory cache if changed"""
        async with self._update_lock:
            try:
                await self.initialize()  # No-op

                # Fetch current rules directly from Rule Microservice to avoid recursion
                current_rules = []
                rule_url = settings.rule_api_url or os.getenv("RULE_URL")
                if rule_url and rule_url not in ["{RULE_URL}", "RULE_URL"]:
                    try:
                        resp = requests.get(rule_url, timeout=5)
                        resp.raise_for_status()
                        current_rules = resp.json() if isinstance(resp.json(), list) else []
                    except Exception as e:
                        logger.warning(f"Failed to fetch rules from Rule MS: {e}")
                if not current_rules:
                    # Fallback to default rules if none returned
                    from app.tools.rule_tools import _get_default_rules
                    current_rules = _get_default_rules()
                if not current_rules:
                    logger.warning("No rules fetched from rule engine")
                    return False

                # Calculate a stable hash of current rules (process-independent)
                payload = json.dumps(current_rules, sort_keys=True, separators=(",", ":")).encode("utf-8")
                current_hash = hashlib.sha256(payload).hexdigest()

                # Check if rules have changed by comparing with stored hash
                stored_hash = await self._get_stored_hash()
                if stored_hash and stored_hash == current_hash:
                    logger.info("GX rules are up to date")
                    return False

                # Update in-memory cache
                timestamp = datetime.utcnow().isoformat()
                formatted_rules = []
                for rule in current_rules:
                    # Format rule to exactly match _get_default_rules structure
                    formatted_rule = {
                        "doc_type": "gx_rule",
                        "rule_name": rule.get("rule_name"),
                        "column_name": rule.get("column_name"),
                        # Use rule_value in storage to avoid mapping conflicts with existing index fields
                        "rule_value": rule.get("value"),
                        # Internal fields only for tracking
                        "last_updated": timestamp,
                        "rule_hash": current_hash
                    }
                    formatted_rules.append(formatted_rule)

                # Update in-memory cache only
                self._cache_rules = [
                    {
                        "rule_name": r.get("rule_name"),
                        "column_name": r.get("column_name"),
                        "value": r.get("value"),
                    }
                    for r in current_rules
                ]
                self._current_hash = current_hash
                self._last_update = datetime.utcnow()
                logger.info(f"Updated {len(current_rules)} GX rules in in-memory cache")
                return True

            except Exception as e:
                logger.error(f"Error refreshing GX rules: {e}")
                return False

    async def get_rules(self) -> list:
        """Retrieve GX rules from in-memory cache (fallbacks if empty)"""
        try:
            await self.initialize()
            # Return cached rules if present
            if self._cache_rules:
                return self._cache_rules

            # If cache is empty, try a one-time refresh immediately
            refreshed = await self.refresh_rules()
            if refreshed and self._cache_rules:
                return self._cache_rules

            # Final fallback to defaults
            logger.warning("No cached GX rules available; returning default rules")
            from app.tools.rule_tools import _get_default_rules
            return _get_default_rules()
        except Exception as e:
            logger.error(f"Error retrieving GX rules: {e}")
            from app.tools.rule_tools import _get_default_rules
            return _get_default_rules()  # Fallback to default rules

    async def _get_stored_hash(self) -> str | None:
        """Get the hash of the most recently stored rules"""
        try:
            await self.initialize()  # No-op
            return self._current_hash
        except Exception as e:
            logger.error(f"Error getting stored rule hash: {e}")
        return None

    def _clean_rule(self, rule: dict) -> dict:
        """Remove internal fields from rule before returning"""
        cleaned = rule.copy()
        cleaned.pop('last_updated', None)
        cleaned.pop('rule_hash', None)
        return cleaned

    def get_cached_rules(self) -> list | None:
        """Synchronous accessor for cached rules (no fallbacks)."""
        return self._cache_rules

    # All OpenSearch-related helpers removed; in-memory cache only

# Global instance
_rules_store = GXRulesStore()

async def get_rules_store() -> GXRulesStore:
    """Get the global GX rules store instance"""
    # In-memory store requires no special initialization
    return _rules_store

def get_rules_store_sync() -> GXRulesStore:
    """Synchronous accessor for the global GX rules store (for non-async callers)."""
    return _rules_store

async def start_rules_refresh_task():
    """Start background task to periodically refresh rules"""
    store = await get_rules_store()
    while True:
        await store.refresh_rules()
        await asyncio.sleep(store._refresh_interval)