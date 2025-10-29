import os
import pytest
from unittest.mock import patch, Mock
from app.core.gx_rules_store import GXRulesStore, get_rules_store, get_rules_store_sync
import types
from app.core import gx_rules_store as gxmod


@pytest.fixture(autouse=True)
def force_no_rule_url(monkeypatch):
    """Ensure the rules store does not attempt external HTTP fetches via RULE_URL/settings."""
    monkeypatch.setenv("RULE_URL", "RULE_URL")  # placeholder triggers fallback
    # Replace settings object in module with a minimal stub
    monkeypatch.setattr(gxmod, "settings", types.SimpleNamespace(rule_api_url=None), raising=False)

SAMPLE_RULES = [
    {"rule_name": "ExpectColumnValuesToBeInSet", "column_name": "gender", "value": ["M", "F"]},
    {"rule_name": "ExpectColumnValuesToBeBetween", "column_name": "age", "value": {"min_value": 0, "max_value": 99}},
]

@pytest.mark.asyncio
async def test_refresh_rules_updates_cache(monkeypatch):
    # Ensure RULE_URL is ignored so defaults are used
    monkeypatch.setenv("RULE_URL", "RULE_URL")

    store = GXRulesStore()

    with patch("app.tools.rule_tools._get_default_rules", return_value=SAMPLE_RULES):
        changed = await store.refresh_rules()
        assert changed is True
        cached = store.get_cached_rules()
        assert cached == SAMPLE_RULES  # cache stores the same shape (rule_name, column_name, value)
        assert store._current_hash is not None
        assert store._last_update is not None

@pytest.mark.asyncio
async def test_refresh_rules_no_change_returns_false(monkeypatch):
    monkeypatch.setenv("RULE_URL", "RULE_URL")
    store = GXRulesStore()

    with patch("app.tools.rule_tools._get_default_rules", return_value=SAMPLE_RULES):
        first = await store.refresh_rules()
        assert first is True
        # second with same rules should early-out as no change
        second = await store.refresh_rules()
        assert second is False
        assert store.get_cached_rules() == SAMPLE_RULES

@pytest.mark.asyncio
async def test_get_rules_returns_cached(monkeypatch):
    monkeypatch.setenv("RULE_URL", "RULE_URL")
    store = GXRulesStore()

    # Seed cache directly
    store._cache_rules = SAMPLE_RULES.copy()
    result = await store.get_rules()
    assert result == SAMPLE_RULES

@pytest.mark.asyncio
async def test_get_rules_refreshes_when_empty(monkeypatch):
    monkeypatch.setenv("RULE_URL", "RULE_URL")
    store = GXRulesStore()

    with patch("app.tools.rule_tools._get_default_rules", return_value=SAMPLE_RULES):
        # Cache is empty, get_rules should refresh then return cache
        result = await store.get_rules()
        assert result == SAMPLE_RULES
        assert store.get_cached_rules() == SAMPLE_RULES

@pytest.mark.asyncio
async def test_get_stored_hash_after_refresh(monkeypatch):
    monkeypatch.setenv("RULE_URL", "RULE_URL")
    store = GXRulesStore()

    with patch("app.tools.rule_tools._get_default_rules", return_value=SAMPLE_RULES):
        await store.refresh_rules()
        h = await store._get_stored_hash()
        assert isinstance(h, str) and len(h) > 0

@pytest.mark.asyncio
async def test_get_rules_store_singleton():
    s1 = await get_rules_store()
    s2 = await get_rules_store()
    assert s1 is s2


# ------------------ Merged extra tests ------------------

EXTRA_SAMPLE_RULES = [
    {"rule_name": "ExpectColumnValuesToBeInSet", "column_name": "gender", "value": ["M", "F"]},
    {"rule_name": "ExpectColumnValuesToBeBetween", "column_name": "age", "value": {"min_value": 0, "max_value": 99}},
]


@pytest.mark.asyncio
async def test_refresh_rules_uses_rule_url(monkeypatch):
    # Point settings.rule_api_url to a non-placeholder URL to trigger HTTP fetch branch
    monkeypatch.setattr(
        gxmod,
        "settings",
        types.SimpleNamespace(rule_api_url="http://example.com/rules"),
        raising=False,
    )

    # Mock requests.get to return a JSON list
    fake_resp = Mock()
    fake_resp.json.return_value = EXTRA_SAMPLE_RULES
    fake_resp.raise_for_status.return_value = None

    with patch("app.core.gx_rules_store.requests.get", return_value=fake_resp):
        store = GXRulesStore()
        changed = await store.refresh_rules()
        assert changed is True
        assert store.get_cached_rules() == EXTRA_SAMPLE_RULES
        assert isinstance(store._current_hash, str)
        assert store._last_update is not None


@pytest.mark.asyncio
async def test_refresh_rules_non_list_response_falls_back_to_defaults(monkeypatch):
    # Valid RULE URL but returns non-list
    monkeypatch.setattr(
        gxmod, "settings", types.SimpleNamespace(rule_api_url="http://example.com/rules"), raising=False
    )

    fake_resp = Mock()
    fake_resp.json.return_value = {"not": "a list"}
    fake_resp.raise_for_status.return_value = None

    with patch("app.core.gx_rules_store.requests.get", return_value=fake_resp), \
         patch("app.tools.rule_tools._get_default_rules", return_value=EXTRA_SAMPLE_RULES):
        store = GXRulesStore()
        changed = await store.refresh_rules()
        assert changed is True
        assert store.get_cached_rules() == EXTRA_SAMPLE_RULES


def test_clean_rule_removes_internal_fields():
    store = GXRulesStore()
    rule = {"rule_name": "x", "last_updated": "t", "rule_hash": "h"}
    cleaned = store._clean_rule(rule)
    assert "last_updated" not in cleaned and "rule_hash" not in cleaned


@pytest.mark.asyncio
async def test_get_rules_store_sync_matches_singleton():
    sync_store = get_rules_store_sync()
    async_store = await get_rules_store()
    assert sync_store is async_store


# Additional edge-case coverage to increase branch execution

@pytest.mark.asyncio
async def test_refresh_rules_http_error_fallback_to_defaults(monkeypatch):
    """When HTTP fetch raises, we fall back to default rules."""
    # Set a non-placeholder URL so HTTP path is attempted
    monkeypatch.setattr(
        gxmod, "settings", types.SimpleNamespace(rule_api_url="http://example.com/rules"), raising=False
    )

    # Simulate HTTP failure
    with patch("app.core.gx_rules_store.requests.get", side_effect=Exception("boom")), \
         patch("app.tools.rule_tools._get_default_rules", return_value=SAMPLE_RULES):
        store = GXRulesStore()
        changed = await store.refresh_rules()
        assert changed is True
        assert store.get_cached_rules() == SAMPLE_RULES


@pytest.mark.asyncio
async def test_get_rules_exception_fallback_defaults(monkeypatch):
    """If initialize raises, get_rules should return defaults (exception path)."""
    monkeypatch.setenv("RULE_URL", "RULE_URL")
    store = GXRulesStore()

    async def boom():
        raise RuntimeError("init failed")

    # Force exception in initialize to hit except branch
    store.initialize = boom

    with patch("app.tools.rule_tools._get_default_rules", return_value=SAMPLE_RULES):
        result = await store.get_rules()
        assert result == SAMPLE_RULES


@pytest.mark.asyncio
async def test_get_stored_hash_exception_returns_none(monkeypatch):
    """_get_stored_hash returns None on initialize error."""
    store = GXRulesStore()

    async def boom():
        raise RuntimeError("init failed")

    store.initialize = boom
    h = await store._get_stored_hash()
    assert h is None
