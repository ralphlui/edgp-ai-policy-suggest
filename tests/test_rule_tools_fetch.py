import types
import sys
from datetime import datetime
from unittest.mock import patch, Mock, AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.tools import rule_tools as rt


def test_fetch_gx_rules_default_when_no_url(monkeypatch):
    # Ensure no URL configured
    monkeypatch.setattr(rt, "settings", types.SimpleNamespace(rule_api_url=None), raising=False)
    monkeypatch.delenv("RULE_URL", raising=False)

    rules = rt.fetch_gx_rules("")
    assert isinstance(rules, list)
    # Should include a known default rule
    assert any(r.get("rule_name") == "ExpectColumnValuesToBeInSet" for r in rules)


def test_fetch_gx_rules_http_error_fallback(monkeypatch):
    # Set a URL but make request fail
    monkeypatch.setattr(rt, "settings", types.SimpleNamespace(rule_api_url="http://example.com/rules"), raising=False)

    with patch.object(rt.requests, "get", side_effect=RuntimeError("boom")):
        rules = rt.fetch_gx_rules("")
        assert isinstance(rules, list)
        assert any(r.get("rule_name") == "ExpectColumnValuesToBeInSet" for r in rules)


# =====================
# Rules refresh routes
# =====================

# Import the router under test
from app.api.rules_refresh_routes import router as rules_refresh_router


@pytest.fixture
def app_with_rules_refresh_router():
    app = FastAPI()
    app.include_router(rules_refresh_router)
    return app


def test_refresh_gx_rules_success(app_with_rules_refresh_router):
    app = app_with_rules_refresh_router
    client = TestClient(app)

    # Mock rules store with expected async methods/attributes
    mock_store = Mock()
    mock_store.refresh_rules = AsyncMock(return_value=True)
    mock_store._get_stored_hash = AsyncMock(return_value="hash123")
    mock_store._cache_rules = [{"rule": 1}, {"rule": 2}]  # type: ignore[attr-defined]
    mock_store._last_update = datetime(2025, 1, 1, 12, 0, 0)  # type: ignore[attr-defined]

    # Inject a fake module for app.core.gx_rules_store so the endpoint import resolves
    fake_mod = types.ModuleType("app.core.gx_rules_store")
    async def _fake_get_rules_store():
        return mock_store
    setattr(fake_mod, "get_rules_store", _fake_get_rules_store)

    with patch.dict(sys.modules, {"app.core.gx_rules_store": fake_mod}):
        resp = client.post("/api/aips/rules/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["refreshed"] is True
        assert data["rule_count"] == 2
        assert data["current_hash"] == "hash123"
        assert data["last_update"] == mock_store._last_update.isoformat()


def test_refresh_gx_rules_failure_returns_error(app_with_rules_refresh_router):
    app = app_with_rules_refresh_router
    client = TestClient(app)

    # Inject a fake module whose get_rules_store raises, to exercise error branch
    fake_mod = types.ModuleType("app.core.gx_rules_store")
    async def _raise_error():
        raise RuntimeError("rules store unavailable")
    setattr(fake_mod, "get_rules_store", _raise_error)

    with patch.dict(sys.modules, {"app.core.gx_rules_store": fake_mod}):
        resp = client.post("/api/aips/rules/refresh")
        assert resp.status_code == 200  # endpoint returns JSON with success False
        data = resp.json()
        assert data["success"] is False
        assert "unavailable" in data["error"].lower()
