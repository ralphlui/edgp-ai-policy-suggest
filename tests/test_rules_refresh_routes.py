import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
import sys
import types

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
