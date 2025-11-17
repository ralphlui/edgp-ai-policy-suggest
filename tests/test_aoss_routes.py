import os
import types
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.api.aoss_routes import router as aoss_router
from app.api import aoss_routes


@pytest.fixture
def app_with_aoss_router():
    app = FastAPI()
    app.include_router(aoss_router)
    return app


def _make_fake_store(index_name="test-index", exists=True, doc_count=42, stats_raises=False):
    # Build a fake OpenSearch client shape used by the route
    client = Mock()

    # indices.exists
    client.indices.exists.return_value = exists

    # indices.stats
    if stats_raises:
        client.indices.stats.side_effect = Exception("stats error")
    else:
        client.indices.stats.return_value = {
            "indices": {
                index_name: {
                    "total": {"docs": {"count": doc_count}}
                }
            }
        }

    fake_store = types.SimpleNamespace(client=client, index_name=index_name)
    return fake_store, client


def test_status_store_unavailable(app_with_aoss_router):
    app = app_with_aoss_router
    client = TestClient(app)

    with patch("app.api.aoss_routes.get_store", return_value=None):
        r = client.get("/api/aips/vector/status")
        assert r.status_code == 503
        data = r.json()
        assert data["status"] == "error"
        assert data["connection"] == "failed"
        assert data["validation_status"] == "unavailable"


def test_status_connected_index_exists_with_metrics(app_with_aoss_router):
    app = app_with_aoss_router
    client = TestClient(app)

    fake_store, os_client = _make_fake_store(index_name="idx", exists=True, doc_count=123)

    # Stub ValidationMetrics.get_current_metrics
    class _StubMetrics:
        total_validations = 10
        success_rate = 0.95
        last_updated = datetime(2025, 1, 1, 12, 0, 0)

    class _DummyVM:
        @staticmethod
        def get_current_metrics():
            return _StubMetrics

    with patch("app.api.aoss_routes.get_store", return_value=fake_store), \
         patch("app.validation.metrics.ValidationMetrics", _DummyVM, create=True):
        r = client.get("/api/aips/vector/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "connected"
        assert data["index_name"] == "idx"
        assert data["index_exists"] is True
        assert data["document_count"] == 123
        assert data["validation_status"] == "available"
        assert data["validation_metrics"]["total_validations"] == 10
        assert data["validation_metrics"]["success_rate"] == 0.95
        assert data["validation_metrics"]["last_validation"] == _StubMetrics.last_updated.isoformat()
        # Ensure stats was queried
        os_client.indices.stats.assert_called_once()


def test_status_connected_index_not_exists(app_with_aoss_router):
    app = app_with_aoss_router
    client = TestClient(app)

    fake_store, os_client = _make_fake_store(index_name="idx", exists=False)

    class _StubMetrics:
        total_validations = 0
        success_rate = 0.0
        last_updated = None

    class _DummyVM:
        @staticmethod
        def get_current_metrics():
            return _StubMetrics

    with patch("app.api.aoss_routes.get_store", return_value=fake_store), \
         patch("app.validation.metrics.ValidationMetrics", _DummyVM, create=True):
        r = client.get("/api/aips/vector/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "connected"
        assert data["index_exists"] is False
        # stats should not be called when index does not exist
        os_client.indices.stats.assert_not_called()


def test_status_stats_error_sets_unknown(app_with_aoss_router):
    app = app_with_aoss_router
    client = TestClient(app)

    fake_store, os_client = _make_fake_store(index_name="idx", exists=True, stats_raises=True)

    class _StubMetrics:
        total_validations = 1
        success_rate = 1.0
        last_updated = datetime.utcnow()

    class _DummyVM:
        @staticmethod
        def get_current_metrics():
            return _StubMetrics

    with patch("app.api.aoss_routes.get_store", return_value=fake_store), \
         patch("app.validation.metrics.ValidationMetrics", _DummyVM, create=True):
        r = client.get("/api/aips/vector/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "connected"
        assert data["index_exists"] is True
        assert data["document_count"] == "unknown"
        assert "stats_error" in data


def test_status_validation_metrics_error(app_with_aoss_router):
    app = app_with_aoss_router
    client = TestClient(app)

    fake_store, _ = _make_fake_store(index_name="idx", exists=True, doc_count=5)

    class _DummyVM:
        @staticmethod
        def get_current_metrics():
            raise RuntimeError("metrics failed")

    with patch("app.api.aoss_routes.get_store", return_value=fake_store), \
         patch("app.validation.metrics.ValidationMetrics", _DummyVM, create=True):
        r = client.get("/api/aips/vector/status")
        assert r.status_code == 200
        data = r.json()
        assert data["validation_status"] == "available"
        assert "error" in data.get("validation_metrics", {})


def test_status_top_level_exception_returns_500(app_with_aoss_router):
    app = app_with_aoss_router
    client = TestClient(app)

    with patch("app.api.aoss_routes.get_store", side_effect=Exception("boom")):
        r = client.get("/api/aips/vector/status")
        assert r.status_code == 500
        data = r.json()
        assert data["status"] == "error"
        assert data["connection"] == "failed"
        assert data["validation_status"] == "error"


def test_status_no_env_set(app_with_aoss_router):
    """Test AOSS status when no environment variables are set"""
    app = app_with_aoss_router
    test_client = TestClient(app)
    
    with patch.dict(os.environ, {}, clear=True):
        with patch("app.api.aoss_routes.OpenSearchColumnStore", side_effect=Exception("No env")):
            # Reset the global store
            aoss_routes._store = None
            
            response = test_client.get("/api/aips/vector/status")
            assert response.status_code in [500, 503]  # Accept both error codes
            data = response.json()
            assert data["status"] in ["unavailable", "error"]
