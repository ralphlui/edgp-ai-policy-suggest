import json
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.validator_routes import (
    setup_validation_routes,
)

@pytest.fixture()
def app_client(monkeypatch):
    """
    Builds a minimal FastAPI app with the validation routes mounted,
    and patches common dependencies with harmless defaults.
    """
    app = FastAPI()
    setup_validation_routes(app)
    client = TestClient(app)

    # --- Common stubs used by many endpoints ---

    # 1) load_validation_config -> config object with expected attributes
    def fake_load_validation_config():
        # Mimic an enum-like profile having a .value
        profile = SimpleNamespace(value="strict")
        return SimpleNamespace(
            profile=profile,
            max_issues_allowed=5,
            min_confidence_score=0.8,
            enable_auto_correction=True,
            schema_validation_enabled=True,
            rule_validation_enabled=True,
            content_validation_enabled=True,
            domain_rules=[],
            domain_rules_count=0,  # not used but kept for clarity
        )

    monkeypatch.setattr(
        "app.api.validator_routes.load_validation_config",
        fake_load_validation_config,
        raising=True,
    )

    # 2) LLMResponseValidator class -> object with .validate_response(...)
    class FakeLLMResponseValidator:
        def __init__(self, *_args, **_kwargs):
            pass

        def validate_response(self, *args, **kwargs):
            # For /health we don't care about the return, only that it doesn't throw.
            # For /test we’ll override via a local monkeypatch in that test.
            return {"ok": True}

    monkeypatch.setattr(
        "app.api.validator_routes.LLMResponseValidator",
        FakeLLMResponseValidator,
        raising=True,
    )

    # 3) get_validation_summary(hours) -> minimal shape used by /stats
    def fake_get_validation_summary(hours: int):
        return {
            "success_rate": 0.9,
            "avg_confidence": 0.87,
            "total_validations": 123,
            "domain_stats": {"customer": {"count": 50}},
            "response_type_stats": {"schema": 70, "rule": 30},
        }

    monkeypatch.setattr(
        "app.api.validator_routes.get_validation_summary",
        fake_get_validation_summary,
        raising=True,
    )

    # 4) get_metrics_collector() -> object with methods used by endpoints
    class FakeCollector:
        def get_domain_performance(self, domain, hours):
            return {"domain": domain, "hours": hours, "success_rate": 0.88}

        def export_metrics(self, hours, fmt):
            if fmt.lower() == "json":
                return json.dumps({"hours": hours, "export": "ok"})
            return "col1,col2\nv1,v2\n"

    monkeypatch.setattr(
        "app.api.validator_routes.get_metrics_collector",
        lambda: FakeCollector(),
        raising=True,
    )

    # 5) _validation_monitor.check_performance(hours)
    class FakeMonitor:
        def check_performance(self, hours: int):
            return {"overall_health": "good", "hours": hours, "issues": []}

    # Replace the module-level singleton with our fake
    monkeypatch.setattr(
        "app.api.validator_routes._validation_monitor",
        FakeMonitor(),
        raising=True,
    )

    return client


def test_health_ok(app_client):
    res = app_client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert body["config_loaded"] is True
    assert body["validator_functional"] is True
    assert "timestamp" in body


def test_metrics_summary_ok(app_client):
    res = app_client.get("/metrics/summary?hours=24")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert "data" in body and "timestamp" in body
    # Check a couple of expected keys from fake_get_validation_summary
    assert "success_rate" in body["data"]
    assert "avg_confidence" in body["data"]


def test_domain_metrics_ok(app_client):
    res = app_client.get("/metrics/domain/customer?hours=12")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["data"]["domain"] == "customer"
    assert body["data"]["hours"] == 12


def test_performance_check_ok(app_client):
    res = app_client.get("/performance/check?hours=6")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["data"]["overall_health"] == "good"
    assert body["data"]["hours"] == 6


def test_get_config_ok(app_client):
    res = app_client.get("/config")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    data = body["data"]
    # From fake_load_validation_config
    assert data["profile"] == "strict"
    assert data["max_issues_allowed"] == 5
    assert data["min_confidence_score"] == 0.8
    assert data["enable_auto_correction"] is True
    assert data["schema_validation_enabled"] is True
    assert data["rule_validation_enabled"] is True
    assert data["content_validation_enabled"] is True
    assert data["domain_rules_count"] == 0


def test_metrics_export_json_ok(app_client):
    res = app_client.get("/metrics/export?hours=24&format=json")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/json")
    assert "attachment; filename=validation_metrics_24h.json" in res.headers.get(
        "content-disposition", ""
    )
    payload = res.json()
    assert payload["hours"] == 24
    assert payload["export"] == "ok"


def test_metrics_export_csv_ok(app_client):
    res = app_client.get("/metrics/export?hours=48&format=csv")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=validation_metrics_48h.csv" in res.headers.get(
        "content-disposition", ""
    )
    assert "col1,col2" in res.text


def test_test_validation_ok(app_client, monkeypatch):
    # Override the validator to return a realistic ValidationResult-like object
    class Issue:
        def __init__(self, field, severity, message, suggestion):
            self.field = field
            self.severity = SimpleNamespace(value=severity)
            self.message = message
            self.suggestion = suggestion

    class FakeValidationResult:
        is_valid = True
        confidence_score = 0.93
        issues = [Issue("name", "low", "Minor format issue", "Trim whitespace")]
        corrected_data = {"name": "Alice"}
        metadata = {"elapsed_ms": 12}

    class FakeLLMResponseValidator2:
        def __init__(self, *_args, **_kwargs):
            pass

        def validate_response(self, **kwargs):
            return FakeValidationResult

    monkeypatch.setattr(
        "app.api.validator_routes.LLMResponseValidator",
        FakeLLMResponseValidator2,
        raising=True,
    )

    payload = {"response_data": {"name": " Alice "}}
    res = app_client.post("/test?response_type=schema&strict_mode=true", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    data = body["data"]
    assert data["is_valid"] is True
    assert data["confidence_score"] == 0.93
    assert data["corrected_data"]["name"] == "Alice"
    assert data["issues"][0]["field"] == "name"
    assert data["issues"][0]["severity"] == "low"
    assert data["metadata"]["elapsed_ms"] == 12


def test_stats_ok(app_client, monkeypatch):
    # Ensure weekly/daily/hourly summaries are present and performance health returned
    res = app_client.get("/stats")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    data = body["data"]

    assert data["config_profile"] == "strict"
    for window in ("hourly_metrics", "daily_metrics", "weekly_metrics"):
        assert "success_rate" in data[window]
        assert "avg_confidence" in data[window]
        assert "total_validations" in data[window]

    assert data["performance_health"] in ("good", "warning", "bad")
    assert "active_domains" in data
    assert "response_types" in data


# --------- A couple of failure-path checks ---------

def test_metrics_summary_failure_returns_500(app_client, monkeypatch):
    def boom(_hours: int):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.api.validator_routes.get_validation_summary",
        boom,
        raising=True,
    )

    res = app_client.get("/metrics/summary?hours=24")
    assert res.status_code == 500
    body = res.json()
    assert body["detail"].startswith("Failed to retrieve metrics:")


def test_export_metrics_failure_returns_500(app_client, monkeypatch):
    class BadCollector:
        def export_metrics(self, *_args, **_kwargs):
            raise ValueError("nope")

    monkeypatch.setattr(
        "app.api.validator_routes.get_metrics_collector",
        lambda: BadCollector(),
        raising=True,
    )

    res = app_client.get("/metrics/export?hours=24&format=json")
    assert res.status_code == 500
    body = res.json()
    assert body["detail"].startswith("Failed to export metrics:")
