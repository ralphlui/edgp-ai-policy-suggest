import sys
import types
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.rule_suggestion_routes import router as rules_router
import app.api.rule_suggestion_routes as rules_module


# ------------------ Test helpers / fakes ------------------

class FakeUser:
    def __init__(self, email="tester@example.com", scopes=None):
        self.email = email
        self.scopes = scopes or ["manage:mdm"]

class FakeStore:
    def __init__(self):
        self.refreshed = False
    def force_refresh_index(self):
        self.refreshed = True
        return True


# ------------------ Pytest fixtures ------------------

@pytest.fixture
def app():
    app = FastAPI()
    # Override auth dependency to avoid real JWT checks
    app.dependency_overrides[rules_module.verify_any_scope_token] = lambda: FakeUser()
    app.include_router(rules_router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)


# ------------------ Tests ------------------

def test_suggest_rules_schema_found_with_insights(client, monkeypatch):
    """
    Schema exists + include_insights=True:
    - Patches get_schema_by_domain to return a schema
    - Injects a fake app.agents.agent_runner module (AgentState & build_graph)
    - Expects 200 with rule_suggestions + confidence + agent_insights
    """
    schema = {"domain": "customer", "id": {"type": "string"}, "email": {"type": "string"}}
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: schema)

    class FakeAgentState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.errors = getattr(self, "errors", [])
            self.step_history = getattr(self, "step_history", [1, 2, 3])
            self.execution_metrics = getattr(self, "execution_metrics", {"total_execution_time": 1.7})
            self.rule_suggestions = getattr(
                self, "rule_suggestions",
                [{"column": "email", "rule": "format:email"}, {"column": "id", "rule": "not_empty"}]
            )
            self.data_schema = getattr(self, "data_schema", schema)
            self.thoughts = getattr(self, "thoughts", ["t1", "t2", "t3", "t4", "t5"])
            self.observations = getattr(self, "observations", ["o1"])
            self.reflections = getattr(self, "reflections", ["r1"])
            self.confidence_scores = getattr(self, "confidence_scores", {"dummy": 1.0})

    class FakeGraph:
        def invoke(self, initial_state):
            return FakeAgentState(
                data_schema=initial_state.data_schema,
                enhanced_prompt=initial_state.enhanced_prompt,
                rule_suggestions=[{"column": "email", "rule": "format:email"},
                                  {"column": "id", "rule": "not_empty"}],
                execution_metrics={"total_execution_time": 1.7},
                thoughts=["t1", "t2", "t3", "t4", "t5"],
                observations=["o1"],
                reflections=["r1"]
            )

    # Mock the RuleRAGEnhancer
    class FakeRAGEnhancer:
        async def enhance_prompt_with_history(self, schema, domain):
            return "Enhanced prompt for testing"
            
        async def store_successful_policy(self, domain, schema, rules, performance_metrics):
            return True

    # Ensure the dynamic imports inside the route resolve to our fake modules
    fake_agent_runner = types.SimpleNamespace(AgentState=FakeAgentState, build_graph=lambda: FakeGraph())
    monkeypatch.setitem(sys.modules, "app.agents.agent_runner", fake_agent_runner)
    monkeypatch.setattr(rules_module, "RuleRAGEnhancer", lambda: FakeRAGEnhancer())

    payload = {"domain": "customer", "include_insights": True}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("rule_suggestions", []), list)
    assert "confidence" in data and "overall" in data["confidence"]
    assert "agent_insights" in data and "execution_time" in data["agent_insights"]


def test_suggest_rules_schema_found_no_insights(client, monkeypatch):
    """
    Schema exists + include_insights=False: should call run_agent(schema)
    and return its output.
    """
    schema = {"domain": "orders", "order_id": {"type": "string"}}
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: schema)

    def fake_run_agent(s):
        assert s == schema
        return [{"column": "order_id", "rule": "not_empty"}]

    # run_agent was imported at module import time → patch at rules_module site
    monkeypatch.setattr(rules_module, "run_agent", fake_run_agent)

    payload = {"domain": "orders", "include_insights": False}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"rule_suggestions": [{"column": "order_id", "rule": "not_empty"}]}


def test_suggest_rules_vector_db_connection_failed(client, monkeypatch):
    """
    get_schema_by_domain raises a connection-like error: expect 503 with error_type=connection_failed.
    """
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda _: (_ for _ in ()).throw(Exception("ConnectionError: timed out")))

    payload = {"domain": "any", "include_insights": True}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 503
    data = resp.json()
    assert data["error"] == "Vector database connection failed"
    assert data["error_type"] == "connection_failed"


def test_suggest_rules_schema_not_found_generates_ai_suggestions(client, monkeypatch):
    """
    Vector DB accessible but schema not found: return 404 with AI-suggested column names.
    """
    # Always return None (schema not found), before and after refresh
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: None)

    # Provide get_store() for the refresh step (imported dynamically inside the handler)
    fake_schema_loader = types.SimpleNamespace(get_store=lambda: FakeStore())
    monkeypatch.setitem(sys.modules, "app.vector_db.schema_loader", fake_schema_loader)

    # Predictable suggestions from bootstrap
    def _bootstrap(domain):
        assert domain == "newdomain"
        return {"id": {}, "created_at": {}, "status": {}}
    monkeypatch.setattr(rules_module, "bootstrap_schema_for_domain", _bootstrap)

    payload = {"domain": "newdomain", "include_insights": True}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "Domain not found"
    assert data["domain"] == "newdomain"
    assert set(data["suggested_columns"]) == {"id", "created_at", "status"}
    # Confirm action endpoints (your latest version points to /api/aips/domains/create)
    assert data["actions"]["create_schema_with_csv"]["endpoint"] == "/api/aips/domains/create"
    assert data["actions"]["create_schema_only"]["endpoint"] == "/api/aips/domains/create"


def test_suggest_rules_unexpected_error_returns_500(client, monkeypatch):
    """
    Non-connection exception (accessible_but_error), then bootstrap also fails → outer except returns 500.
    """
    def _get_schema(_):
        # No connection keywords → accessible_but_error branch
        raise Exception("Some other error")
    monkeypatch.setattr(rules_module, "get_schema_by_domain", _get_schema)

    def _bootstrap(_):
        raise Exception("LLM failure")
    monkeypatch.setattr(rules_module, "bootstrap_schema_for_domain", _bootstrap)

    payload = {"domain": "oops", "include_insights": True}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 500
    assert resp.json()["error"] == "Internal server error"


def test_suggest_rules_bootstrap_failure_returns_500(client, monkeypatch):
    """
    Vector DB accessible and returns None for schema, but bootstrap_schema_for_domain raises.
    Expect 500 from outer exception handler.
    """
    # Vector DB accessible (no exception), but no schema found
    monkeypatch.setattr(rules_module, "get_schema_by_domain", lambda d: None)

    # bootstrap fails inside the domain-not-found flow
    def _bootstrap(_):
        raise Exception("boom")
    monkeypatch.setattr(rules_module, "bootstrap_schema_for_domain", _bootstrap)

    payload = {"domain": "finance", "include_insights": False}
    resp = client.post("/api/aips/rules/suggest", json=payload)
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"] == "Internal server error"
    assert data["domain"] == "finance"
