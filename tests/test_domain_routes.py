# tests/test_domain_routes.py
import sys
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typing import Any, List

from app.api.domain_schema_routes import router as domains_router
import app.api.domain_schema_routes as domains_module



class FakeUser:
    def __init__(self, email="tester@example.com", scopes=None):
        self.email = email
        self.scopes = scopes or ["manage:mdm"]

class FakeIndices:
    def __init__(self, exists=True):
        self._exists = exists
    def exists(self, index):
        return self._exists

class FakeClient:
    def __init__(self, indices_exists=True, search_payload=None):
        self.indices = FakeIndices(indices_exists)
        self._search_payload = search_payload or {"hits": {"total": {"value": 0}, "hits": []}}
    def search(self, index, body):
        return self._search_payload

class FakeStore:
    def __init__(self, index_name="test_index", domains=None, indices_exists=True, search_payload=None):
        self.index_name = index_name
        self._domains = domains or []
        self.client = FakeClient(indices_exists=indices_exists, search_payload=search_payload)
        self._upserts: List[Any] = []
        self._refreshed = False

    # used by create_domain duplicate check
    def check_domain_exists_case_insensitive(self, domain):
        lower = domain.lower()
        for d in self._domains:
            if d.lower() == lower:
                return {"exists": True, "existing_domain": d}
        return {"exists": False}

    def get_columns_by_domain(self, domain):
        return [{"column_name": "id"}, {"column_name": "name"}]

    def upsert_columns(self, docs):
        self._upserts.extend(docs)

    def force_refresh_index(self):
        self._refreshed = True
        return True

    def get_all_domains_realtime(self, force_refresh=False):
        return list(self._domains)


# ---------- Pytest fixtures ----------

@pytest.fixture(autouse=True)
def reset_store():
    # reset the module-level cache each test
    domains_module._store = None
    yield
    domains_module._store = None

@pytest.fixture
def app():
    app = FastAPI()
    # Override auth dependency globally for tests
    app.dependency_overrides[domains_module.verify_any_scope_token] = lambda: FakeUser()
    app.include_router(domains_router)
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

@pytest.fixture
def mock_embeddings(monkeypatch):
    async def _fake_embed(names):
        return [[0.1, 0.2, 0.3] for _ in names]
    monkeypatch.setattr(domains_module, "embed_column_names_batched_async", _fake_embed)


# ---------- Tests: /create ----------

def test_create_missing_domain_returns_400(client, monkeypatch):
    monkeypatch.setattr(domains_module, "get_store", lambda: None)
    resp = client.post("/api/aips/domains/create", json={"columns": ["id", "name"]})
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "Missing required field: 'domain'"

def test_create_invalid_columns_type_returns_400(client, monkeypatch):
    monkeypatch.setattr(domains_module, "get_store", lambda: None)
    resp = client.post("/api/aips/domains/create", json={"domain": "Customer", "columns": "id,name"})
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"].startswith("Invalid format")

def test_create_duplicate_domain_returns_409(client, monkeypatch, mock_embeddings):
    # store reports existing domain (case-insensitive)
    store = FakeStore(domains=["customer"])
    monkeypatch.setattr(domains_module, "get_store", lambda: store)

    body = {"domain": "Customer", "columns": ["id", "name"]}
    resp = client.post("/api/aips/domains/create", json=body)
    assert resp.status_code == 409
    data = resp.json()
    assert data["status"] == "exists"
    assert data["existing_domain"] == "customer"
    # Because normalized_domain == "customer", this is NOT a case conflict
    assert data["case_conflict"] is False
    assert "extend-schema" in data["actions"]

def test_create_success_stores_docs_and_generates_rules(client, monkeypatch, mock_embeddings):
    store = FakeStore(domains=[])
    monkeypatch.setattr(domains_module, "get_store", lambda: store)

    # Patch rule agent import path
    class _FakeRunner:
        @staticmethod
        def run_agent(schema):
            return [{"column": k, "rule": "not_empty"} for k in schema][:2]
    sys.modules["app.agents.agent_runner"] = type("X", (), {"run_agent": _FakeRunner.run_agent})

    body = {"domain": "Orders", "columns": ["order_id", "created_at"], "return_csv": False}
    resp = client.post("/api/aips/domains/create", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "partial_success")
    assert data["domain"] == "orders"
    # Should have upserted 2 docs
    assert len(store._upserts) == 2
    # rules section present
    assert "rules_available" in data
    assert "rule_suggestions" in data

def test_create_return_csv_includes_download_info(client, monkeypatch, mock_embeddings):
    store = FakeStore(domains=[])
    monkeypatch.setattr(domains_module, "get_store", lambda: store)

    class _FakeRunner:
        @staticmethod
        def run_agent(schema):
            return [{"column": k, "rule": "not_empty"} for k in schema][:1]
    sys.modules["app.agents.agent_runner"] = type("X", (), {"run_agent": _FakeRunner.run_agent})

    body = {"domain": "Inventory", "columns": ["sku", "qty"], "return_csv": True}
    resp = client.post("/api/aips/domains/create", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["csv_download"]["available"] is True
    assert data["csv_download"]["filename"].endswith(".csv")
    assert "download_url" in data["csv_download"]


# ---------- Tests: /download-csv/{filename} ----------

def test_download_csv_success_after_create(client, monkeypatch, mock_embeddings):
    store = FakeStore(domains=[])
    monkeypatch.setattr(domains_module, "get_store", lambda: store)

    # Clear any existing file mappings
    domains_module.clear_file_mappings()

    class _FakeRunner:
        @staticmethod
        def run_agent(schema):
            return []
    sys.modules["app.agents.agent_runner"] = type("X", (), {"run_agent": _FakeRunner.run_agent})

    # Create with return_csv to generate the temp CSV
    body = {"domain": "Leads", "columns": ["email"], "return_csv": True}
    resp = client.post("/api/aips/domains/create", json=body)
    assert resp.status_code == 200
    response_data = resp.json()
    
    # Extract file ID from download URL
    download_url = response_data["csv_download"]["download_url"]
    file_id = download_url.split("/")[-1]

    # Now download it using the file ID
    dl = client.get(f"/api/aips/domains/download-csv/{file_id}")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("text/csv")
    
    # Clean up file mappings after test
    domains_module.clear_file_mappings()

def test_download_csv_invalid_name(client):
    # Test with invalid UUID format
    resp = client.get("/api/aips/domains/download-csv/not-a-uuid")
    assert resp.status_code == 400
    data = resp.json()
    assert "Invalid file ID format" in data["error"]

def test_download_csv_wrong_extension_returns_400(client):
    # Test with invalid UUID format ending with .txt
    resp = client.get("/api/aips/domains/download-csv/not-a-uuid.txt")
    assert resp.status_code == 400
    data = resp.json()
    assert "Invalid file ID format" in data["error"]


# ---------- Tests: / (list) and /verify/{domain} ----------

def test_get_domains_store_down_returns_503(client, monkeypatch):
    monkeypatch.setattr(domains_module, "get_store", lambda: None)
    resp = client.get("/api/aips/domains")
    assert resp.status_code == 503
    j = resp.json()
    assert j["message"] == "OpenSearch store not available"

def test_get_domains_success(client, monkeypatch):
    store = FakeStore(domains=["customer", "orders"])
    monkeypatch.setattr(domains_module, "get_store", lambda: store)
    resp = client.get("/api/aips/domains")
    assert resp.status_code == 200
    j = resp.json()
    assert j["success"] is True
    assert j["totalRecord"] == 2
    assert set(j["data"]) == {"customer", "orders"}

def test_verify_domain_exists_true(client, monkeypatch):
    store = FakeStore(domains=["customer"])
    monkeypatch.setattr(domains_module, "get_store", lambda: store)
    resp = client.get("/api/aips/domains/verify/Customer")
    assert resp.status_code == 200
    j = resp.json()
    assert j["exists"] is True
    assert j["normalized_domain"] == "customer"

def test_verify_domain_exists_false(client, monkeypatch):
    store = FakeStore(domains=["orders"])
    monkeypatch.setattr(domains_module, "get_store", lambda: store)
    resp = client.get("/api/aips/domains/verify/customer")
    assert resp.status_code == 200
    j = resp.json()
    assert j["exists"] is False


# ---------- Tests: /schema and /{domain} ----------

def test_schema_when_index_missing(client, monkeypatch):
    store = FakeStore(indices_exists=False)
    monkeypatch.setattr(domains_module, "get_store", lambda: store)
    resp = client.get("/api/aips/domains/schema")
    assert resp.status_code == 200
    j = resp.json()
    assert j["total_domains"] == 0
    assert "does not exist yet" in j["message"]

def test_get_domain_from_vectordb_not_found(client, monkeypatch):
    store = FakeStore(indices_exists=True, search_payload={"hits": {"total": {"value": 0}, "hits": []}})
    monkeypatch.setattr(domains_module, "get_store", lambda: store)
    resp = client.get("/api/aips/domains/unknown")
    assert resp.status_code == 404
    assert resp.json()["found"] is False

def test_get_domain_from_vectordb_found(client, monkeypatch):
    search_payload = {
        "hits": {
            "total": {"value": 2},
            "hits": [
                {"_source": {"column_name": "id", "metadata": {"type": "string"}, "sample_values": []}},
                {"_source": {"column_name": "name", "metadata": {"type": "string"}, "sample_values": []}},
            ],
        }
    }
    store = FakeStore(indices_exists=True, search_payload=search_payload)
    monkeypatch.setattr(domains_module, "get_store", lambda: store)
    resp = client.get("/api/aips/domains/customer")
    assert resp.status_code == 200
    j = resp.json()
    assert j["found"] is True
    assert j["column_count"] == 2
    assert [c["column_name"] for c in j["columns"]] == ["id", "name"]


# ---------- Tests: AI suggest endpoints ----------

def test_suggest_schema_needs_domain(client):
    resp = client.post("/api/aips/domains/suggest-schema", json={})
    assert resp.status_code == 400
    assert resp.json()["error"] == "Missing required field: 'domain'"

def test_suggest_schema_success(client, monkeypatch):
    # Patch SchemaSuggesterEnhanced to return fixed columns
    class FakeSuggester:
        async def bootstrap_schema_with_preferences(self, business_description, user_preferences):
            return {"columns": [{"column_name": "id"}, {"column_name": "email"}]}

    sys.modules["app.agents.schema_suggester"] = type(
        "X", (), {"SchemaSuggesterEnhanced": FakeSuggester}
    )

    resp = client.post("/api/aips/domains/suggest-schema", json={"domain": "customer"})
    assert resp.status_code == 200
    j = resp.json()
    assert j["domain"] == "customer"
    assert j["suggested_columns"] == ["id", "email"]

def test_suggest_extend_schema_404_when_domain_missing(client, monkeypatch):
    # store returns zero hits
    store = FakeStore(indices_exists=True, search_payload={"hits": {"total": {"value": 0}, "hits": []}})
    monkeypatch.setattr(domains_module, "get_store", lambda: store)

    resp = client.post("/api/aips/domains/suggest-extend-schema/customer", json={})
    assert resp.status_code == 404
    msg = resp.json()["message"].lower()
    assert "cannot suggest extensions for a domain that doesn't exist" in msg

def test_suggest_extend_schema_success(client, monkeypatch):
    # existing domain with 1 column
    search_payload = {
        "hits": {
            "total": {"value": 1},
            "hits": [{"_source": {"column_name": "id", "metadata": {"type": "string"}, "sample_values": []}}],
        }
    }
    store = FakeStore(indices_exists=True, search_payload=search_payload)
    monkeypatch.setattr(domains_module, "get_store", lambda: store)

    # Patch SchemaSuggesterEnhanced
    class FakeSuggester:
        async def bootstrap_schema_with_preferences(self, business_description, user_preferences):
            return {"columns": [{"column_name": "created_at"}, {"column_name": "status"}]}
    sys.modules["app.agents.schema_suggester"] = type("X", (), {"SchemaSuggesterEnhanced": FakeSuggester})

    resp = client.post("/api/aips/domains/suggest-extend-schema/customer",
                       json={"suggestion_preferences": {"column_count": 2}})
    assert resp.status_code == 200
    j = resp.json()
    assert j["status"] == "success"
    assert "suggestions" in j
    # Ensure structure is present for the default focus areas branch
    assert "suggestion_summary" in j
