import os, pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_ingest_and_search(monkeypatch):
    # Monkeypatch ensure_index to avoid hitting real OpenSearch
    from app.aoss.column_store import OpenSearchColumnStore
    monkeypatch.setattr(OpenSearchColumnStore, "ensure_index", lambda self: None)
    monkeypatch.setattr(OpenSearchColumnStore, "upsert_columns", lambda self, docs: None)
    monkeypatch.setattr(OpenSearchColumnStore, "semantic_search", lambda self, **kwargs: [{"column_id": "finance_core.amount"}])

    ingest_payload = {
        "domain": "finance",
        "docs": [{
            "column_id": "finance_core.amount",
            "column_name": "amount",
            "sample_values": ["100"],
            "metadata": {"domain": "finance", "type": "integer", "pii": False, "table": "finance_core"}
        }]
    }
    r = client.post("/opensearch/ingest", json=ingest_payload)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    search_payload = {"domain": "finance", "query_text": "amount"}
    r = client.post("/opensearch/search", json=search_payload)
    assert r.status_code == 200
    assert r.json()["results"][0]["column_id"] == "finance_core.amount"
