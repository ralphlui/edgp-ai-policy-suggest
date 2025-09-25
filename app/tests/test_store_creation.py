import pytest

from app.core.state import STORES
from app.api.routes_opensearch import get_or_create_store
from app.aoss.column_store import OpenSearchColumnStore

def setup_function():
    # Reset STORES before each test
    STORES.clear()

def test_create_new_store(monkeypatch):
    # Monkeypatch ensure_index so it doesn't hit real OpenSearch
    monkeypatch.setattr(OpenSearchColumnStore, "ensure_index", lambda self: None)

    # Act
    store = get_or_create_store("customer")

    # Assert
    assert "customer" in STORES
    assert isinstance(store, OpenSearchColumnStore)
    assert STORES["customer"] is store

def test_reuse_existing_store(monkeypatch):
    monkeypatch.setattr(OpenSearchColumnStore, "ensure_index", lambda self: None)

    first = get_or_create_store("vendor")
    second = get_or_create_store("vendor")

    assert first is second
    assert len(STORES) == 1  # still only one store

def test_invalid_domain_rejected():
    with pytest.raises(Exception) as excinfo:
        get_or_create_store("Finance!")  # invalid char
    assert "Invalid domain name" in str(excinfo.value)
