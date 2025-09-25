# app/core/state.py
from typing import Dict
from app.aoss.column_store import OpenSearchColumnStore

# Global registry of domain -> store
STORES: Dict[str, OpenSearchColumnStore] = {}
