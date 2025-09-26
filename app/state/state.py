from typing import List, Dict, Optional
from pydantic import BaseModel

class ColumnInfo(BaseModel):
    dtype: str
    sample_values: List[str]

class LangGraphState(BaseModel):
    domain: str
    schema: Optional[Dict[str, ColumnInfo]] = None
    rules: Optional[List[str]] = None
    query_embedding: Optional[List[float]] = None
    results: Optional[List[Dict]] = None
    filtered_columns: Optional[List[Dict]] = None
    pii_only: Optional[bool] = False
    allowed_types: Optional[List[str]] = ["string", "integer", "date"]
    csv_ready: Optional[bool] = False
