from typing import List, Dict, Any, Optional, Iterable
from dataclasses import dataclass
from opensearchpy import OpenSearch, helpers
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from app.aoss.aoss_client import create_aoss_client

@dataclass
class ColumnDoc:
    column_id: str                 # unique id (e.g., "customer_core.email" or UUID)
    column_name: str
    embedding: List[float]         # length must match embedding_dim
    sample_values: List[str]
    metadata: Dict[str, Any]       # {"domain": "customer", "type": "string", "pii": True, "table": "customer_core"}

    def to_doc(self) -> Dict[str, Any]:
        return {
            "column_id": self.column_id,
            "column_name": self.column_name,
            "embedding": self.embedding,
            "sample_values": self.sample_values,
            "metadata": self.metadata,
        }

class OpenSearchColumnStore:
    def __init__(self, index_name: str, embedding_dim: int = 1536, client: Optional[OpenSearch] = None):
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.client = client or create_aoss_client()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.5, max=2.0))
    def ensure_index(self) -> None:
        """
        Idempotently create a k-NN index for vector search on column embeddings.
        """
        if self.client.indices.exists(index=self.index_name):
            return

        body = {
            "settings": {
                "index": {"knn": True}
            },
            "mappings": {
                "properties": {
                    "column_id": {"type": "keyword"},
                    "column_name": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                            "parameters": {"m": 16, "ef_construction": 128}
                        }
                    },
                    "sample_values": {"type": "keyword"},
                    "metadata": {
                        "properties": {
                            "domain": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "pii": {"type": "boolean"},
                            "table": {"type": "keyword"},
                            # Optional facets: owner, sensitivity, tags, lineage
                        }
                    }
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=body)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.5, max=2.0))
    def upsert_columns(self, docs: Iterable[ColumnDoc], request_timeout: int = 10) -> None:
        """
        Bulk upsert column documents. 'index' acts as upsert.
        """
        actions = [{
            "_index": self.index_name,
            "_id": d.column_id,
            "_op_type": "index",
            "_source": d.to_doc(),
        } for d in docs]

        if not actions:
            return

        helpers.bulk(self.client, actions, request_timeout=request_timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.5, max=2.0))
    def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        domain: Optional[str] = None,
        table: Optional[str] = None,
        pii_only: Optional[bool] = None,
        return_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic k-NN search with optional filters. Returns selected fields for downstream agents.
        """
        fields = return_fields or ["column_id", "column_name", "metadata", "sample_values"]

        must_clause = [{
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": max(top_k, 10)  # retrieve slightly more to improve ranking
                }
            }
        }]

        filter_clause = []
        if domain:
            filter_clause.append({"term": {"metadata.domain": domain}})
        if table:
            filter_clause.append({"term": {"metadata.table": table}})
        if pii_only is not None:
            filter_clause.append({"term": {"metadata.pii": pii_only}})

        query = {
            "size": top_k,
            "_source": fields,
            "query": {
                "bool": {
                    "must": must_clause,
                    "filter": filter_clause
                }
            }
        }

        res = self.client.search(index=self.index_name, body=query)
        hits = res.get("hits", {}).get("hits", [])
        return [{
            "score": h.get("_score"),
            **{k: h.get("_source", {}).get(k) for k in fields}
        } for h in hits]
