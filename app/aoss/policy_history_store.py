"""
Policy History Store for RAG implementation
Stores and retrieves historical policy data with hybrid (BM25 + kNN) capabilities.

Updates:
- rules stored as an array of structured objects (nested mapping), not a JSON string
- added fields: history_id, org_id, success_score, last_used, embedding_text
- timestamps kept as ISO8601 with Z suffix
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json
from opensearchpy import OpenSearch
from app.core.config import settings
from app.aoss.aoss_client import create_aoss_client
from app.embedding.embedder import embed_column_names_batched_async
import logging

logger = logging.getLogger(__name__)

class PolicyHistoryDoc:
    def __init__(
        self,
        policy_id: str,
        domain: str,
        schema: Dict[str, Any],
        rules: List[Dict[str, Any]],
        performance_metrics: Dict[str, Any],
        created_at: datetime,
        embedding: List[float],
        history_id: Optional[str] = None,
        org_id: Optional[str] = None,
        success_score: Optional[float] = None,
        last_used: Optional[datetime] = None,
        embedding_text: Optional[str] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        updated_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[int] = None,
    ):
        self.history_id = history_id
        self.org_id = org_id
        self.policy_id = policy_id
        self.domain = domain
        self.schema = schema
        self.rules = rules
        self.performance_metrics = performance_metrics
        self.created_at = created_at
        self.updated_at = updated_at or created_at
        self.last_used = last_used
        self.embedding = embedding
        self.embedding_text = embedding_text
        self.success_score = success_score
        self.confidence_scores = confidence_scores or {}
        self.metadata = metadata or {}
        if version is not None:
            # allow callers to track schema version in metadata
            self.metadata.setdefault("version", version)

    @staticmethod
    def _to_iso_z(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        # ensure trailing Z
        iso = dt.isoformat().replace("+00:00", "Z")
        return iso

    def to_doc(self) -> Dict[str, Any]:
        doc = {
            "history_id": self.history_id,
            "org_id": self.org_id,
            "policy_id": self.policy_id,
            "domain": self.domain,
            "schema": self.schema,
            "rules": self.rules,
            "performance_metrics": self.performance_metrics,
            "success_score": self.success_score,
            "last_used": self._to_iso_z(self.last_used),
            "created_at": self._to_iso_z(self.created_at),
            "updated_at": self._to_iso_z(self.updated_at),
            "embedding_text": self.embedding_text,
            "embedding": self.embedding,
            "confidence_scores": self.confidence_scores,
            "metadata": self.metadata,
        }
        # remove None values to keep index lean
        return {k: v for k, v in doc.items() if v is not None}

class PolicyHistoryStore:
    def __init__(self, index_name: str = "edgp-policy-history"):
        self.index_name = index_name
        self.client = create_aoss_client()
        self.embedding_dim = settings.embed_dim
        self._ensure_index()

    def _ensure_index(self):
        """Create the index if it doesn't exist"""
        try:
            if not self.client.indices.exists(index=self.index_name):
                index_mapping = {
                    "settings": {
                        "index": {"knn": True}
                    },
                    "mappings": {
                        "properties": {
                            "history_id": {"type": "keyword"},
                            "org_id": {"type": "keyword"},
                            "policy_id": {"type": "keyword"},
                            "domain": {"type": "keyword"},
                            "schema": {"type": "object", "enabled": True},
                            "rules": {
                                "type": "nested",
                                "properties": {
                                    "rule_id": {"type": "keyword"},
                                    "gx_expectation": {"type": "keyword"},
                                    "column_name": {"type": "keyword"},
                                    "params": {"type": "object", "enabled": True},
                                    "category": {"type": "keyword"},
                                    "dtypes": {"type": "keyword"},
                                    "source": {"type": "keyword"},
                                    "why": {"type": "text"}
                                }
                            },
                            "performance_metrics": {"type": "object", "enabled": True},
                            "success_score": {"type": "float"},
                            "last_used": {"type": "date"},
                            "confidence_scores": {"type": "object", "enabled": True},
                            "metadata": {"type": "object", "enabled": True},
                            "embedding_text": {"type": "text"},
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
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"}
                        }
                    }
                }
                self.client.indices.create(index=self.index_name, body=index_mapping)
                logger.info(f"Created policy history index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to create policy history index: {e}")
            raise

    async def store_policy(self, policy: PolicyHistoryDoc):
        """Store a policy document"""
        try:
            doc = policy.to_doc()

            # Do NOT serialize rules; stored as nested structured array per mapping.
            # Normalize rules to approved fields to avoid mapping conflicts (e.g., rules.value).
            def _normalize_rule(r: Dict[str, Any]) -> Dict[str, Any]:
                allowed = {"rule_id", "gx_expectation", "column_name", "params", "category", "dtypes", "source", "why"}
                gx = r.get("gx_expectation") or r.get("rule_name") or r.get("type")
                params = r.get("params")
                if params is None:
                    v = r.get("value") or r.get("parameters")
                    if isinstance(v, dict):
                        params = v
                    elif v is not None:
                        params = {"value": v}
                rule_id = r.get("rule_id") or (f"{gx}_{r.get('column_name','')}" if gx else None)
                out = {
                    "rule_id": rule_id,
                    "gx_expectation": gx,
                    "column_name": r.get("column_name") or r.get("column"),
                    "params": params,
                    "category": r.get("category"),
                    "dtypes": r.get("dtypes"),
                    "source": r.get("source"),
                    "why": r.get("why") or r.get("description"),
                }
                return {k: v for k, v in out.items() if v is not None and k in allowed}

            if "rules" in doc and isinstance(doc["rules"], list):
                doc["rules"] = [_normalize_rule(r) for r in doc["rules"] if isinstance(r, dict)]

            # Ensure created_at/updated_at present as ISO Z
            if "created_at" not in doc:
                doc["created_at"] = PolicyHistoryDoc._to_iso_z(datetime.now(timezone.utc))
            if "updated_at" not in doc:
                doc["updated_at"] = doc["created_at"]

            # Store without refresh parameter for AOSS compatibility
            response = self.client.index(
                index=self.index_name,
                body=doc
            )
            # Get the generated ID from the response
            generated_id = response['_id']
            
            # Manually wait for a short time to allow indexing
            import asyncio
            await asyncio.sleep(0.1)
            
            logger.info(f"Stored policy with generated ID {generated_id} for domain {policy.domain}")
            return generated_id
        except Exception as e:
            logger.error(f"Failed to store policy: {e}")
            raise

    async def retrieve_similar_policies(
        self,
        query_embedding: List[float],
        domain: Optional[str] = None,
        min_success_rate: float = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve similar policies using vector search"""
        # Use values from settings with fallbacks
        min_success_rate = min_success_rate if min_success_rate is not None else settings.policy_min_success_rate
        top_k = top_k if top_k is not None else settings.policy_top_k
        try:
            query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [{
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k
                                }
                            }
                        }],
                        "filter": [
                            {"range": {"performance_metrics.success_rate": {"gte": min_success_rate}}}
                        ]
                    }
                }
            }

            if domain:
                query["query"]["bool"]["filter"].append({"term": {"domain": domain}})

            response = self.client.search(
                index=self.index_name,
                body=query
            )

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                # Deserialize rules from JSON string
                if "rules" in source and isinstance(source["rules"], str):
                    source["rules"] = json.loads(source["rules"])
                results.append({
                    "score": hit["_score"],
                    **source
                })
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve similar policies: {e}")
            return []

    async def retrieve_policies_hybrid(
        self,
        query_text: str,
        query_embedding: List[float],
        org_id: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: int = 5,
        min_success_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval: BM25 on embedding_text + kNN on embedding with client-side re-ranking.

        Re-rank by a blend of: BM25 score, kNN score, success_score, and recency (last_used).
        """
        # Defaults
        min_success = min_success_score if min_success_score is not None else settings.policy_min_success_rate
        candidate_size = max(top_k * 2, 10)
        try:
            filters: List[Dict[str, Any]] = []
            if org_id:
                filters.append({"term": {"org_id": org_id}})
            if domain:
                filters.append({"term": {"domain": domain}})
            if min_success is not None:
                filters.append({"range": {"success_score": {"gte": min_success}}})

            # 1) BM25 search on embedding_text
            bm25_query = {
                "size": candidate_size,
                "track_scores": True,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "embedding_text": {
                                        "query": query_text,
                                        "operator": "and"
                                    }
                                }
                            }
                        ],
                        "filter": filters
                    }
                }
            }
            bm25_resp = self.client.search(index=self.index_name, body=bm25_query)

            bm25_hits: Dict[str, Dict[str, Any]] = {}
            max_bm25 = 0.0
            for hit in bm25_resp.get("hits", {}).get("hits", []):
                _id = hit.get("_id")
                _score = float(hit.get("_score", 0.0))
                max_bm25 = max(max_bm25, _score)
                src = hit.get("_source", {})
                # Deserialize rules if old docs store string
                if isinstance(src.get("rules"), str):
                    try:
                        src["rules"] = json.loads(src["rules"])  # legacy
                    except Exception:
                        pass
                bm25_hits[_id] = {
                    "source": src,
                    "bm25": _score,
                }

            # 2) kNN search on embedding
            knn_query = {
                "size": candidate_size,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": candidate_size
                                    }
                                }
                            }
                        ],
                        "filter": filters
                    }
                }
            }
            knn_resp = self.client.search(index=self.index_name, body=knn_query)

            merged: Dict[str, Dict[str, Any]] = {}
            max_knn = 0.0
            for hit in knn_resp.get("hits", {}).get("hits", []):
                _id = hit.get("_id")
                _score = float(hit.get("_score", 0.0))
                max_knn = max(max_knn, _score)
                src = hit.get("_source", {})
                if isinstance(src.get("rules"), str):
                    try:
                        src["rules"] = json.loads(src["rules"])  # legacy
                    except Exception:
                        pass
                merged[_id] = {
                    "source": src,
                    "knn": _score,
                }

            # Merge BM25 into merged map, preserving sources
            for _id, val in bm25_hits.items():
                if _id in merged:
                    merged[_id]["bm25"] = val["bm25"]
                else:
                    merged[_id] = {
                        "source": val["source"],
                        "bm25": val["bm25"],
                    }

            # 3) Re-rank
            def to_iso(dt_str: Optional[str]) -> Optional[datetime]:
                if not dt_str:
                    return None
                try:
                    s = dt_str.replace("Z", "+00:00")
                    return datetime.fromisoformat(s)
                except Exception:
                    return None

            # compute max for normalization safety
            max_bm25 = max(max_bm25, 1e-9)
            max_knn = max(max_knn, 1e-9)

            scored: List[Dict[str, Any]] = []
            now = datetime.now(timezone.utc)
            for _id, item in merged.items():
                src = item["source"]
                bm25_s = float(item.get("bm25", 0.0)) / max_bm25
                knn_s = float(item.get("knn", 0.0)) / max_knn
                base = 0.5 * bm25_s + 0.5 * knn_s

                # success score boost
                success = src.get("success_score")
                if success is None:
                    # fallback to performance_metrics.success_rate
                    success = src.get("performance_metrics", {}).get("success_rate")
                success = float(success) if success is not None else 0.0

                # recency: exp decay over ~30 days
                last_used_iso = src.get("last_used") or src.get("updated_at")
                dt = to_iso(last_used_iso)
                if dt is None:
                    recency = 0.0
                else:
                    # ensure timezone-aware for subtraction
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    days = (now - dt).total_seconds() / 86400.0
                    import math
                    recency = math.exp(-max(days, 0.0) / 30.0)

                # weights
                hybrid = base + 0.2 * success + 0.2 * recency
                scored.append({
                    "_id": _id,
                    "score": hybrid,
                    "bm25": bm25_s,
                    "knn": knn_s,
                    "success": success,
                    "recency": recency,
                    **src,
                })

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        except Exception as e:
            logger.error(f"Failed hybrid retrieval: {e}")
            return []

    async def get_domain_policies(
        self,
        domain: str,
        sort_by: str = "created_at",
        order: str = "desc",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all policies for a domain"""
        try:
            query = {
                "size": limit,
                "query": {
                    "term": {"domain": domain}
                },
                "sort": [{sort_by: {"order": order}}]
            }

            response = self.client.search(
                index=self.index_name,
                body=query
            )

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                # Deserialize rules from JSON string
                if "rules" in source and isinstance(source["rules"], str):
                    source["rules"] = json.loads(source["rules"])
                results.append(source)
            return results
        except Exception as e:
            logger.error(f"Failed to get domain policies: {e}")
            return []

    async def update_policy_feedback(
        self,
        policy_id: str,
        feedback: Dict[str, Any]
    ):
        """Update policy with user feedback by using a query instead of document ID"""
        try:
            # First find the document by policy_id field
            query = {
                "query": {
                    "term": {
                        "policy_id": policy_id
                    }
                }
            }
            
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            if response["hits"]["total"]["value"] == 0:
                raise ValueError(f"No policy found with policy_id {policy_id}")
                
            # Get the internal _id of the document
            internal_id = response["hits"]["hits"][0]["_id"]
            
            update_body = {
                "doc": {
                    "metadata": {
                        "feedback": feedback,
                        "feedback_updated_at": PolicyHistoryDoc._to_iso_z(datetime.now(timezone.utc))
                    },
                    "updated_at": PolicyHistoryDoc._to_iso_z(datetime.now(timezone.utc))
                }
            }

            self.client.update(
                index=self.index_name,
                id=internal_id,  # Use the internal _id
                body=update_body
            )
            
            # Manually wait for a short time to allow indexing
            import asyncio
            await asyncio.sleep(0.1)
            logger.info(f"Updated feedback for policy {policy_id} (internal ID: {internal_id})")
        except Exception as e:
            logger.error(f"Failed to update policy feedback: {e}")
            raise

    async def purge_all(self) -> Dict[str, Any]:
        """Delete all documents in the policy history index (dangerous)."""
        try:
            body = {"query": {"match_all": {}}}
            resp = self.client.delete_by_query(index=self.index_name, body=body)
            return resp
        except Exception as e:
            logger.error(f"Failed to purge all policy history docs: {e}")
            raise

    async def purge_by_org(self, org_id: str) -> Dict[str, Any]:
        """Delete documents by org_id."""
        try:
            body = {"query": {"term": {"org_id": org_id}}}
            resp = self.client.delete_by_query(index=self.index_name, body=body)
            return resp
        except Exception as e:
            logger.error(f"Failed to purge policy history by org_id={org_id}: {e}")
            raise

    async def purge_by_domain(self, domain: str) -> Dict[str, Any]:
        """Delete documents by domain."""
        try:
            body = {"query": {"term": {"domain": domain}}}
            resp = self.client.delete_by_query(index=self.index_name, body=body)
            return resp
        except Exception as e:
            logger.error(f"Failed to purge policy history by domain={domain}: {e}")
            raise

    async def delete_index_and_recreate(self) -> Dict[str, Any]:
        """Delete the entire index and recreate with current mapping."""
        try:
            # Delete index if exists
            try:
                if self.client.indices.exists(index=self.index_name):
                    self.client.indices.delete(index=self.index_name)
                    logger.info(f"Deleted index {self.index_name}")
            except Exception as e:
                logger.warning(f"Index delete warning for {self.index_name}: {e}")

            # Recreate
            self._ensure_index()
            return {"status": "recreated", "index": self.index_name}
        except Exception as e:
            logger.error(f"Failed to delete and recreate index {self.index_name}: {e}")
            raise