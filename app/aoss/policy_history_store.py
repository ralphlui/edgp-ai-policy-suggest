"""
Policy History Store for RAG implementation
Stores and retrieves historical policy data with vector search capabilities
"""

from datetime import datetime
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
        confidence_scores: Dict[str, float] = None,
        updated_at: datetime = None,
        metadata: Dict[str, Any] = None
    ):
        self.policy_id = policy_id
        self.domain = domain
        self.schema = schema
        self.rules = rules
        self.performance_metrics = performance_metrics
        self.created_at = created_at
        self.updated_at = updated_at or created_at
        self.embedding = embedding
        self.confidence_scores = confidence_scores or {}
        self.metadata = metadata or {}

    def to_doc(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "domain": self.domain,
            "schema": self.schema,
            "rules": self.rules,
            "performance_metrics": self.performance_metrics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "embedding": self.embedding,
            "confidence_scores": self.confidence_scores,
            "metadata": self.metadata
        }

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
                            "policy_id": {"type": "keyword"},
                            "domain": {"type": "keyword"},
                            "schema": {"type": "object"},
                            "rules": {
                                "type": "keyword",  # Store rules as JSON string
                            },
                            "performance_metrics": {
                                "properties": {
                                    "success_rate": {"type": "float"},
                                    "validation_score": {"type": "float"},
                                    "usage_count": {"type": "integer"}
                                }
                            },
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
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
                            "confidence_scores": {
                                "properties": {
                                    "overall": {"type": "float"},
                                    "rule_generation": {"type": "float"},
                                    "validation": {"type": "float"},
                                    "formatting": {"type": "float"},
                                    "normalization": {"type": "float"}
                                }
                            }
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
            
            # Serialize rules to JSON string
            if "rules" in doc:
                doc["rules"] = json.dumps(doc["rules"])
            
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
                        "feedback_updated_at": datetime.now().isoformat()
                    },
                    "updated_at": datetime.now().isoformat()
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