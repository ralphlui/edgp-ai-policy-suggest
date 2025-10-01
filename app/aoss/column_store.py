from typing import List, Dict, Any, Optional, Iterable
from dataclasses import dataclass
from opensearchpy import OpenSearch, helpers
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
import logging

from app.aoss.aoss_client import create_aoss_client

logger = logging.getLogger(__name__)

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
        
        try:
            self.client = client or create_aoss_client()
            logger.info(f" OpenSearchColumnStore initialized with index: {index_name}")
        except Exception as e:
            logger.error(f" Failed to initialize OpenSearchColumnStore: {e}")
            logger.error("Check AWS_SETUP_GUIDE.md for troubleshooting")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.5, max=2.0))
    def ensure_index(self) -> None:
        """
        Idempotently create a k-NN index for vector search on column embeddings.
        """
        try:
            if self.client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists")
                return

            logger.info(f"Creating index: {self.index_name}")
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
            
            result = self.client.indices.create(index=self.index_name, body=body)
            logger.info(f" Successfully created index: {self.index_name}")
            
        except Exception as e:
            logger.error(f" Failed to create index {self.index_name}: {e}")
            if "AuthorizationException" in str(e):
                logger.error(" Permission denied - check your AOSS Data Access Policy")
            elif "ResourceAlreadyExistsException" in str(e):
                logger.info("Index already exists (race condition)")
                return  # This is OK
            raise

    def upsert_columns(self, docs: Iterable[ColumnDoc], request_timeout: int = 30) -> None:
        """
        Bulk upsert column documents. 'index' acts as upsert.
        """
        
        def _do_upsert():
            """Internal upsert function for retry mechanism"""
            try:
                # Ensure index exists before upserting
                self.ensure_index()
                
                # Convert docs to list for logging and validation
                doc_list = list(docs)
                if not doc_list:
                    logger.warning("No documents to upsert")
                    return

                # Validate embedding dimensions before creating actions
                for i, doc in enumerate(doc_list):
                    if len(doc.embedding) != self.embedding_dim:
                        logger.error(f"Document {i} has embedding dimension {len(doc.embedding)}, expected {self.embedding_dim}")
                        raise ValueError(f"Embedding dimension mismatch for document {i}: got {len(doc.embedding)}, expected {self.embedding_dim}")

                # For OpenSearch Serverless VECTORSEARCH indices, we cannot specify _id
                # Documents will be auto-assigned IDs by OpenSearch
                actions = [{
                    "_index": self.index_name,
                    "_op_type": "index",
                    "_source": d.to_doc()
                } for d in doc_list]

                logger.info(f"Upserting {len(actions)} documents to {self.index_name}")
                
                # Use bulk helper with detailed error handling
                try:
                    success_count, failed_items = helpers.bulk(
                        self.client, 
                        actions, 
                        request_timeout=request_timeout,
                        max_retries=0,  # Disable internal retries to get clearer errors
                        chunk_size=10   # Smaller chunks to avoid timeouts
                    )
                    
                    logger.info(f" Successfully upserted {success_count} documents")
                    
                    if failed_items:
                        logger.warning(f" {len(failed_items)} documents failed to upsert")
                        for failure in failed_items:
                            logger.error(f"Failed item: {failure}")
                    
                    return success_count
                            
                except Exception as bulk_exception:
                    logger.error(f" Bulk operation failed: {bulk_exception}")
                    logger.error(f"Exception type: {type(bulk_exception)}")
                    
                    # Try to extract more details from the exception
                    if hasattr(bulk_exception, 'errors'):
                        logger.error(f"Bulk errors: {bulk_exception.errors}")
                        for i, error in enumerate(bulk_exception.errors):
                            logger.error(f"Error {i+1}: {error}")
                    
                    # Check if it's a permission issue by trying to get index info
                    try:
                        index_info = self.client.indices.get(index=self.index_name)
                        logger.info(f"Index exists and is accessible: {self.index_name}")
                        logger.info(f"Index mappings: {index_info.get(self.index_name, {}).get('mappings', 'N/A')}")
                    except Exception as index_error:
                        logger.error(f"Cannot access index {self.index_name}: {index_error}")
                    
                    raise
                    
            except Exception as e:
                logger.error(f" Upsert operation failed: {e}")
                logger.error(f"Index: {self.index_name}, Documents: {len(list(doc_list)) if 'doc_list' in locals() else 'unknown'}")
                
                if "AuthorizationException" in str(e):
                    logger.error(" Permission denied - check your AOSS Data Access Policy")
                elif "timeout" in str(e).lower():
                    logger.error(" Request timeout - try reducing batch size or increasing timeout")
                elif "ConnectionError" in str(e):
                    logger.error(" Connection error - check network and AOSS availability")
                    
                raise

        # Apply retry decorator to the internal function
        retry_decorator = retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=0.5, max=2.0))
        retrying_upsert = retry_decorator(_do_upsert)
        
        return retrying_upsert()

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
    
    def get_columns_by_domain(self, domain: str, return_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get all columns for a specific domain without using embeddings.
        """
        fields = return_fields or ["column_id", "column_name", "metadata", "sample_values"]
        
        query = {
            "query": {
                "term": {"metadata.domain": domain}
            },
            "size": 100,  # Get up to 100 columns per domain
            "_source": fields
        }
        
        try:
            res = self.client.search(index=self.index_name, body=query)
            hits = res.get("hits", {}).get("hits", [])
            return [{k: h.get("_source", {}).get(k) for k in fields} for h in hits]
        except Exception as e:
            logger.error(f"Error searching for domain {domain}: {e}")
            return []
    
    def check_domain_exists_case_insensitive(self, domain: str) -> Dict[str, Any]:
        """
        Check if a domain exists with case-insensitive matching.
        
        Returns:
            Dict with 'exists' boolean and 'existing_domain' string if found
        """
        try:
            # First check if index exists
            if not self.client.indices.exists(index=self.index_name):
                return {"exists": False, "existing_domain": None}
            
            # Get all unique domains and check case-insensitively
            all_domains = self.get_all_domains()
            domain_lower = domain.lower()
            
            for existing_domain in all_domains:
                if existing_domain.lower() == domain_lower:
                    return {
                        "exists": True,
                        "existing_domain": existing_domain,
                        "requested_domain": domain
                    }
            
            return {"exists": False, "existing_domain": None}
            
        except Exception as e:
            logger.error(f"Error checking domain existence for {domain}: {e}")
            return {"exists": False, "existing_domain": None}

    def force_refresh_index(self) -> bool:
        """
        Force refresh the OpenSearch index to make recently indexed documents 
        immediately available for search. This is particularly useful for 
        OpenSearch Serverless which has eventual consistency.
        
        Note: OpenSearch Serverless doesn't support _refresh endpoint,
        so this will gracefully fail and rely on eventual consistency.
        
        Returns:
            bool: True if refresh succeeded, False otherwise
        """
        try:
            if self.client.indices.exists(index=self.index_name):
                # Try to refresh, but expect it to fail in OpenSearch Serverless
                self.client.indices.refresh(index=self.index_name)
                logger.info(f"Successfully refreshed index: {self.index_name}")
                return True
            else:
                logger.warning(f"Cannot refresh - index {self.index_name} does not exist")
                return False
        except Exception as e:
            # This is expected for OpenSearch Serverless - just log as debug
            logger.debug(f"Index refresh not supported (expected for OpenSearch Serverless): {e}")
            return False

    def get_all_domains(self) -> List[str]:
        """
        Get all unique domain names from the vector database.
        
        Returns:
            List of unique domain names
        """
        try:
            # First check if index exists
            if not self.client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} does not exist yet")
                return []
            
            # Use aggregation to get unique domains
            query = {
                "size": 0,  # We don't need the actual documents, just aggregation results
                "aggs": {
                    "unique_domains": {
                        "terms": {
                            "field": "metadata.domain",
                            "size": 1000  # Max domains to return
                        }
                    }
                }
            }
            
            res = self.client.search(index=self.index_name, body=query)
            buckets = res.get("aggregations", {}).get("unique_domains", {}).get("buckets", [])
            
            # Extract domain names from aggregation buckets
            domains = [bucket["key"] for bucket in buckets]
            return sorted(domains)  # Return sorted list for consistency
            
        except Exception as e:
            logger.error(f"Error getting all domains: {e}")
            return []

    def get_all_domains_realtime(self, force_refresh: bool = True) -> List[str]:
        """
        Get all unique domain names with real-time visibility of recently created domains.
        
        Args:
            force_refresh: Whether to attempt refresh the index before querying
        
        Returns:
            List of unique domain names
        """
        try:
            # First check if index exists
            if not self.client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} does not exist yet")
                return []
            
            # Try to refresh to see recently added documents
            # This may fail in OpenSearch Serverless, which is expected
            if force_refresh:
                self.force_refresh_index()
            
            # Use the standard get_all_domains method
            return self.get_all_domains()
            
        except Exception as e:
            logger.error(f"Error getting all domains (realtime): {e}")
            # Fall back to regular get_all_domains
            return self.get_all_domains()

# Global store instance for backward compatibility with tests
_store_instance = None

def get_store(index_name: str = "edgp-ai-policy-suggest-column-index") -> Optional[OpenSearchColumnStore]:
    """
    Get or create a global OpenSearchColumnStore instance.
    This function exists for backward compatibility with existing tests.
    """
    global _store_instance
    if _store_instance is None:
        try:
            _store_instance = OpenSearchColumnStore(index_name=index_name)
            logger.info(f"Created global column store instance with index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to create global column store: {e}")
            return None
    return _store_instance
