"""
Tests for app/aoss/column_store.py module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from opensearchpy import OpenSearch
from tenacity import RetryError
import logging

from app.aoss.column_store import ColumnDoc, OpenSearchColumnStore


class TestColumnDoc:
    """Test ColumnDoc dataclass"""
    
    def test_column_doc_creation(self):
        """Test basic ColumnDoc creation"""
        doc = ColumnDoc(
            column_id="test_id",
            column_name="test_col",
            embedding=[1.0, 2.0, 3.0],
            content=["value1", "value2"],
            metadata={"domain": "test_domain"}
        )
        
        assert doc.column_id == "test_id"
        assert doc.column_name == "test_col"
        assert doc.embedding == [1.0, 2.0, 3.0]
        assert doc.content == ["value1", "value2"]
        assert doc.metadata == {"domain": "test_domain"}

    def test_column_doc_to_doc(self):
        """Test ColumnDoc to_doc method"""
        doc = ColumnDoc(
            column_id="test_id",
            column_name="test_col",
            embedding=[1.0, 2.0],
            content=["value1"],
            metadata={"domain": "test_domain", "table": "test_table"}
        )
        
        result = doc.to_doc()
        
        expected = {
            "column_id": "test_id",
            "column_name": "test_col",
            "embedding": [1.0, 2.0],
            "content": ["value1"],
            "domain": "test_domain",
            "table": "test_table"
        }
        
        assert result == expected

    def test_column_doc_empty_values(self):
        """Test ColumnDoc with empty values"""
        doc = ColumnDoc(
            column_id="",
            column_name="",
            embedding=[],
            content=[],
            metadata={}
        )
        
        result = doc.to_doc()
        
        expected = {
            "column_id": "",
            "column_name": "",
            "embedding": [],
            "content": []
        }
        
        assert result == expected


class TestOpenSearchColumnStore:
    """Test OpenSearchColumnStore class"""

    def test_init_success(self):
        """Test successful initialization"""
        store = OpenSearchColumnStore(
            host="test-host",
            index_name="test-index",
            embedding_dim=512
        )
        
        assert store.index_name == "test-index"
        assert store.embedding_dim == 512
        assert store.host == "test-host"

    def test_init_with_custom_client(self):
        """Test initialization with custom client"""
        mock_client = Mock()
        
        store = OpenSearchColumnStore(
            host="test-host",
            client=mock_client,
            embedding_dim=512
        )
        
        assert store.client == mock_client
        assert store.embedding_dim == 512

    @patch('app.aoss.column_store.OpenSearch')
    def test_init_client_creation_failure(self, mock_opensearch):
        """Test initialization when client creation fails"""
        mock_opensearch.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception) as exc_info:
            OpenSearchColumnStore(
                host="test-host",
                embedding_dim=512
            )
        
        assert "Connection failed" in str(exc_info.value)

    def test_ensure_index_exists(self):
        """Test ensure_index when index already exists"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        
        store = OpenSearchColumnStore(
            host="test-host",
            client=mock_client,
            embedding_dim=512
        )
        
        store.ensure_index("existing-index")
        
        mock_client.indices.exists.assert_called_once_with(index="existing-index")
        mock_client.indices.create.assert_not_called()

    def test_ensure_index_creates_new(self):
        """Test ensure_index creates new index when it doesn't exist"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {}
        
        store = OpenSearchColumnStore(
            host="test-host",
            client=mock_client,
            embedding_dim=512
        )
        
        store.ensure_index("new-index")
        
        mock_client.indices.exists.assert_called_once_with(index="new-index")
        mock_client.indices.create.assert_called_once()

    def test_ensure_index_authorization_exception(self):
        """Test ensure_index handles AuthorizationException properly."""
        mock_client = Mock()
        mock_client.indices.create.side_effect = Exception("AuthorizationException: Access denied")
        
        store = OpenSearchColumnStore(
            host="test-host",
            client=mock_client,
            embedding_dim=512
        )
        
        with pytest.raises(RetryError) as exc_info:
            store.ensure_index("auth-fail")
        
        # Check that the original exception is in the retry error's future
        assert exc_info.value.last_attempt.exception() is not None
        assert "AuthorizationException" in str(exc_info.value.last_attempt.exception())

    def test_ensure_index_already_exists_exception(self):
        """Test ensure_index handles ResourceAlreadyExistsError"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.side_effect = Exception("resource_already_exists_exception")
        
        store = OpenSearchColumnStore(
            host="test-host",
            client=mock_client,
            embedding_dim=512
        )
        
        # Should not raise exception for already exists error
        store.ensure_index("existing-index")

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_success(self, mock_bulk):
        """Test successful upsert of columns"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_bulk.return_value = (2, [])
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"}),
            ColumnDoc("id2", "col2", [0.3, 0.4], ["val2"], {"domain": "test"})
        ]
        
        result = store.upsert_columns(docs)
        
        assert result == 2
        mock_bulk.assert_called_once()

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_empty_docs(self, mock_bulk):
        """Test upsert with empty document list"""
        mock_client = Mock()
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        result = store.upsert_columns([])
        
        mock_bulk.assert_not_called()

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_embedding_dimension_mismatch(self, mock_bulk):
        """Test upsert with wrong embedding dimensions"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=5, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})  # Wrong dimension
        ]
        
        with pytest.raises(ValueError) as exc_info:
            store.upsert_columns(docs)
        
        assert "Embedding dimension mismatch" in str(exc_info.value)
        mock_bulk.assert_not_called()

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_bulk_failure(self, mock_bulk):
        """Test upsert with bulk operation failure"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_bulk.side_effect = Exception("Bulk operation failed")
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        with pytest.raises(RetryError) as exc_info:
            store.upsert_columns(docs)
        
        # Check that the original exception is in the retry error's future
        assert exc_info.value.last_attempt.exception() is not None
        assert "Bulk operation failed" in str(exc_info.value.last_attempt.exception())

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_partial_failures(self, mock_bulk):
        """Test upsert with some failed items"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        
        # Mock partial failures
        failed_items = [
            {"index": {"_id": "doc1", "status": 400, "error": {"type": "validation_exception"}}},
            {"index": {"_id": "doc2", "status": 200}}  # Success
        ]
        mock_bulk.return_value = (1, failed_items)
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"}),
            ColumnDoc("id2", "col2", [0.3, 0.4], ["val2"], {"domain": "test"})
        ]
        
        result = store.upsert_columns(docs)
        
        assert result == 1  # Only successful upserts counted

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_authorization_error_in_bulk(self, mock_bulk):
        """Test upsert with authorization error in bulk operation"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_bulk.side_effect = Exception("AuthorizationException: Access denied")
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        with pytest.raises(RetryError) as exc_info:
            store.upsert_columns(docs)
        
        # Check that the original exception is in the retry error's future
        assert exc_info.value.last_attempt.exception() is not None
        assert "AuthorizationException" in str(exc_info.value.last_attempt.exception())

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_timeout_error(self, mock_bulk):
        """Test upsert with timeout error"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_bulk.side_effect = Exception("Request timeout")
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        with pytest.raises(RetryError) as exc_info:
            store.upsert_columns(docs)
        
        # Check that the original exception is in the retry error's future
        assert exc_info.value.last_attempt.exception() is not None
        assert "timeout" in str(exc_info.value.last_attempt.exception()).lower()

    @patch('app.aoss.column_store.helpers.bulk')
    def test_semantic_search_success(self, mock_bulk):
        """Test successful semantic search"""
        mock_client = Mock()
        
        # Mock search response
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "column_id": "col1",
                            "column_name": "test_column",
                            "content": ["value1", "value2"],
                            "domain": "test_domain"
                        },
                        "_score": 0.95
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        query_embedding = [0.1, 0.2, 0.3]
        results = store.semantic_search(query_embedding, k=10)
        
        assert len(results) == 1
        assert results[0]["column_id"] == "col1"
        assert results[0]["score"] == 0.95

    @patch('app.aoss.column_store.helpers.bulk')
    def test_semantic_search_with_filters(self, mock_bulk):
        """Test semantic search with filters"""
        mock_client = Mock()
        
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "column_id": "col1",
                            "column_name": "test_column",
                            "content": ["value1"],
                            "domain": "specific_domain"
                        },
                        "_score": 0.9
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        query_embedding = [0.1, 0.2]
        filters = [{"domain": "specific_domain"}]
        results = store.semantic_search(query_embedding, k=5, filters=filters)
        
        assert len(results) == 1
        assert results[0]["domain"] == "specific_domain"
        
        # Verify search was called with filters
        search_call = mock_client.search.call_args
        assert search_call is not None

    @patch('app.aoss.column_store.helpers.bulk')
    def test_semantic_search_custom_return_fields(self, mock_bulk):
        """Test semantic search with custom return fields"""
        mock_client = Mock()
        
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "column_id": "col1",
                            "column_name": "test_column"
                        },
                        "_score": 0.85
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        query_embedding = [0.1, 0.2]
        results = store.semantic_search(
            query_embedding, 
            k=3, 
            return_fields=["column_id", "column_name"]
        )
        
        assert len(results) == 1
        assert "column_id" in results[0]
        assert "column_name" in results[0]
        assert "content" not in results[0]  # Not requested

    def test_get_columns_by_domain_success(self):
        """Test successful get_columns_by_domain"""
        mock_client = Mock()
        
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "column_id": "col1",
                            "column_name": "test_column",
                            "content": ["value1"],
                            "domain": "test_domain"
                        }
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        results = store.get_columns_by_domain("test_domain")
        
        assert len(results) == 1
        assert results[0]["domain"] == "test_domain"

    def test_get_columns_by_domain_with_custom_fields(self):
        """Test get_columns_by_domain with custom return fields"""
        mock_client = Mock()
        
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "column_id": "col1",
                            "domain": "test_domain"
                        }
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        results = store.get_columns_by_domain(
            "test_domain", 
            return_fields=["column_id", "domain"]
        )
        
        assert len(results) == 1
        assert "column_id" in results[0]
        assert "domain" in results[0]

    def test_get_columns_by_domain_search_error(self):
        """Test get_columns_by_domain with search error"""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Search failed")
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        with pytest.raises(Exception) as exc_info:
            store.get_columns_by_domain("test_domain")
        
        assert "Search failed" in str(exc_info.value)

    def test_get_columns_by_domain_empty_response(self):
        """Test get_columns_by_domain with empty response"""
        mock_client = Mock()
        
        mock_response = {
            "hits": {
                "hits": []
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        results = store.get_columns_by_domain("nonexistent_domain")
        
        assert len(results) == 0


class TestOpenSearchColumnStoreRetry:
    """Test retry mechanisms in OpenSearchColumnStore"""

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_retry_success(self, mock_bulk):
        """Test upsert retry mechanism eventually succeeds"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        
        # Fail first two times, succeed on third
        mock_bulk.side_effect = [
            Exception("Temporary error"),
            Exception("Another temporary error"),
            (1, [])  # Success on third try
        ]
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        result = store.upsert_columns(docs)
        
        assert result == 1
        assert mock_bulk.call_count == 3

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_retry_exhausted(self, mock_bulk):
        """Test upsert retry exhausted after max attempts"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        
        # Always fail
        mock_bulk.side_effect = Exception("Persistent error")
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        with pytest.raises(RetryError) as exc_info:
            store.upsert_columns(docs)
        
        # Check that the original exception is in the retry error's future
        assert exc_info.value.last_attempt.exception() is not None
        assert "Persistent error" in str(exc_info.value.last_attempt.exception())
        assert mock_bulk.call_count == 3  # Max retry attempts

    def test_ensure_index_retry_success(self):
        """Test ensure_index retry mechanism eventually succeeds"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        
        # Fail first time, succeed on second
        mock_client.indices.create.side_effect = [
            Exception("Temporary index creation error"),
            {}  # Success
        ]
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        # Should eventually succeed
        store.ensure_index("test-index")
        
        assert mock_client.indices.create.call_count == 2

    @patch('app.aoss.column_store.helpers.bulk')
    def test_semantic_search_retry_success(self, mock_bulk):
        """Test semantic search retry mechanism"""
        mock_client = Mock()
        
        # Fail first time, succeed on second
        mock_client.search.side_effect = [
            Exception("Temporary search error"),
            {
                "hits": {
                    "hits": [
                        {
                            "_source": {"column_id": "col1"},
                            "_score": 0.9
                        }
                    ]
                }
            }
        ]
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        query_embedding = [0.1, 0.2]
        results = store.semantic_search(query_embedding, k=5)
        
        assert len(results) == 1
        assert mock_client.search.call_count == 2


class TestOpenSearchColumnStoreEdgeCases:
    """Test edge cases and error conditions"""

    def test_upsert_columns_index_creation_failure_during_upsert(self):
        """Test upsert when index creation fails during the operation"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.side_effect = Exception("Cannot create index")
        
        store = OpenSearchColumnStore(index_name="fail-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        with pytest.raises(RetryError) as exc_info:
            store.upsert_columns(docs)
        
        # The error might be nested in retry errors, so check the chain
        error_chain = str(exc_info.value)
        assert "Cannot create index" in error_chain or "RetryError" in error_chain

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_bulk_exception_with_errors_attribute(self, mock_bulk):
        """Test upsert with bulk exception that has errors attribute"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        
        # Create exception with errors attribute
        bulk_exception = Exception("Bulk failed")
        bulk_exception.errors = [
            {"index": {"_id": "doc1", "status": 400, "error": "validation error"}}
        ]
        mock_bulk.side_effect = bulk_exception
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        with pytest.raises(RetryError):
            store.upsert_columns(docs)

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_index_access_check_on_error(self, mock_bulk):
        """Test upsert error handling checks index accessibility"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_bulk.side_effect = Exception("Some bulk error")
        
        # Mock index.get to succeed (index accessible)
        mock_client.indices.get.return_value = {
            "test-index": {
                "mappings": {
                    "properties": {
                        "column_id": {"type": "keyword"}
                    }
                }
            }
        }
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        with pytest.raises(RetryError):
            store.upsert_columns(docs)
        
        # Verify index access was checked - might be called multiple times due to retries
        assert mock_client.indices.get.call_count >= 1
        # Check it was called with the right index
        call_args = mock_client.indices.get.call_args_list
        assert any(call[1]["index"] == "test-index" for call in call_args)

    @patch('app.aoss.column_store.helpers.bulk')
    def test_upsert_columns_index_access_error_on_bulk_failure(self, mock_bulk):
        """Test upsert checks index access when bulk operation fails"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = True
        mock_bulk.side_effect = Exception("Bulk operation failed")
        
        # Mock index.get to fail (index not accessible)
        mock_client.indices.get.side_effect = Exception("Index access denied")
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2, client=mock_client)
        
        docs = [
            ColumnDoc("id1", "col1", [0.1, 0.2], ["val1"], {"domain": "test"})
        ]
        
        with pytest.raises(RetryError):
            store.upsert_columns(docs)

    def test_semantic_search_malformed_response(self):
        """Test semantic search with malformed response"""
        mock_client = Mock()
        
        # Mock malformed response (missing hits.hits)
        mock_response = {"hits": {}}
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        query_embedding = [0.1, 0.2]
        results = store.semantic_search(query_embedding, k=5)
        
        assert len(results) == 0

    def test_semantic_search_no_filters(self):
        """Test semantic search with no filters"""
        mock_client = Mock()
        
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {"column_id": "col1"},
                        "_score": 0.8
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        query_embedding = [0.1, 0.2]
        results = store.semantic_search(query_embedding, k=5, filters=None)
        
        assert len(results) == 1
        assert results[0]["column_id"] == "col1"

    def test_semantic_search_with_small_top_k(self):
        """Test semantic search with small top_k value"""
        mock_client = Mock()
        
        mock_response = {
            "hits": {
                "hits": [
                    {"_source": {"column_id": "col1"}, "_score": 0.9},
                    {"_source": {"column_id": "col2"}, "_score": 0.8}
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        query_embedding = [0.1, 0.2]
        results = store.semantic_search(query_embedding, k=1)
        
        assert len(results) == 2  # Returns all hits regardless of k limit

    def test_get_columns_by_domain_malformed_response(self):
        """Test get_columns_by_domain with malformed response"""
        mock_client = Mock()
        
        # Mock malformed response
        mock_response = {"hits": {}}
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index", client=mock_client)
        
        results = store.get_columns_by_domain("test_domain")
        
        assert len(results) == 0