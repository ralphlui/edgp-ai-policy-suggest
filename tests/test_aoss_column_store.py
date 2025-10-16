"""
Comprehensive unit tests for app.aoss.column_store module.
This test file aims to achieve high code coverage by testing all methods and edge cases.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any
import logging

# Import the classes and functions to test
from app.aoss.column_store import (
    ColumnDoc,
    OpenSearchColumnStore,
    get_store
)


class TestColumnDoc:
    """Test the ColumnDoc dataclass"""
    
    def test_column_doc_creation(self):
        """Test basic ColumnDoc creation"""
        doc = ColumnDoc(
            column_id="customer.email",
            column_name="email",
            embedding=[0.1, 0.2, 0.3],
            sample_values=["user@example.com", "admin@test.com"],
            metadata={"domain": "customer", "type": "string", "pii": True}
        )
        
        assert doc.column_id == "customer.email"
        assert doc.column_name == "email"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.sample_values == ["user@example.com", "admin@test.com"]
        assert doc.metadata["domain"] == "customer"
        
    def test_column_doc_to_doc(self):
        """Test the to_doc() method"""
        doc = ColumnDoc(
            column_id="test.id",
            column_name="id",
            embedding=[1.0, 2.0],
            sample_values=["123", "456"],
            metadata={"type": "integer"}
        )
        
        result = doc.to_doc()
        expected = {
            "column_id": "test.id",
            "column_name": "id", 
            "embedding": [1.0, 2.0],
            "sample_values": ["123", "456"],
            "metadata": {"type": "integer"}
        }
        
        assert result == expected
        
    def test_column_doc_empty_values(self):
        """Test ColumnDoc with empty sample values"""
        doc = ColumnDoc(
            column_id="test.empty",
            column_name="empty",
            embedding=[0.0],
            sample_values=[],
            metadata={}
        )
        
        assert doc.sample_values == []
        assert doc.metadata == {}
        
        result = doc.to_doc()
        assert result["sample_values"] == []
        assert result["metadata"] == {}


class TestOpenSearchColumnStore:
    """Test the OpenSearchColumnStore class"""
    
    @patch('app.aoss.column_store.create_aoss_client')
    def test_initialization_success(self, mock_client_factory):
        """Test successful initialization"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=768)
        
        assert store.index_name == "test-index"
        assert store.embedding_dim == 768
        assert store.client == mock_client
        mock_client_factory.assert_called_once()
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_initialization_with_custom_client(self, mock_client_factory):
        """Test initialization with custom client"""
        custom_client = Mock()
        
        store = OpenSearchColumnStore(
            index_name="custom-index",
            embedding_dim=512,
            client=custom_client
        )
        
        assert store.client == custom_client
        assert store.index_name == "custom-index"
        assert store.embedding_dim == 512
        mock_client_factory.assert_not_called()
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_initialization_failure(self, mock_client_factory):
        """Test initialization failure"""
        mock_client_factory.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            OpenSearchColumnStore(index_name="test-index")
    
    @patch('app.aoss.column_store.create_aoss_client')
    def test_ensure_index_exists(self, mock_client_factory):
        """Test ensure_index when index already exists"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        store = OpenSearchColumnStore(index_name="existing-index")
        store.ensure_index()
        
        mock_client.indices.exists.assert_called_once_with(index="existing-index")
        mock_client.indices.create.assert_not_called()
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_ensure_index_creates_new(self, mock_client_factory):
        """Test ensure_index creates new index"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {"acknowledged": True}
        
        store = OpenSearchColumnStore(index_name="new-index", embedding_dim=384)
        store.ensure_index()
        
        mock_client.indices.exists.assert_called_once_with(index="new-index")
        mock_client.indices.create.assert_called_once()
        
        # Check the create call arguments
        call_args = mock_client.indices.create.call_args
        assert call_args[1]["index"] == "new-index"
        assert "body" in call_args[1]
        body = call_args[1]["body"]
        assert body["mappings"]["properties"]["embedding"]["dimension"] == 384
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_ensure_index_authorization_error(self, mock_client_factory):
        """Test ensure_index with authorization error"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.side_effect = Exception("AuthorizationException: Access denied")
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        # The @retry decorator will wrap the exception in RetryError
        with pytest.raises(Exception):  # Don't match specific text due to retry wrapper
            store.ensure_index()
            
    @patch('app.aoss.column_store.create_aoss_client')
    def test_ensure_index_already_exists_error(self, mock_client_factory):
        """Test ensure_index with ResourceAlreadyExistsException (race condition)"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.side_effect = Exception("ResourceAlreadyExistsException")
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        # Should not raise exception (race condition is OK)
        store.ensure_index()
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_upsert_columns_empty_docs(self, mock_client_factory):
        """Test upsert_columns with empty document list"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        # Should not raise exception with empty list
        store.upsert_columns([])
        
    @patch('app.aoss.column_store.create_aoss_client')
    @patch('opensearchpy.helpers.bulk')
    def test_upsert_columns_success(self, mock_bulk, mock_client_factory):
        """Test successful upsert_columns"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_bulk.return_value = (2, [])  # success_count, failed_items
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=3)
        
        docs = [
            ColumnDoc(
                column_id="test.col1",
                column_name="col1",
                embedding=[0.1, 0.2, 0.3],
                sample_values=["val1"],
                metadata={"domain": "test"}
            ),
            ColumnDoc(
                column_id="test.col2",
                column_name="col2", 
                embedding=[0.4, 0.5, 0.6],
                sample_values=["val2"],
                metadata={"domain": "test"}
            )
        ]
        
        result = store.upsert_columns(docs)
        
        mock_bulk.assert_called_once()
        # Verify the actions structure
        call_args = mock_bulk.call_args[0]
        actions = call_args[1]
        assert len(actions) == 2
        assert actions[0]["_index"] == "test-index"
        assert actions[0]["_op_type"] == "index"
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_upsert_columns_wrong_embedding_dimension(self, mock_client_factory):
        """Test upsert_columns with wrong embedding dimensions"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=5)
        
        docs = [
            ColumnDoc(
                column_id="test.col1",
                column_name="col1",
                embedding=[0.1, 0.2],  # Wrong dimension (2 instead of 5)
                sample_values=["val1"],
                metadata={"domain": "test"}
            )
        ]
        
        with pytest.raises(Exception):  # Will be wrapped in RetryError due to @retry decorator
            store.upsert_columns(docs)
            
    @patch('app.aoss.column_store.create_aoss_client')
    @patch('opensearchpy.helpers.bulk')
    def test_upsert_columns_bulk_failures(self, mock_bulk, mock_client_factory):
        """Test upsert_columns with some bulk failures"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        # Mock bulk to return some failures
        failed_items = [{"index": {"_id": "doc1", "status": 400, "error": "Bad request"}}]
        mock_bulk.return_value = (1, failed_items)
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2)
        
        docs = [
            ColumnDoc(
                column_id="test.col1",
                column_name="col1",
                embedding=[0.1, 0.2],
                sample_values=["val1"],
                metadata={"domain": "test"}
            )
        ]
        
        # Should not raise exception, just log warnings
        store.upsert_columns(docs)
        
    @patch('app.aoss.column_store.create_aoss_client')
    @patch('opensearchpy.helpers.bulk')
    def test_upsert_columns_bulk_exception(self, mock_bulk, mock_client_factory):
        """Test upsert_columns with bulk operation exception"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.indices.get.return_value = {"test-index": {"mappings": {}}}
        
        # Mock bulk to raise exception
        mock_bulk.side_effect = Exception("Bulk operation failed")
        
        store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2)
        
        docs = [
            ColumnDoc(
                column_id="test.col1",
                column_name="col1",
                embedding=[0.1, 0.2],
                sample_values=["val1"],
                metadata={"domain": "test"}
            )
        ]
        
        with pytest.raises(Exception):
            store.upsert_columns(docs)
            
    @patch('app.aoss.column_store.create_aoss_client')
    def test_semantic_search_basic(self, mock_client_factory):
        """Test basic semantic search functionality"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        
        # Mock search response
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.95,
                        "_source": {
                            "column_id": "customer.email",
                            "column_name": "email", 
                            "metadata": {"domain": "customer"},
                            "sample_values": ["test@example.com"]
                        }
                    },
                    {
                        "_score": 0.80,
                        "_source": {
                            "column_id": "user.email_address",
                            "column_name": "email_address",
                            "metadata": {"domain": "user"},
                            "sample_values": ["user@test.com"]
                        }
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        query_embedding = [0.1, 0.2, 0.3]
        results = store.semantic_search(query_embedding, top_k=5)
        
        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert results[0]["column_id"] == "customer.email"
        assert results[1]["score"] == 0.80
        assert results[1]["column_id"] == "user.email_address"
        
        # Verify the search call
        mock_client.search.assert_called_once()
        search_args = mock_client.search.call_args[1]
        assert search_args["index"] == "test-index"
        assert "body" in search_args
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_semantic_search_with_filters(self, mock_client_factory):
        """Test semantic search with domain and other filters"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        query_embedding = [0.1, 0.2, 0.3]
        store.semantic_search(
            query_embedding=query_embedding,
            top_k=10,
            domain="customer",
            table="customer_core", 
            pii_only=True,
            return_fields=["column_id", "column_name"]
        )
        
        # Verify the search query structure
        search_args = mock_client.search.call_args[1]
        query = search_args["body"]["query"]
        
        # Should have filter clauses
        filter_clause = query["bool"]["filter"]
        assert len(filter_clause) == 3  # domain, table, pii_only
        
        # Check specific filters
        domain_filter = next(f for f in filter_clause if "metadata.domain" in str(f))
        assert domain_filter == {"term": {"metadata.domain": "customer"}}
        
        table_filter = next(f for f in filter_clause if "metadata.table" in str(f))
        assert table_filter == {"term": {"metadata.table": "customer_core"}}
        
        pii_filter = next(f for f in filter_clause if "metadata.pii" in str(f))
        assert pii_filter == {"term": {"metadata.pii": True}}
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_columns_by_domain(self, mock_client_factory):
        """Test get_columns_by_domain method"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "column_id": "customer.name",
                            "column_name": "name",
                            "metadata": {"domain": "customer", "type": "string"},
                            "sample_values": ["John", "Jane"]
                        }
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        results = store.get_columns_by_domain("customer")
        
        assert len(results) == 1
        assert results[0]["column_id"] == "customer.name"
        assert results[0]["metadata"]["domain"] == "customer"
        
        # Verify search parameters
        search_args = mock_client.search.call_args[1]
        query = search_args["body"]["query"]
        assert query["term"]["metadata.domain"] == "customer"
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_columns_by_domain_error(self, mock_client_factory):
        """Test get_columns_by_domain with search error"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.search.side_effect = Exception("Search failed")
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        results = store.get_columns_by_domain("customer")
        
        assert results == []  # Should return empty list on error
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_check_domain_exists_case_insensitive_found(self, mock_client_factory):
        """Test check_domain_exists_case_insensitive when domain exists"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        with patch.object(store, 'get_all_domains') as mock_get_domains:
            mock_get_domains.return_value = ["Customer", "Product", "Order"]
            
            result = store.check_domain_exists_case_insensitive("customer")
            
            assert result["exists"] is True
            assert result["existing_domain"] == "Customer"
            assert result["requested_domain"] == "customer"
            
    @patch('app.aoss.column_store.create_aoss_client')
    def test_check_domain_exists_case_insensitive_not_found(self, mock_client_factory):
        """Test check_domain_exists_case_insensitive when domain doesn't exist"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        with patch.object(store, 'get_all_domains') as mock_get_domains:
            mock_get_domains.return_value = ["Customer", "Product"]
            
            result = store.check_domain_exists_case_insensitive("nonexistent")
            
            assert result["exists"] is False
            assert result["existing_domain"] is None
            
    @patch('app.aoss.column_store.create_aoss_client')
    def test_check_domain_exists_index_not_exists(self, mock_client_factory):
        """Test check_domain_exists_case_insensitive when index doesn't exist"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = False
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        result = store.check_domain_exists_case_insensitive("customer")
        
        assert result["exists"] is False
        assert result["existing_domain"] is None
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_check_domain_exists_error(self, mock_client_factory):
        """Test check_domain_exists_case_insensitive with error"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.side_effect = Exception("Connection error")
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        result = store.check_domain_exists_case_insensitive("customer")
        
        assert result["exists"] is False
        assert result["existing_domain"] is None
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_force_refresh_index_success(self, mock_client_factory):
        """Test successful force_refresh_index"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.indices.refresh.return_value = {"_shards": {"successful": 1}}
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        result = store.force_refresh_index()
        
        assert result is True
        mock_client.indices.refresh.assert_called_once_with(index="test-index")
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_force_refresh_index_not_supported(self, mock_client_factory):
        """Test force_refresh_index when refresh is not supported (OpenSearch Serverless)"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.indices.refresh.side_effect = Exception("Refresh not supported")
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        result = store.force_refresh_index()
        
        assert result is False
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_force_refresh_index_not_exists(self, mock_client_factory):
        """Test force_refresh_index when index doesn't exist"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = False
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        result = store.force_refresh_index()
        
        assert result is False
        mock_client.indices.refresh.assert_not_called()
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_all_domains_success(self, mock_client_factory):
        """Test successful get_all_domains"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        mock_response = {
            "aggregations": {
                "unique_domains": {
                    "buckets": [
                        {"key": "customer", "doc_count": 5},
                        {"key": "product", "doc_count": 3},
                        {"key": "order", "doc_count": 8}
                    ]
                }
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        domains = store.get_all_domains()
        
        assert domains == ["customer", "order", "product"]  # Should be sorted
        
        # Verify the aggregation query
        search_args = mock_client.search.call_args[1]
        assert search_args["body"]["size"] == 0
        assert "unique_domains" in search_args["body"]["aggs"]
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_all_domains_index_not_exists(self, mock_client_factory):
        """Test get_all_domains when index doesn't exist"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = False
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        domains = store.get_all_domains()
        
        assert domains == []
        mock_client.search.assert_not_called()
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_all_domains_error(self, mock_client_factory):
        """Test get_all_domains with search error"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.search.side_effect = Exception("Search failed")
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        domains = store.get_all_domains()
        
        assert domains == []
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_all_domains_realtime_with_refresh(self, mock_client_factory):
        """Test get_all_domains_realtime with force refresh"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        with patch.object(store, 'force_refresh_index') as mock_refresh, \
             patch.object(store, 'get_all_domains') as mock_get_domains:
            mock_get_domains.return_value = ["customer", "product"]
            
            result = store.get_all_domains_realtime(force_refresh=True)
            
            mock_refresh.assert_called_once()
            mock_get_domains.assert_called_once()
            assert result == ["customer", "product"]
            
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_all_domains_realtime_without_refresh(self, mock_client_factory):
        """Test get_all_domains_realtime without force refresh"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        with patch.object(store, 'force_refresh_index') as mock_refresh, \
             patch.object(store, 'get_all_domains') as mock_get_domains:
            mock_get_domains.return_value = ["customer"]
            
            result = store.get_all_domains_realtime(force_refresh=False)
            
            mock_refresh.assert_not_called()
            mock_get_domains.assert_called_once()
            assert result == ["customer"]
            
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_all_domains_realtime_error_fallback(self, mock_client_factory):
        """Test get_all_domains_realtime error handling with fallback"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.side_effect = Exception("Connection error")
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        with patch.object(store, 'get_all_domains') as mock_get_domains:
            mock_get_domains.return_value = ["fallback_domain"]
            
            result = store.get_all_domains_realtime()
            
            mock_get_domains.assert_called_once()
            assert result == ["fallback_domain"]


class TestGlobalStoreFunction:
    """Test the global get_store function"""
    
    @patch('app.aoss.column_store.OpenSearchColumnStore')
    def test_get_store_creates_instance(self, mock_store_class):
        """Test get_store creates new instance"""
        # Reset the global instance
        import app.aoss.column_store as store_module
        store_module._store_instance = None
        
        mock_instance = Mock()
        mock_store_class.return_value = mock_instance
        
        result = get_store("custom-index")
        
        assert result == mock_instance
        mock_store_class.assert_called_once_with(index_name="custom-index")
        
    @patch('app.aoss.column_store.OpenSearchColumnStore')
    def test_get_store_returns_existing_instance(self, mock_store_class):
        """Test get_store returns existing instance"""
        # Set up existing instance
        import app.aoss.column_store as store_module
        existing_instance = Mock()
        store_module._store_instance = existing_instance
        
        result = get_store("any-index")
        
        assert result == existing_instance
        mock_store_class.assert_not_called()
        
    @patch('app.aoss.column_store.OpenSearchColumnStore')
    def test_get_store_handles_creation_error(self, mock_store_class):
        """Test get_store handles creation error"""
        # Reset the global instance
        import app.aoss.column_store as store_module
        store_module._store_instance = None
        
        mock_store_class.side_effect = Exception("Creation failed")
        
        result = get_store("test-index")
        
        assert result is None
        
    def teardown_method(self):
        """Clean up global state after each test"""
        import app.aoss.column_store as store_module
        store_module._store_instance = None


class TestErrorHandlingAndEdgeCases:
    """Test various error scenarios and edge cases"""
    
    @patch('app.aoss.column_store.create_aoss_client')
    def test_upsert_timeout_error(self, mock_client_factory):
        """Test upsert with timeout error"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        with patch('opensearchpy.helpers.bulk') as mock_bulk:
            mock_bulk.side_effect = Exception("timeout occurred")
            
            store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2)
            
            docs = [
                ColumnDoc(
                    column_id="test.col1",
                    column_name="col1",
                    embedding=[0.1, 0.2],
                    sample_values=["val1"],
                    metadata={"domain": "test"}
                )
            ]
            
            with pytest.raises(Exception):
                store.upsert_columns(docs)
                
    @patch('app.aoss.column_store.create_aoss_client')
    def test_upsert_connection_error(self, mock_client_factory):
        """Test upsert with connection error"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        with patch('opensearchpy.helpers.bulk') as mock_bulk:
            mock_bulk.side_effect = Exception("ConnectionError: Unable to connect")
            
            store = OpenSearchColumnStore(index_name="test-index", embedding_dim=2)
            
            docs = [
                ColumnDoc(
                    column_id="test.col1",
                    column_name="col1",
                    embedding=[0.1, 0.2],
                    sample_values=["val1"],
                    metadata={"domain": "test"}
                )
            ]
            
            with pytest.raises(Exception):
                store.upsert_columns(docs)
                
    @patch('app.aoss.column_store.create_aoss_client')
    def test_semantic_search_empty_response(self, mock_client_factory):
        """Test semantic search with empty response"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.search.return_value = {"hits": {"hits": []}}
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        results = store.semantic_search([0.1, 0.2, 0.3])
        
        assert results == []
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_semantic_search_malformed_response(self, mock_client_factory):
        """Test semantic search with malformed response"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        mock_client.search.return_value = {}  # Missing "hits" key
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        results = store.semantic_search([0.1, 0.2, 0.3])
        
        assert results == []
        
    @patch('app.aoss.column_store.create_aoss_client')
    def test_get_columns_by_domain_custom_fields(self, mock_client_factory):
        """Test get_columns_by_domain with custom return fields"""
        mock_client = Mock()
        mock_client_factory.return_value = mock_client
        
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "column_id": "customer.name",
                            "column_name": "name",
                            "metadata": {"domain": "customer"},
                            "sample_values": ["John"]
                        }
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        
        store = OpenSearchColumnStore(index_name="test-index")
        
        results = store.get_columns_by_domain(
            "customer", 
            return_fields=["column_id", "column_name"]
        )
        
        assert len(results) == 1
        assert "column_id" in results[0]
        assert "column_name" in results[0]
        # Fields not in return_fields should have None values
        assert results[0].get("metadata") is None
        assert results[0].get("sample_values") is None
        
        # Verify _source parameter in search call
        search_args = mock_client.search.call_args[1]
        assert search_args["body"]["_source"] == ["column_id", "column_name"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])