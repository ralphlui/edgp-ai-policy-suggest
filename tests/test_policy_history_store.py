"""
Tests for the PolicyHistoryStore implementation
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json
from app.aoss.policy_history_store import PolicyHistoryStore, PolicyHistoryDoc

@pytest.fixture
def mock_aoss_client():
    mock = Mock()
    mock.indices.exists.return_value = False
    return mock

@pytest.fixture
def sample_policy_doc():
    return PolicyHistoryDoc(
        policy_id="test-policy-1",
        domain="customer",
        schema={
            "email": {"dtype": "string", "sample_values": ["test@example.com"]},
            "age": {"dtype": "integer", "sample_values": ["25", "30"]}
        },
        rules=[
            {
                "rule_name": "ExpectColumnValuesToBeValidEmail",
                "column_name": "email",
                "value": None
            },
            {
                "rule_name": "ExpectColumnValuesToBeBetween",
                "column_name": "age",
                "value": {"min_value": 0, "max_value": 120}
            }
        ],
        performance_metrics={
            "success_rate": 0.95,
            "validation_score": 0.92,
            "usage_count": 10
        },
        created_at=datetime.now(),
        updated_at=datetime.now(),
        embedding=[0.1] * 1536,
        metadata={
            "author": "test_user",
            "version": "1.0"
        }
    )

@pytest.mark.asyncio
async def test_policy_history_store_initialization(mock_aoss_client):
    """Test PolicyHistoryStore initialization and index creation"""
    with patch('app.aoss.policy_history_store.create_aoss_client', return_value=mock_aoss_client):
        store = PolicyHistoryStore(index_name="test-policy-index")
        
        # Verify index creation
        assert mock_aoss_client.indices.create.called
        create_body = mock_aoss_client.indices.create.call_args[1]['body']
        
        # Verify index mapping
        assert 'settings' in create_body
        assert 'mappings' in create_body
        assert create_body['settings']['index']['knn'] is True
        
        # Verify field mappings
        properties = create_body['mappings']['properties']
        assert 'policy_id' in properties
        assert 'domain' in properties
        assert 'embedding' in properties
        assert properties['embedding']['type'] == 'knn_vector'
        assert properties['embedding']['dimension'] == 1536

@pytest.mark.asyncio
async def test_store_policy(mock_aoss_client, sample_policy_doc):
    """Test storing a policy document"""
    # Mock the index response
    mock_aoss_client.index.return_value = {'_id': 'generated-id-123'}
    
    with patch('app.aoss.policy_history_store.create_aoss_client', return_value=mock_aoss_client):
        store = PolicyHistoryStore(index_name="test-policy-index")
        
        # Store the policy
        generated_id = await store.store_policy(sample_policy_doc)
        
        # Verify the document was indexed
        assert mock_aoss_client.index.called
        index_args = mock_aoss_client.index.call_args[1]
        
        assert index_args['index'] == "test-policy-index"
        assert 'id' not in index_args  # Should not include ID for AOSS
        assert 'refresh' not in index_args  # Should not include refresh parameter for AOSS
        
        # Verify document content
        doc = index_args['body']
        assert doc['domain'] == sample_policy_doc.domain
        assert doc['rules'] == json.dumps(sample_policy_doc.rules)  # Rules should be JSON serialized
        assert doc['embedding'] == sample_policy_doc.embedding
        
        # Verify returned ID
        assert generated_id == 'generated-id-123'

@pytest.mark.asyncio
async def test_retrieve_similar_policies(mock_aoss_client):
    """Test retrieving similar policies"""
    # Mock search response
    mock_aoss_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 0.95,
                    "_source": {
                        "policy_id": "test-1",
                        "domain": "customer",
                        "rules": [{"rule_name": "ExpectColumnValuesToBeValidEmail", "column_name": "email", "value": None}],
                        "performance_metrics": {"success_rate": 0.9}
                    }
                },
                {
                    "_score": 0.85,
                    "_source": {
                        "policy_id": "test-2",
                        "domain": "customer",
                        "rules": [{"rule_name": "ExpectColumnValuesToBeBetween", "column_name": "age", "value": {"min_value": 0, "max_value": 120}}],
                        "performance_metrics": {"success_rate": 0.85}
                    }
                }
            ]
        }
    }

    with patch('app.aoss.policy_history_store.create_aoss_client', return_value=mock_aoss_client):
        store = PolicyHistoryStore(index_name="test-policy-index")
        
        # Search for similar policies
        results = await store.retrieve_similar_policies(
            query_embedding=[0.1] * 1536,
            domain="customer",
            min_success_rate=0.8,
            top_k=2
        )
        
        # Verify search was called with correct parameters
        assert mock_aoss_client.search.called
        search_body = mock_aoss_client.search.call_args[1]['body']
        
        assert search_body['size'] == 2
        assert 'bool' in search_body['query']
        assert 'must' in search_body['query']['bool']
        assert 'filter' in search_body['query']['bool']
        
        # Verify results
        assert len(results) == 2
        assert results[0]['score'] == 0.95
        assert results[0]['policy_id'] == 'test-1'
        assert results[1]['score'] == 0.85
        assert results[1]['policy_id'] == 'test-2'

@pytest.mark.asyncio
async def test_get_domain_policies(mock_aoss_client):
    """Test retrieving all policies for a domain"""
    mock_aoss_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "policy_id": "test-1",
                        "domain": "customer",
                        "created_at": "2025-10-23T00:00:00",
                        "rules": [{"rule_name": "ExpectColumnValuesToBeValidEmail", "column_name": "email", "value": None}]
                    }
                }
            ]
        }
    }

    with patch('app.aoss.policy_history_store.create_aoss_client', return_value=mock_aoss_client):
        store = PolicyHistoryStore(index_name="test-policy-index")
        
        # Get domain policies
        policies = await store.get_domain_policies(
            domain="customer",
            sort_by="created_at",
            order="desc",
            limit=100
        )
        
        # Verify search parameters
        assert mock_aoss_client.search.called
        search_body = mock_aoss_client.search.call_args[1]['body']
        
        assert search_body['size'] == 100
        assert search_body['query']['term']['domain'] == 'customer'
        assert 'sort' in search_body
        
        # Verify results
        assert len(policies) == 1
        assert policies[0]['policy_id'] == 'test-1'
        assert policies[0]['domain'] == 'customer'

@pytest.mark.asyncio
async def test_update_policy_feedback(mock_aoss_client):
    """Test updating policy feedback"""
    # Mock the search response
    mock_aoss_client.search.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [{"_id": "internal-id-123"}]
        }
    }
    
    with patch('app.aoss.policy_history_store.create_aoss_client', return_value=mock_aoss_client):
        store = PolicyHistoryStore(index_name="test-policy-index")
        
        feedback = {
            "is_helpful": True,
            "comments": "Great suggestions"
        }
        
        # Update feedback
        await store.update_policy_feedback("test-policy-1", feedback)
        
        # Verify search call
        assert mock_aoss_client.search.called
        search_args = mock_aoss_client.search.call_args[1]
        assert search_args['index'] == "test-policy-index"
        assert search_args['body']['query']['term']['policy_id'] == "test-policy-1"
        
        # Verify update call
        assert mock_aoss_client.update.called
        update_args = mock_aoss_client.update.call_args[1]
        
        assert update_args['index'] == "test-policy-index"
        assert update_args['id'] == "internal-id-123"
        assert 'feedback' in update_args['body']['doc']['metadata']
        assert 'refresh' not in update_args  # Should not include refresh parameter for AOSS
        
@pytest.mark.asyncio
async def test_error_handling(mock_aoss_client):
    """Test error handling in PolicyHistoryStore"""
    mock_aoss_client.indices.create.side_effect = Exception("Failed to create index")
    
    with patch('app.aoss.policy_history_store.create_aoss_client', return_value=mock_aoss_client):
        with pytest.raises(Exception) as exc_info:
            store = PolicyHistoryStore(index_name="test-policy-index")
        assert "Failed to create index" in str(exc_info.value)