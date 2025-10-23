"""
Tests for the RuleRAGEnhancer implementation
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json
from app.agents.rule_rag_enhancer import RuleRAGEnhancer

@pytest.fixture
def sample_schema():
    return {
        "email": {
            "dtype": "string",
            "sample_values": ["test@example.com"]
        },
        "age": {
            "dtype": "integer",
            "sample_values": ["25", "30"]
        }
    }

@pytest.fixture
def sample_similar_policies():
    return [
        {
            "score": 0.95,
            "policy_id": "test-1",
            "domain": "customer",
            "rules": [
                {
                    "type": "email_validation",
                    "parameters": {"field": "email"}
                }
            ],
            "performance_metrics": {
                "success_rate": 0.9,
                "validation_score": 0.92
            }
        },
        {
            "score": 0.85,
            "policy_id": "test-2",
            "domain": "customer",
            "rules": [
                {
                    "type": "range_check",
                    "parameters": {
                        "field": "age",
                        "min": 0,
                        "max": 120
                    }
                }
            ],
            "performance_metrics": {
                "success_rate": 0.85,
                "validation_score": 0.88
            }
        }
    ]

@pytest.mark.asyncio
async def test_enhance_prompt_with_history(sample_schema, sample_similar_policies):
    """Test prompt enhancement with historical context"""
    with patch('app.agents.rule_rag_enhancer.embed_column_names_batched_async') as mock_embed:
        mock_embed.return_value = [[0.1] * 1536]  # Mock embedding
        
        # Mock policy store
        mock_policy_store = AsyncMock()
        mock_policy_store.retrieve_similar_policies.return_value = sample_similar_policies
        
        with patch('app.agents.rule_rag_enhancer.PolicyHistoryStore', return_value=mock_policy_store):
            enhancer = RuleRAGEnhancer()
            
            # Test prompt enhancement
            enhanced_prompt = await enhancer.enhance_prompt_with_history(
                schema=sample_schema,
                domain="customer"
            )
            
            # Verify embedding was called
            assert mock_embed.called
            
            # Verify policy retrieval was called
            assert mock_policy_store.retrieve_similar_policies.called
            retrieval_args = mock_policy_store.retrieve_similar_policies.call_args[1]
            assert retrieval_args['domain'] == "customer"
            assert retrieval_args['min_success_rate'] == 0.8
            
            # Verify prompt content
            assert "current schema" in enhanced_prompt.lower()
            assert "successful validation rules" in enhanced_prompt.lower()
            assert "email" in enhanced_prompt
            assert "age" in enhanced_prompt

@pytest.mark.asyncio
async def test_store_successful_policy(sample_schema):
    """Test storing a successful policy"""
    with patch('app.agents.rule_rag_enhancer.embed_column_names_batched_async') as mock_embed:
        mock_embed.return_value = [[0.1] * 1536]  # Mock embedding
        
        # Mock policy store
        mock_policy_store = AsyncMock()
        
        with patch('app.agents.rule_rag_enhancer.PolicyHistoryStore', return_value=mock_policy_store):
            enhancer = RuleRAGEnhancer()
            
            # Test storing policy
            rules = [
                {
                    "type": "email_validation",
                    "parameters": {"field": "email"}
                }
            ]
            performance_metrics = {
                "success_rate": 0.95,
                "validation_score": 0.92
            }
            
            await enhancer.store_successful_policy(
                domain="customer",
                schema=sample_schema,
                rules=rules,
                performance_metrics=performance_metrics
            )
            
            # Verify embedding was called
            assert mock_embed.called
            
            # Verify policy storage was called
            assert mock_policy_store.store_policy.called
            store_args = mock_policy_store.store_policy.call_args[0][0]
            
            assert store_args.domain == "customer"
            assert store_args.schema == sample_schema
            assert store_args.rules == rules
            assert store_args.performance_metrics == performance_metrics
            assert len(store_args.embedding) == 1536



@pytest.mark.asyncio
async def test_enhance_prompt_with_no_similar_policies():
    """Test prompt enhancement when no similar policies are found"""
    with patch('app.agents.rule_rag_enhancer.embed_column_names_batched_async') as mock_embed:
        mock_embed.return_value = [[0.1] * 1536]  # Mock embedding
        
        # Mock policy store with no results
        mock_policy_store = AsyncMock()
        mock_policy_store.retrieve_similar_policies.return_value = []
        
        with patch('app.agents.rule_rag_enhancer.PolicyHistoryStore', return_value=mock_policy_store):
            enhancer = RuleRAGEnhancer()
            
            schema = {"test_field": {"dtype": "string", "sample_values": ["test"]}}
            
            # Test prompt enhancement
            enhanced_prompt = await enhancer.enhance_prompt_with_history(
                schema=schema,
                domain="test"
            )
            
            # Verify prompt still contains basic instructions
            assert "current schema" in enhanced_prompt.lower()
            assert "suggest appropriate validation rules" in enhanced_prompt.lower()

