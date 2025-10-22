"""
Test configuration for RAG implementation
"""
import pytest
import os
from unittest.mock import patch

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    with patch.dict(os.environ, {
        'POLICY_HISTORY_INDEX': 'test-policy-history',
        'POLICY_RETENTION_DAYS': '365',
        'POLICY_MIN_SUCCESS_RATE': '0.7',
        'EMBED_MODEL': 'text-embedding-3-small',
        'EMBED_DIM': '1536'
    }):
        yield