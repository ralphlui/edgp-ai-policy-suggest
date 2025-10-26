"""
Test configuration for RAG implementation
"""
import pytest
import os
from unittest.mock import patch, Mock, MagicMock
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    with patch.dict(os.environ, {
        'POLICY_HISTORY_INDEX': 'test-policy-history',
        'POLICY_RETENTION_DAYS': '365',
        'POLICY_MIN_SUCCESS_RATE': '0.7',
        'EMBED_MODEL': 'text-embedding-3-small',
        'EMBED_DIM': '1536',
        'AOSS_HOST': 'mock-aoss-host.test.local',
        'AWS_REGION': 'ap-southeast-1'
    }):
        yield

@pytest.fixture(autouse=True)
def mock_opensearch():
    """Mock OpenSearch client and all its dependencies"""
    mock_client = MagicMock()
    mock_client.info.return_value = {
        'cluster_name': 'mock-cluster',
        'version': {'number': '1.0.0'}
    }
    mock_client.indices.exists.return_value = True
    mock_client.indices.create.return_value = {"acknowledged": True}
    mock_client.search.return_value = {
        'hits': {
            'hits': [
                {
                    '_source': {
                        'policy_id': 'test-1',
                        'domain': 'test',
                        'rules': [],
                        'performance_metrics': {
                            'success_rate': 0.9
                        }
                    },
                    '_score': 0.95
                }
            ]
        }
    }
    mock_client.bulk.return_value = {
        "took": 30,
        "errors": False,
        "items": []
    }
    
    with patch.multiple(
        'opensearchpy.OpenSearch',
        __init__=Mock(return_value=None),
        info=mock_client.info,
        indices=mock_client.indices,
        search=mock_client.search,
        bulk=mock_client.bulk
    ), patch(
        'app.aoss.aoss_client.create_aoss_client',
        return_value=mock_client
    ), patch(
        'app.aoss.aoss_client.test_aoss_connection',
        return_value=True
    ), patch(
        'requests.Session.send',
        return_value=MagicMock(
            status_code=200,
            content=b'{"acknowledged":true}',
            ok=True
        )
    ):
        yield mock_client

@pytest.fixture(autouse=True)
def mock_aws_auth():
    """Mock AWS authentication"""
    with patch('app.aoss.aoss_client.boto3.Session') as mock_session:
        mock_creds = Mock()
        mock_creds.access_key = 'test-key'
        mock_creds.secret_key = 'test-secret'
        mock_creds.token = 'test-token'
        
        session_instance = Mock()
        session_instance.get_credentials.return_value = mock_creds
        mock_session.return_value = session_instance
        
        yield mock_session

@pytest.fixture(autouse=True)
def mock_requests():
    """Mock the underlying requests library"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'{"acknowledged":true}'
    mock_response.ok = True
    
    with patch('requests.Session') as mock_session:
        mock_session.return_value.send.return_value = mock_response
        yield mock_session