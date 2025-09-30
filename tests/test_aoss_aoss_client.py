"""
Tests for app/aoss/aoss_client.py module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os


class TestAOSSClientModule:
    """Test aoss_client.py module functionality"""
    
    def test_aoss_client_module_import(self):
        """Test aoss_client module can be imported"""
        from app.aoss import aoss_client
        assert hasattr(aoss_client, '__name__')

    def test_client_functions_exist(self):
        """Test client functions exist"""
        from app.aoss.aoss_client import create_aoss_client, test_aoss_connection
        
        assert callable(create_aoss_client)
        assert callable(test_aoss_connection)

    @patch('app.aoss.aoss_client.settings')
    @patch('app.aoss.aoss_client.boto3.Session')
    @patch('app.aoss.aoss_client.OpenSearch')
    def test_create_aoss_client_success(self, mock_opensearch, mock_session_class, mock_settings):
        """Test successful AOSS client creation"""
        # Setup settings mock
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = "us-east-1"
        
        # Setup boto3 mock
        mock_session = Mock()
        mock_credentials = Mock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        
        # Setup OpenSearch mock
        mock_client = Mock()
        mock_client.info.return_value = {"cluster_name": "test-cluster", "version": {"number": "1.0"}}
        mock_opensearch.return_value = mock_client
        
        from app.aoss.aoss_client import create_aoss_client
        
        result = create_aoss_client()
        
        assert result is not None
        mock_opensearch.assert_called_once()
        mock_session.get_credentials.assert_called_once()

    @patch('app.aoss.aoss_client.settings')
    def test_create_aoss_client_missing_host(self, mock_settings):
        """Test AOSS client creation with missing host"""
        mock_settings.aoss_host = None
        mock_settings.aws_region = "us-east-1"
        
        from app.aoss.aoss_client import create_aoss_client
        
        with pytest.raises(RuntimeError) as exc_info:
            create_aoss_client()
        
        assert "Missing AOSS host" in str(exc_info.value)

    @patch('app.aoss.aoss_client.settings')
    def test_create_aoss_client_missing_region(self, mock_settings):
        """Test AOSS client creation with missing region"""
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = None
        
        from app.aoss.aoss_client import create_aoss_client
        
        with pytest.raises(RuntimeError) as exc_info:
            create_aoss_client()
        
        assert "Missing AOSS host or region" in str(exc_info.value)

    @patch('app.aoss.aoss_client.settings')
    @patch('app.aoss.aoss_client.boto3.Session')
    def test_create_aoss_client_no_credentials(self, mock_session_class, mock_settings):
        """Test AOSS client creation with no credentials"""
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = "us-east-1"
        
        mock_session = Mock()
        mock_session.get_credentials.return_value = None
        mock_session_class.return_value = mock_session
        
        from app.aoss.aoss_client import create_aoss_client
        
        with pytest.raises(RuntimeError) as exc_info:
            create_aoss_client()
        
        assert "AWS credentials not found" in str(exc_info.value)

    @patch('app.aoss.aoss_client.settings')
    @patch('app.aoss.aoss_client.boto3.Session')
    @patch('app.aoss.aoss_client.OpenSearch')
    def test_create_aoss_client_connection_test_failure(self, mock_opensearch, mock_session_class, mock_settings):
        """Test AOSS client creation with connection test failure"""
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = "us-east-1"
        
        mock_session = Mock()
        mock_credentials = Mock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        
        mock_client = Mock()
        mock_client.info.side_effect = Exception("Connection failed")
        mock_opensearch.return_value = mock_client
        
        from app.aoss.aoss_client import create_aoss_client
        
        # Should not raise exception, but return client with warning
        result = create_aoss_client()
        assert result is not None

    @patch('app.aoss.aoss_client.settings')
    @patch('app.aoss.aoss_client.boto3.Session')
    @patch('app.aoss.aoss_client.OpenSearch')
    def test_create_aoss_client_authorization_error(self, mock_opensearch, mock_session_class, mock_settings):
        """Test AOSS client creation with authorization error"""
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = "us-east-1"
        
        mock_session = Mock()
        mock_credentials = Mock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        
        auth_error = Exception("AuthorizationException: Access denied")
        mock_opensearch.side_effect = auth_error
        
        from app.aoss.aoss_client import create_aoss_client
        
        with pytest.raises(Exception) as exc_info:
            create_aoss_client()
        
        assert "AuthorizationException" in str(exc_info.value)

    @patch('app.aoss.aoss_client.settings')
    @patch('app.aoss.aoss_client.boto3.Session')
    @patch('app.aoss.aoss_client.OpenSearch')
    def test_create_aoss_client_credential_error(self, mock_opensearch, mock_session_class, mock_settings):
        """Test AOSS client creation with credential error"""
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = "us-east-1"
        
        mock_session = Mock()
        mock_credentials = Mock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        
        cred_error = Exception("credentials error")
        mock_opensearch.side_effect = cred_error
        
        from app.aoss.aoss_client import create_aoss_client
        
        with pytest.raises(Exception) as exc_info:
            create_aoss_client()
        
        assert "credentials" in str(exc_info.value).lower()

    @patch('app.aoss.aoss_client.create_aoss_client')
    def test_test_aoss_connection_success(self, mock_create_client):
        """Test successful AOSS connection test"""
        mock_client = Mock()
        mock_client.info.return_value = {"cluster_name": "test"}
        mock_create_client.return_value = mock_client
        
        from app.aoss.aoss_client import test_aoss_connection
        
        result = test_aoss_connection()
        
        assert result is True
        mock_create_client.assert_called_once()
        mock_client.info.assert_called_once()

    @patch('app.aoss.aoss_client.create_aoss_client')
    def test_test_aoss_connection_failure(self, mock_create_client):
        """Test failed AOSS connection test"""
        mock_create_client.side_effect = Exception("Connection failed")
        
        from app.aoss.aoss_client import test_aoss_connection
        
        result = test_aoss_connection()
        
        assert result is False

    @patch('app.aoss.aoss_client.os.getenv')
    @patch('app.aoss.aoss_client.settings')
    @patch('app.aoss.aoss_client.boto3.Session')
    @patch('app.aoss.aoss_client.OpenSearch')
    def test_create_aoss_client_env_vars(self, mock_opensearch, mock_session_class, mock_settings, mock_getenv):
        """Test AOSS client creation with environment variables"""
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = "us-east-1"
        
        mock_getenv.return_value = "test-key"  # Mock AWS_ACCESS_KEY_ID
        
        mock_session = Mock()
        mock_credentials = Mock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        
        mock_client = Mock()
        mock_client.info.return_value = {"cluster_name": "test"}
        mock_opensearch.return_value = mock_client
        
        from app.aoss.aoss_client import create_aoss_client
        
        result = create_aoss_client()
        
        assert result is not None

    @patch('app.aoss.aoss_client.settings')
    @patch('app.aoss.aoss_client.boto3.Session')
    @patch('app.aoss.aoss_client.OpenSearch')
    def test_create_aoss_client_custom_timeout(self, mock_opensearch, mock_session_class, mock_settings):
        """Test AOSS client creation with custom timeout"""
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = "us-east-1"
        
        mock_session = Mock()
        mock_credentials = Mock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        
        mock_client = Mock()
        mock_client.info.return_value = {"cluster_name": "test"}
        mock_opensearch.return_value = mock_client
        
        from app.aoss.aoss_client import create_aoss_client
        
        result = create_aoss_client(timeout_sec=60)
        
        assert result is not None
        # Verify timeout was passed to OpenSearch constructor
        call_args = mock_opensearch.call_args
        assert call_args[1]['timeout'] == 60


class TestAOSSClientUtilities:
    """Test AOSS client utility functions and edge cases"""
    
    @patch('app.aoss.aoss_client.AWSV4SignerAuth')
    @patch('app.aoss.aoss_client.settings')
    @patch('app.aoss.aoss_client.boto3.Session')
    @patch('app.aoss.aoss_client.OpenSearch')
    def test_awsv4_signer_auth_creation(self, mock_opensearch, mock_session_class, mock_settings, mock_auth):
        """Test AWSV4SignerAuth creation"""
        mock_settings.aoss_host = "test-host.aoss.amazonaws.com"
        mock_settings.aws_region = "us-east-1"
        
        mock_session = Mock()
        mock_credentials = Mock()
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        
        mock_client = Mock()
        mock_opensearch.return_value = mock_client
        
        from app.aoss.aoss_client import create_aoss_client
        
        create_aoss_client()
        
        # Verify AWSV4SignerAuth was called with correct parameters
        mock_auth.assert_called_once_with(mock_credentials, "us-east-1", "aoss")

    @patch('app.aoss.aoss_client.logger')
    def test_logging_functionality(self, mock_logger):
        """Test logging functionality in AOSS client"""
        from app.aoss import aoss_client
        
        # Just verify module loads with logger
        assert hasattr(aoss_client, '__name__')

    def test_module_constants(self):
        """Test module-level constants and imports"""
        from app.aoss.aoss_client import create_aoss_client, test_aoss_connection
        
        # Verify functions are importable
        assert callable(create_aoss_client)
        assert callable(test_aoss_connection)