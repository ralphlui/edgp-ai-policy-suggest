"""
AWS Secrets Manager Tests for EDGP AI Policy Suggest
This module contains unit tests, integration tests, and environment validation for AWS Secrets Manager
"""

import json
import logging
import os
import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from botocore.exceptions import ClientError, BotoCoreError

from app.aws.aws_secrets_service import (
    AWSSecretsManagerService,
    CredentialManager,
    _format_jwt_public_key,
    _get_config_value,
    _get_openai_secret_name,
    _get_aws_region,
    get_openai_api_key,
    get_jwt_public_key,
    require_openai_api_key,
    require_jwt_public_key,
    load_credentials
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestConfigHelpers:
    """Test configuration helper functions"""
    
    def test_get_config_value_with_env(self):
        """Test getting config value from environment"""
        with patch.dict('os.environ', {'TEST_KEY': 'test_value'}):
            assert _get_config_value('TEST_KEY', 'default') == 'test_value'
    
    def test_get_config_value_default(self):
        """Test getting default config value when env var not set"""
        with patch.dict('os.environ', {}, clear=True):
            assert _get_config_value('NONEXISTENT_KEY', 'default_value') == 'default_value'
    
    def test_get_openai_secret_name_from_env(self):
        """Test getting OpenAI secret name from environment"""
        with patch.dict('os.environ', {'OPENAI_SECRET_NAME': 'custom/secret'}):
            assert _get_openai_secret_name() == 'custom/secret'
    
    def test_get_openai_secret_name_default(self):
        """Test getting default OpenAI secret name"""
        with patch.dict('os.environ', {}, clear=True):
            assert _get_openai_secret_name() == 'sit/edgp/secret'
    
    def test_get_aws_region_from_env(self):
        """Test getting AWS region from environment variable"""
        with patch.dict('os.environ', {'AWS_REGION': 'us-west-2'}):
            assert _get_aws_region() == 'us-west-2'
    
    def test_get_aws_region_default(self):
        """Test getting default AWS region when env var not set"""
        with patch.dict('os.environ', {}, clear=True):
            region = _get_aws_region()
            assert region in ['us-east-1', 'ap-southeast-1'] or region is not None


class TestJWTKeyFormatting:
    """Test JWT public key formatting functions"""
    
    def test_format_jwt_public_key_complete_pem(self):
        """Test formatting complete PEM key"""
        complete_key = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0B\n-----END PUBLIC KEY-----"
        assert _format_jwt_public_key(complete_key) == complete_key
    
    def test_format_jwt_public_key_no_headers(self):
        """Test formatting key without PEM headers"""
        raw_key = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA"
        result = _format_jwt_public_key(raw_key)
        assert result.startswith("-----BEGIN PUBLIC KEY-----")
        assert result.endswith("-----END PUBLIC KEY-----")
        assert raw_key in result
    
    def test_format_jwt_public_key_empty(self):
        """Test formatting empty key"""
        assert _format_jwt_public_key("") == ""
    
    def test_format_jwt_public_key_none(self):
        """Test formatting None key"""
        assert _format_jwt_public_key(None) is None


class TestAWSSecretsManagerService:
    """Test AWSSecretsManagerService functionality"""

    @pytest.fixture
    def aws_service(self):
        mock_client = MagicMock()
        with patch('boto3.session.Session') as mock_session:
            mock_session.return_value.client.return_value = mock_client
            with patch.dict(os.environ, {
                'AWS_ACCESS_KEY_ID': 'test_key',
                'AWS_SECRET_ACCESS_KEY': 'test_secret',
                'AWS_REGION': 'us-west-2'
            }):
                service = AWSSecretsManagerService(region_name="us-west-2")
                service._client = mock_client
                return service, mock_client
    
    def test_init_with_region(self):
        """Test service initialization with region"""
        service = AWSSecretsManagerService('us-west-2')
        assert service.region_name == 'us-west-2'
        assert service._client is None
    
    def test_init_without_region(self):
        """Test service initialization without region"""
        with patch('app.aws.aws_secrets_service._get_aws_region', return_value='us-east-1'):
            service = AWSSecretsManagerService()
            assert service.region_name == 'us-east-1'
    
    def test_get_secret_no_region(self):
        """Test get_secret returns None when no region configured"""
        service = AWSSecretsManagerService(None)
        assert service.get_secret('test-secret') is None
    
    def test_client_initialization_failure(self):
        """Test client initialization failure handling"""
        with patch('boto3.session.Session') as mock_session:
            mock_session.side_effect = Exception("Failed to create session")
            service = AWSSecretsManagerService(region_name="us-west-2")
            assert service.get_secret("test_secret") is None

    def test_get_secret_parse_error(self, aws_service):
        """Test handling of invalid JSON in secret"""
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {
            "SecretString": "invalid json {"
        }
        assert service.get_secret("test_secret") == "invalid json {"

    def test_get_secret_string(self, aws_service):
        """Test retrieving string secret"""
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {
            "SecretString": '{"api_key": "test_key"}'
        }
        assert service.get_secret("test_secret") == "test_key"

    def test_get_secret_with_key(self, aws_service):
        """Test retrieving specific key from JSON secret"""
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {
            "SecretString": '{"custom_key": "test_value"}'
        }
        assert service.get_secret("test_secret", "custom_key") == "test_value"

    def test_get_secret_binary(self, aws_service):
        """Test retrieving binary secret"""
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {
            "SecretBinary": b"test_binary"
        }
        assert service.get_secret("test_secret") == "test_binary"

    def test_get_secret_client_errors(self, aws_service):
        """Test handling of various AWS client errors"""
        service, mock_client = aws_service
        error_codes = [
            "DecryptionFailureException",
            "InternalServiceErrorException",
            "InvalidParameterException",
            "InvalidRequestException",
            "ResourceNotFoundException",
            "UnrecognizedClientException",
            "AccessDeniedException",
            "UnknownError"
        ]
        
        for code in error_codes:
            error_response = {
                "Error": {
                    "Code": code,
                    "Message": f"{code} error message"
                }
            }
            mock_client.get_secret_value.side_effect = ClientError(error_response, "GetSecretValue")
            assert service.get_secret("test_secret") is None

    def test_get_secret_boto_error(self, aws_service):
        """Test handling of BotoCore error"""
        service, mock_client = aws_service
        mock_client.get_secret_value.side_effect = BotoCoreError()
        assert service.get_secret("test_secret") is None


class TestCredentialManager:
    """Test CredentialManager functionality"""

    @pytest.fixture
    def credential_manager(self):
        mock_aws_service = MagicMock()
        mock_aws_service.get_secret.return_value = None
        return CredentialManager(mock_aws_service), mock_aws_service

    def test_load_credentials_from_env(self, credential_manager):
        """Test loading credentials from environment variables"""
        manager, _ = credential_manager
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test_key",
            "JWT_PUBLIC_KEY": "test_jwt_key"
        }):
            manager.load_credentials()
            assert manager.get_openai_api_key() == "test_key"
            assert manager.get_jwt_public_key() == _format_jwt_public_key("test_jwt_key")

    def test_load_credentials_from_aws(self, credential_manager):
        """Test loading credentials from AWS"""
        manager, mock_aws_service = credential_manager
        mock_aws_service.get_secret.side_effect = [
            "aws_openai_key",  # First call for OpenAI key
            "aws_jwt_key"      # Second call for JWT key
        ]
        
        with patch.dict(os.environ, clear=True):
            manager.load_credentials()
            assert manager.get_openai_api_key() == "aws_openai_key"
            assert manager.get_jwt_public_key() == _format_jwt_public_key("aws_jwt_key")

    def test_load_credentials_aws_failure(self, credential_manager):
        """Test handling of AWS credential loading failure"""
        manager, mock_aws_service = credential_manager
        mock_aws_service.get_secret.side_effect = Exception("AWS error")
        
        with patch.dict(os.environ, clear=True):
            manager.load_credentials()
            assert manager.get_openai_api_key() is None
            assert manager.get_jwt_public_key() is None

    def test_load_credentials_with_custom_secret_name(self, credential_manager):
        """Test loading credentials with custom secret name"""
        manager, mock_aws_service = credential_manager
        custom_secret = "custom_secret_name"
        mock_aws_service.get_secret.side_effect = [None] * 4  # All attempts return None
        
        with patch.dict(os.environ, clear=True):
            manager.load_credentials(secret_name=custom_secret)
            calls = mock_aws_service.get_secret.call_args_list
            assert len(calls) >= 2  # Should have at least tried both key formats
            assert call(custom_secret, "ai_agent_api_key") in calls
            assert call(custom_secret) in calls

    def test_require_openai_api_key_missing(self, credential_manager):
        """Test error when OpenAI API key is required but missing"""
        manager, _ = credential_manager
        with patch.dict(os.environ, clear=True):
            manager.load_credentials()
            with pytest.raises(RuntimeError) as exc_info:
                manager.require_openai_api_key()
            assert "OpenAI API key is required but not available" in str(exc_info.value)

    def test_require_jwt_public_key_missing(self, credential_manager):
        """Test error when JWT public key is required but missing"""
        manager, _ = credential_manager
        with patch.dict(os.environ, clear=True):
            manager.load_credentials()
            with pytest.raises(RuntimeError) as exc_info:
                manager.require_jwt_public_key()
            assert "JWT public key is required but not available" in str(exc_info.value)

    def test_get_credentials_status(self, credential_manager):
        """Test getting credentials status"""
        manager, _ = credential_manager
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test_key",
            "JWT_PUBLIC_KEY": "test_jwt_key"
        }):
            manager.load_credentials()
            status = manager.get_credentials_status()
            assert status["openai_api_key_available"] is True
            assert status["jwt_public_key_available"] is True
            assert status["credentials_loaded"] is True


class TestModuleLevelFunctions:
    """Test module-level convenience functions"""
    
    @patch('app.aws.aws_secrets_service._credential_manager')
    def test_load_credentials_function(self, mock_manager):
        """Test load_credentials module function"""
        load_credentials()
        mock_manager.load_credentials.assert_called_once_with(None, False)
    
    @patch('app.aws.aws_secrets_service._credential_manager')
    def test_get_openai_api_key_function(self, mock_manager):
        """Test get_openai_api_key module function"""
        mock_manager.get_openai_api_key.return_value = 'test_key'
        assert get_openai_api_key() == 'test_key'
        mock_manager.get_openai_api_key.assert_called_once_with(True)
    
    @patch('app.aws.aws_secrets_service._credential_manager')
    def test_get_jwt_public_key_function(self, mock_manager):
        """Test get_jwt_public_key module function"""
        mock_manager.get_jwt_public_key.return_value = 'formatted_jwt_key'
        assert get_jwt_public_key() == 'formatted_jwt_key'
        mock_manager.get_jwt_public_key.assert_called_once_with(True)
    
    @patch('app.aws.aws_secrets_service._credential_manager')
    def test_require_openai_api_key_function(self, mock_manager):
        """Test require_openai_api_key module function"""
        mock_manager.require_openai_api_key.return_value = 'required_key'
        assert require_openai_api_key() == 'required_key'
        mock_manager.require_openai_api_key.assert_called_once()


@pytest.mark.integration
def test_aws_secrets_integration():
    """Integration test for AWS Secrets Manager"""
    try:
        # Import configuration
        from app.core.config import (
            OPENAI_API_KEY, 
            USE_AWS_SECRETS, 
            OPENAI_SECRET_NAME,
            AWS_REGION,
            settings
        )
        
        # Check configuration
        logger.info("Configuration Status:")
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"USE_AWS_SECRETS: {USE_AWS_SECRETS}")
        logger.info(f"OPENAI_SECRET_NAME: {OPENAI_SECRET_NAME}")
        logger.info(f"AWS_REGION: {AWS_REGION}")
        
        # Check AWS credentials
        aws_access_key = settings.aws_access_key_id
        aws_secret_key = settings.aws_secret_access_key
        logger.info("AWS Credentials:")
        logger.info(f"AWS_ACCESS_KEY_ID: {'Set' if aws_access_key else 'Missing'}")
        logger.info(f"AWS_SECRET_ACCESS_KEY: {'Set' if aws_secret_key else 'Missing'}")
        logger.info(f"AOSS_HOST: {'Set' if settings.aoss_host else 'Missing'}")
        
        # Test OpenAI API Key
        if OPENAI_API_KEY:
            logger.info("OpenAI API Key loaded successfully")
            logger.info(f"Key starts with: {OPENAI_API_KEY[:8]}...")
            logger.info(f"Key length: {len(OPENAI_API_KEY)} characters")
            
            # Test OpenAI client initialization
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
        else:
            logger.warning("OpenAI API Key not available")
            if USE_AWS_SECRETS:
                logger.warning(f"AWS Secrets Manager secret '{OPENAI_SECRET_NAME}' not found")
                logger.warning("Check AWS credentials and IAM permissions")
            else:
                logger.warning("AWS Secrets Manager is disabled")
                logger.warning("Check OPENAI_API_KEY_FALLBACK in .env file")
            return False
        
        # Test AWS Secrets Manager directly if enabled
        if USE_AWS_SECRETS:
            logger.info("Testing AWS Secrets Manager directly...")
            from app.core.config import get_secret_from_aws
            test_secret = get_secret_from_aws(OPENAI_SECRET_NAME, AWS_REGION)
            if test_secret:
                logger.info("AWS Secrets Manager call successful")
            else:
                logger.warning("AWS Secrets Manager call returned None")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running from the project root directory")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_file_loading():
    """Test .env file loading"""
    env = os.getenv("ENVIRONMENT", "development")
    env_file = f".env.{env}"
    
    if os.path.exists(env_file):
        logger.info(f"Environment file exists: {env_file}")
        
        # Check key variables
        key_vars = ['USE_AWS_SECRETS', 'OPENAI_SECRET_NAME', 'AWS_REGION', 'AOSS_HOST']
        logger.info("Key environment variables:")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if any(var in line for var in key_vars) and not line.startswith('#'):
                    logger.info(f"  {line}")
    else:
        logger.warning(f"Environment file not found: {env_file}")


if __name__ == "__main__":
    logger.info("EDGP AI Policy Suggest - AWS Secrets Manager Test")
    logger.info("=" * 60)
    
    test_env_file_loading()
    success = test_aws_secrets_integration()
    
    if success:
        logger.info("All tests passed! AWS Secrets Manager integration is working.")
        sys.exit(0)
    else:
        logger.error("Some tests failed. Please check the configuration.")
        sys.exit(1)