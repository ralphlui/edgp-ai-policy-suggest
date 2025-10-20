#!/usr/bin/env python3
"""
AWS Secrets Manager Tests for EDGP AI Policy Suggest
This module contains both unit tests and integration tests for AWS Secrets Manager
"""
import os
import sys
import json
import logging
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch
import pytest
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
    require_jwt_public_key
)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Unit Tests
@pytest.mark.unit
def test_get_config_value():
    with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
        assert _get_config_value("TEST_KEY", "default") == "test_value"
        assert _get_config_value("NONEXISTENT_KEY", "default") == "default"

@pytest.mark.unit
def test_get_openai_secret_name():
    with patch.dict(os.environ, {"OPENAI_SECRET_NAME": "custom_secret"}):
        assert _get_openai_secret_name() == "custom_secret"
    with patch.dict(os.environ, clear=True):
        assert _get_openai_secret_name() == "sit/edgp/secret"

@pytest.mark.unit
def test_get_aws_region():
    with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}):
        assert _get_aws_region() == "us-west-2"
    with patch.dict(os.environ, clear=True):
        assert _get_aws_region() == "ap-southeast-1"

@pytest.mark.unit
def test_format_jwt_public_key():
    # Test with already formatted key
    formatted_key = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA
-----END PUBLIC KEY-----"""
    assert _format_jwt_public_key(formatted_key) == formatted_key

    # Test with raw key
    raw_key = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA"
    expected = f"-----BEGIN PUBLIC KEY-----\n{raw_key}\n-----END PUBLIC KEY-----"
    assert _format_jwt_public_key(raw_key) == expected

    # Test with empty input
    assert _format_jwt_public_key("") == ""
    assert _format_jwt_public_key(None) == None

@pytest.mark.unit
class TestAWSSecretsManagerService:
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

    def test_client_initialization(self, aws_service):
        service, mock_client = aws_service
        assert service.client == mock_client

    def test_client_initialization_no_region(self):
        with patch.dict(os.environ, clear=True):
            with patch("boto3.session.Session") as mock_session:
                mock_session.return_value.client.side_effect = Exception("Failed to create client")
                service = AWSSecretsManagerService(region_name=None)
                assert service.get_secret("test_secret") is None

    def test_client_initialization_failure(self):
        with patch('boto3.session.Session') as mock_session:
            mock_session.side_effect = Exception("Failed to create session")
            service = AWSSecretsManagerService(region_name="us-west-2")
            assert service.get_secret("test_secret") is None

    def test_get_secret_parse_error(self, aws_service):
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {
            "SecretString": "invalid json {"
        }
        assert service.get_secret("test_secret") == "invalid json {"
        
    def test_get_secret_with_empty_response(self, aws_service):
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {}
        assert service.get_secret("test_secret") is None

    def test_get_secret_string(self, aws_service):
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {
            "SecretString": '{"api_key": "test_key"}'
        }
        assert service.get_secret("test_secret") == "test_key"

    def test_get_secret_with_key(self, aws_service):
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {
            "SecretString": '{"custom_key": "test_value"}'
        }
        assert service.get_secret("test_secret", "custom_key") == "test_value"

    def test_get_secret_binary(self, aws_service):
        service, mock_client = aws_service
        mock_client.get_secret_value.return_value = {
            "SecretBinary": b"test_binary"
        }
        assert service.get_secret("test_secret") == "test_binary"

    def test_get_secret_client_errors(self, aws_service):
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
        service, mock_client = aws_service
        mock_client.get_secret_value.side_effect = BotoCoreError()
        assert service.get_secret("test_secret") is None

@pytest.mark.unit
class TestCredentialManager:
    @pytest.fixture
    def credential_manager(self):
        mock_aws_service = MagicMock()
        mock_aws_service.get_secret.return_value = None
        return CredentialManager(mock_aws_service), mock_aws_service

    def test_load_credentials_from_env(self, credential_manager):
        manager, _ = credential_manager
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test_key",
            "JWT_PUBLIC_KEY": "test_jwt_key"
        }):
            manager.load_credentials()
            assert manager.get_openai_api_key() == "test_key"
            assert manager.get_jwt_public_key() == _format_jwt_public_key("test_jwt_key")

    def test_load_credentials_from_aws(self, credential_manager):
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
        manager, mock_aws_service = credential_manager
        mock_aws_service.get_secret.side_effect = Exception("AWS error")
        
        with patch.dict(os.environ, clear=True):
            manager.load_credentials()
            assert manager.get_openai_api_key() is None
            assert manager.get_jwt_public_key() is None

    def test_load_credentials_with_custom_secret_name(self, credential_manager):
        manager, mock_aws_service = credential_manager
        custom_secret = "custom_secret_name"
        mock_aws_service.get_secret.side_effect = [
            None,  # First attempt with ai_agent_api_key
            None,  # Second attempt with openai_api_key
            None,  # Third attempt with jwt_public_key
            None,  # Fourth attempt with all keys
        ]
        
        with patch.dict(os.environ, clear=True):
            manager.load_credentials(secret_name=custom_secret)
            # Check if get_secret was called with the correct secret name and keys
            calls = mock_aws_service.get_secret.call_args_list
            assert len(calls) >= 2  # Should have at least tried both key formats
            assert mock.call(custom_secret, "ai_agent_api_key") in calls
            assert mock.call(custom_secret) in calls

    def test_require_openai_api_key_missing(self, credential_manager):
        manager, mock_aws_service = credential_manager
        mock_aws_service.get_secret.return_value = None
        
        with patch.dict(os.environ, clear=True):
            manager.load_credentials()
            with pytest.raises(RuntimeError) as exc_info:
                manager.require_openai_api_key()
            assert "OpenAI API key is required but not available" in str(exc_info.value)

    def test_require_jwt_public_key_missing(self, credential_manager):
        manager, mock_aws_service = credential_manager
        mock_aws_service.get_secret.return_value = None
        
        with patch.dict(os.environ, clear=True):
            manager.load_credentials()
            with pytest.raises(RuntimeError) as exc_info:
                manager.require_jwt_public_key()
            assert "JWT public key is required but not available" in str(exc_info.value)

    def test_get_credentials_status(self, credential_manager):
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

    def test_force_reload(self, credential_manager):
        manager, mock_aws_service = credential_manager
        mock_aws_service.get_secret.side_effect = [
            "first_key",   # First load OpenAI key
            None,         # First load JWT key
            "second_key", # Force reload OpenAI key
            None         # Force reload JWT key
        ]
        
        with patch.dict(os.environ, clear=True):
            # First load
            manager.load_credentials()
            assert manager.get_openai_api_key() == "first_key"
            
            # Force reload
            manager.load_credentials(force_reload=True)
            assert manager.get_openai_api_key() == "second_key"

# Test global helper functions
@pytest.mark.unit
def test_global_helper_functions():
    # Test all global helper functions with environment variables
    test_api_key = "global-test-key"
    test_jwt_key = "global-jwt-key"
    
    # Create a mock AWS service
    mock_aws_service = MagicMock()
    mock_aws_service.get_secret.return_value = None  # Ensure it falls back to env vars
    
    # Create a test credential manager
    test_credential_manager = CredentialManager(mock_aws_service)
    
    # Patch the global credential manager
    with patch('app.aws.aws_secrets_service._credential_manager', test_credential_manager):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": test_api_key,
            "JWT_PUBLIC_KEY": test_jwt_key
        }, clear=True):  # Clear=True to ensure no other env vars interfere
            # Load credentials to initialize the manager
            test_credential_manager.load_credentials()
            
            # Test getters
            assert get_openai_api_key() == test_api_key
            assert get_jwt_public_key() == _format_jwt_public_key(test_jwt_key)
            
            # Test requirers
            assert require_openai_api_key() == test_api_key
            assert require_jwt_public_key() == _format_jwt_public_key(test_jwt_key)

# Integration Tests
@pytest.mark.integration
def test_aws_secrets_integration():
    
    try:
        # Test importing the configuration
        print(" Importing configuration...")
        from app.core.config import (
            OPENAI_API_KEY, 
            USE_AWS_SECRETS, 
            OPENAI_SECRET_NAME,
            AWS_REGION,
            settings
        )
        
        # Display configuration status
        print(f"\n Configuration Status:")
        print(f"   Environment: {settings.environment}")
        print(f"   USE_AWS_SECRETS: {USE_AWS_SECRETS}")
        print(f"   OPENAI_SECRET_NAME: {OPENAI_SECRET_NAME}")
        print(f"   AWS_REGION: {AWS_REGION}")
        
        # Check AWS credentials
        aws_access_key = settings.aws_access_key_id
        aws_secret_key = settings.aws_secret_access_key
        
        print(f"\n AWS Credentials:")
        print(f"   AWS_ACCESS_KEY_ID: {' Set' if aws_access_key else ' Missing'}")
        print(f"   AWS_SECRET_ACCESS_KEY: {' Set' if aws_secret_key else ' Missing'}")
        print(f"   AOSS_HOST: {' Set' if settings.aoss_host else ' Missing'}")
        
        # Test OpenAI API Key
        print(f"\n OpenAI API Key:")
        if OPENAI_API_KEY:
            print(f"    OpenAI API Key loaded successfully")
            print(f"   Key starts with: {OPENAI_API_KEY[:8]}...")
            print(f"   Key length: {len(OPENAI_API_KEY)} characters")
            
            # Test OpenAI client initialization
            print(f"\n🧪 Testing OpenAI Client...")
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                print(f"    OpenAI client initialized successfully")
                
                # Test a simple API call (optional - uncomment if you want to test API)
                # print(f"   🌐 Testing API connection...")
                # models = client.models.list()
                # print(f"    API connection successful - {len(models.data)} models available")
                
            except Exception as e:
                print(f"    OpenAI client initialization failed: {e}")
                return False
        else:
            print(f"   OpenAI API Key not available")
            print(f"   Possible causes:")
            if USE_AWS_SECRETS:
                print(f"     - AWS Secrets Manager secret '{OPENAI_SECRET_NAME}' not found")
                print(f"     - AWS credentials not configured properly")
                print(f"     - IAM permissions missing for secretsmanager:GetSecretValue")
            else:
                print(f"     - AWS Secrets Manager is disabled")
                print(f"     - OPENAI_API_KEY_FALLBACK not set in .env file")
            return False
        
        # Test AWS Secrets Manager function directly
        if USE_AWS_SECRETS:
            print(f"\n🔍 Testing AWS Secrets Manager function directly...")
            try:
                from app.core.config import get_secret_from_aws
                test_secret = get_secret_from_aws(OPENAI_SECRET_NAME, AWS_REGION)
                if test_secret:
                    print(f"    Direct AWS Secrets Manager call successful")
                else:
                    print(f"    AWS Secrets Manager call returned None")
            except Exception as e:
                print(f"    AWS Secrets Manager function failed: {e}")
        
        print(f"\n🎉 AWS Secrets Manager integration test completed!")
        return True
        
    except ImportError as e:
        print(f" Import error: {e}")
        print(f"   Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_file_loading():
    """Test .env file loading"""
    print(f"\n Testing .env file loading...")
    
    env = os.getenv("ENVIRONMENT", "development")
    env_file = f".env.{env}"
    
    if os.path.exists(env_file):
        print(f"    Environment file exists: {env_file}")
        
        # Read and display key lines
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        key_vars = ['USE_AWS_SECRETS', 'OPENAI_SECRET_NAME', 'AWS_REGION', 'AOSS_HOST']
        print(f"    Key environment variables:")
        for line in lines:
            line = line.strip()
            if any(var in line for var in key_vars) and not line.startswith('#'):
                print(f"     {line}")
    else:
        print(f"   Environment file not found: {env_file}")

if __name__ == "__main__":
    print(" EDGP AI Policy Suggest - AWS Secrets Manager Test")
    print("=" * 60)
    
    # Test environment file loading
    test_env_file_loading()
    
    # Test AWS Secrets Manager integration
    success = test_aws_secrets_integration()
    
    if success:
        print(f"\n All tests passed! Your AWS Secrets Manager integration is working.")
    else:
        print(f"\n Some tests failed. Please check the configuration.")
    
    sys.exit(0 if success else 1)