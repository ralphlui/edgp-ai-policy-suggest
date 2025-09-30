"""
Pytest configuration file for setting up test environment
This ensures test environment is loaded before any test modules are imported
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup test environment BEFORE importing any app modules
def setup_early_test_env():
    """Setup test environment variables early"""
    PROJECT_ROOT = Path(__file__).parent.parent
    TEST_ENV_FILE = PROJECT_ROOT / ".env.test"
    
    # Set test environment variables FIRST
    os.environ["ENVIRONMENT"] = "test"
    os.environ["TESTING"] = "true"
    os.environ["USE_AWS_SECRETS"] = "false"
    os.environ["OPENAI_API_KEY"] = "mock-openai-api-key-sk-test123456789"
    os.environ["DISABLE_EXTERNAL_CALLS"] = "true"
    
    # Load test environment file
    if TEST_ENV_FILE.exists():
        load_dotenv(dotenv_path=TEST_ENV_FILE, override=True)
        
    # Add project root to path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

# Call setup immediately when this file is imported
setup_early_test_env()

# Simple setup function to replace removed test_config
def setup_test_environment():
    """Basic test environment setup"""
    pass

# Pytest configuration
def pytest_configure(config):
    """Called after command line options have been parsed"""
    setup_test_environment()

# Pytest fixtures
import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Auto-setup test session"""
    setup_test_environment()
    yield
    # Cleanup after all tests

@pytest.fixture
def mock_openai_api_key():
    """Provide mock OpenAI API key for tests"""
    return "mock-openai-api-key-sk-test123456789"

@pytest.fixture
def mock_auth_headers():
    """Provide mock authentication headers"""
    return {"Authorization": "Bearer mock-jwt-token"}

@pytest.fixture 
def test_domain():
    """Provide test domain name"""
    return "test_domain"

@pytest.fixture
def sample_schema():
    """Provide sample schema for testing"""
    return {
        "customer_name": {
            "type": "string",
            "sample_values": ["John Doe", "Jane Smith"]
        },
        "age": {
            "type": "integer", 
            "sample_values": [25, 30, 45]
        },
        "email": {
            "type": "string",
            "sample_values": ["john@example.com", "jane@example.com"]
        }
    }