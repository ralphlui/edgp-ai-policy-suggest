"""
Test configuration helper to ensure tests use .env.test environment
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
TEST_ENV_FILE = PROJECT_ROOT / ".env.test"

def setup_test_environment():
    """
    Load test environment variables from .env.test
    This should be called at the beginning of all test modules
    """
    # Set test environment FIRST
    os.environ["ENVIRONMENT"] = "test"
    os.environ["TESTING"] = "true"
    
    # Load test environment file EARLY
    if TEST_ENV_FILE.exists():
        load_dotenv(dotenv_path=TEST_ENV_FILE, override=True)
        print(f" Loaded test environment from {TEST_ENV_FILE}")
    else:
        print(f" Test environment file not found: {TEST_ENV_FILE}")
    
    # Ensure critical test variables are set
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "mock-openai-api-key-sk-test123456789"
    
    if not os.getenv("USE_AWS_SECRETS"):
        os.environ["USE_AWS_SECRETS"] = "false"
    
    # Add project root to Python path for imports
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # Set test-specific configurations
    os.environ["DISABLE_EXTERNAL_CALLS"] = "true"
    os.environ["MOCK_AUTH_SERVICE"] = "true" 
    os.environ["MOCK_OPENSEARCH"] = "true"
    os.environ["MOCK_AWS_SECRETS"] = "true"
    
    return True

def get_test_config():
    """Get test-specific configuration values"""
    return {
        "host": os.getenv("HOST", "localhost"),
        "port": int(os.getenv("PORT", 8092)),
        "environment": os.getenv("ENVIRONMENT", "test"),
        "api_title": os.getenv("API_TITLE", "Test API"),
        "log_level": os.getenv("LOG_LEVEL", "debug"),
        "disable_external_calls": os.getenv("DISABLE_EXTERNAL_CALLS", "true").lower() == "true",
        "mock_auth_service": os.getenv("MOCK_AUTH_SERVICE", "true").lower() == "true",
        "mock_opensearch": os.getenv("MOCK_OPENSEARCH", "true").lower() == "true",
        "mock_aws_secrets": os.getenv("MOCK_AWS_SECRETS", "true").lower() == "true",
    }

# Auto-setup when imported
if __name__ != "__main__":
    setup_test_environment()