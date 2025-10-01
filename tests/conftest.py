"""
Pytest configuration and fixtures for EDGP AI Policy Suggest API tests.
Ensures all tests run with the proper test environment configuration.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to Python path (go up one level from tests/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def pytest_configure(config):
    """
    Configure pytest to use test environment.
    This runs before any tests are collected or executed.
    """
    # Set test environment variables
    os.environ["APP_ENV"] = "test"
    os.environ["ENVIRONMENT"] = "test"
    os.environ["TESTING"] = "true"
    os.environ["USE_AWS_SECRETS"] = "false"
    os.environ["OPENAI_API_KEY"] = "mock-openai-api-key-sk-test123456789"
    
    print("🧪 Test environment configured")

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Session-scoped fixture that ensures test environment is properly configured.
    """
    # Ensure critical test environment variables are set
    required_vars = {
        "APP_ENV": "test",
        "USE_AWS_SECRETS": "false",
        "TESTING": "true"
    }
    
    for key, value in required_vars.items():
        os.environ[key] = value
    
    yield
    
    # Cleanup after tests

@pytest.fixture
def test_client():
    """
    Fixture to provide a test client for the FastAPI application.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    
    with TestClient(app) as client:
        yield client