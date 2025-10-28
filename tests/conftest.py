"""
Pytest configuration and fixtures for EDGP AI Policy Suggest API tests.
Ensures all tests run with the proper test environment configuration.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add project root to PYTHONPATH
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.auth.authentication import UserInfo
from app.api.domain_schema_routes import router

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
    os.environ["TEST_MODE"] = "true"
    os.environ["JWT_TEST_SECRET"] = "test-jwt-secret-key-123"
    
    print(" Test environment configured")

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Session-scoped fixture that ensures test environment is properly configured.
    """
    # Ensure critical test environment variables are set
    required_vars = {
        "APP_ENV": "test",
        "USE_AWS_SECRETS": "false",
        "TESTING": "true",
        "TEST_MODE": "true",
        "JWT_TEST_SECRET": "test-jwt-secret-key-123"
    }
    
    for key, value in required_vars.items():
        os.environ[key] = value
    
    yield
    
    # Cleanup after tests

@pytest.fixture
def test_app():
    """Create a test FastAPI app with our router"""
    app = FastAPI()
    app.include_router(router)
    return app

@pytest.fixture
def test_client(test_app, mock_user, override_auth_dependency):
    """
    Fixture to provide a test client for the FastAPI application.
    """
    return TestClient(test_app)

@pytest.fixture
def mock_store():
    """Create a mock OpenSearchColumnStore"""
    store = Mock()
    store.index_name = "test-index"
    return store

@pytest.fixture
def mock_user():
    """Create a mock authenticated user"""
    return UserInfo(
        email="test@example.com",
        user_id="test123",
        scopes=["manage:policy"],
        token_payload={
            "userEmail": "test@example.com",
            "sub": "test123",
            "scope": "manage:policy",
            "exp": 1735689600,  # 2025-01-01
            "iat": 1672531200   # 2023-01-01
        }
    )

@pytest.fixture
def override_auth_dependency(test_app, mock_user):
    """Override the authentication dependency for testing"""
    from app.auth.authentication import verify_any_scope_token
    
    async def mock_auth():
        return mock_user
    
    test_app.dependency_overrides[verify_any_scope_token] = mock_auth
    yield
    test_app.dependency_overrides = {}