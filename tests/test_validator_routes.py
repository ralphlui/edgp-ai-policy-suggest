"""
Tests for validator_routes.py endpoints
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock
from app.api.validator_routes import setup_validation_routes
from app.validation.config import ValidationProfile, ValidationConfig
from app.validation.llm_validator import ValidationResult, ValidationSeverity, ValidationIssue


@pytest.fixture
def test_app():
    app = FastAPI()
    setup_validation_routes(app)
    return TestClient(app)


@pytest.fixture
def mock_config():
    return ValidationConfig(
        strict_mode=True,
        auto_correct=True,
        min_confidence_score=0.7,
        min_columns=3,
        max_columns=15,
        min_samples_per_column=3,
        check_pii=True,
        check_sensitive_keywords=True,
        validation_timeout=30.0
    )


@pytest.fixture
def mock_validation_result():
    return ValidationResult(
        is_valid=True,
        issues=[
            ValidationIssue(
                field="test_field",
                severity=ValidationSeverity.LOW,
                message="Test message",
                suggested_fix="Test suggestion"
            )
        ],
        confidence_score=0.9,
        corrected_data={"test": "data"}
    )


def test_health_check(test_app, mock_config):
    """Test the health check endpoint"""
    with patch('app.api.validator_routes.load_validation_config', return_value=mock_config):
        with patch('app.api.validator_routes.LLMResponseValidator') as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator_class.return_value = mock_validator
            mock_validator.validate_response.return_value = None
            
            response = test_app.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert data["config_loaded"] is True
            assert data["validator_functional"] is True


def test_health_check_failure(test_app):
    """Test the health check endpoint when validation system is unhealthy"""
    with patch('app.api.validator_routes.load_validation_config', side_effect=Exception("Test error")):
        response = test_app.get("/health")
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]


def test_metrics_summary(test_app):
    """Test getting metrics summary"""
    mock_summary = {
        "total_validations": 100,
        "success_rate": 0.85,
        "avg_confidence": 0.9,
        "issue_distribution": {"critical": 0, "high": 5, "medium": 10, "low": 20}
    }
    
    with patch('app.api.validator_routes.get_validation_summary', return_value=mock_summary):
        response = test_app.get("/metrics/summary?hours=24")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] == mock_summary
        assert "timestamp" in data


def test_domain_metrics(test_app):
    """Test getting domain-specific metrics"""
    mock_domain_performance = {
        "domain": "test_domain",
        "total_validations": 50,
        "success_rate": 0.9,
        "avg_confidence": 0.85,
        "trend": "improving"
    }
    
    with patch('app.api.validator_routes.get_metrics_collector') as mock_collector_fn:
        mock_collector = MagicMock()
        mock_collector.get_domain_performance.return_value = mock_domain_performance
        mock_collector_fn.return_value = mock_collector
        
        response = test_app.get("/metrics/domain/test_domain?hours=24")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] == mock_domain_performance


def test_performance_check(test_app):
    """Test performance check endpoint"""
    mock_performance = {
        "overall_health": "good",
        "issues_found": 0,
        "issues": []
    }
    
    with patch('app.api.validator_routes._validation_monitor.check_performance', return_value=mock_performance):
        response = test_app.get("/performance/check?hours=1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"] == mock_performance


def test_get_validation_config(test_app, mock_config):
    """Test getting validation configuration"""
    with patch('app.api.validator_routes.load_validation_config', return_value=mock_config):
        response = test_app.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        
        config_data = data["data"]
        assert config_data["min_confidence_score"] == 0.7
        assert isinstance(config_data["strict_mode"], bool)
        assert isinstance(config_data["auto_correct"], bool)
        assert isinstance(config_data["check_pii"], bool)
        assert isinstance(config_data["check_sensitive_keywords"], bool)
        assert "validation_timeout" in config_data


@pytest.mark.parametrize("format", ["json", "csv"])
def test_export_metrics(test_app, format):
    """Test exporting metrics in different formats"""
    mock_exported_data = "test data"
    
    with patch('app.api.validator_routes.get_metrics_collector') as mock_collector_fn:
        mock_collector = MagicMock()
        mock_collector.export_metrics.return_value = mock_exported_data
        mock_collector_fn.return_value = mock_collector
        
        response = test_app.get(f"/metrics/export?hours=24&format={format}")
        assert response.status_code == 200
        assert "Content-Disposition" in response.headers
        assert f"validation_metrics_24h.{format}" in response.headers["Content-Disposition"]
        assert response.text == mock_exported_data


def test_test_validation_endpoint(test_app, mock_config, mock_validation_result):
    """Test the validation test endpoint"""
    test_data = {"test": "data"}
    
    with patch('app.api.validator_routes.load_validation_config', return_value=mock_config):
        with patch('app.api.validator_routes.LLMResponseValidator') as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator_class.return_value = mock_validator
            mock_validator.validate_response.return_value = mock_validation_result
            
            response = test_app.post(
                "/test?response_type=schema&strict_mode=true",
                json=test_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            
            validation_data = data["data"]
            assert validation_data["is_valid"] is True
            assert validation_data["confidence_score"] == 0.9
            assert len(validation_data["issues"]) == 1
            
            issue = validation_data["issues"][0]
            assert issue["field"] == "test_field"
            assert issue["severity"] == "low"
            assert issue["message"] == "Test message"
            assert issue["suggestion"] == "Test suggestion"
            
            assert validation_data["corrected_data"] == {"test": "data"}


def test_get_validation_stats(test_app, mock_config):
    """Test getting validation statistics"""
    mock_summaries = {
        1: {"success_rate": 0.9, "avg_confidence": 0.85, "total_validations": 10},
        24: {"success_rate": 0.88, "avg_confidence": 0.86, "total_validations": 100},
        168: {"success_rate": 0.87, "avg_confidence": 0.84, "total_validations": 500}
    }
    
    mock_performance = {"overall_health": "good"}
    
    with patch('app.api.validator_routes.get_validation_summary', 
              side_effect=lambda h: mock_summaries.get(h, {})):
        with patch('app.api.validator_routes.load_validation_config', 
                  return_value=mock_config):
            with patch('app.api.validator_routes._validation_monitor.check_performance', 
                      return_value=mock_performance):
                
                response = test_app.get("/stats")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "hourly_metrics" in data["data"]
                assert "daily_metrics" in data["data"]
                assert "weekly_metrics" in data["data"]
                assert data["data"]["performance_health"] == "good"


def test_invalid_hours_parameter(test_app):
    """Test validation of hours parameter"""
    # Test with hours > 168 (maximum allowed)
    response = test_app.get("/metrics/summary?hours=169")
    assert response.status_code == 422
    
    # Test with hours < 1 (minimum allowed)
    response = test_app.get("/metrics/summary?hours=0")
    assert response.status_code == 422


def test_invalid_format_parameter(test_app):
    """Test validation of format parameter"""
    response = test_app.get("/metrics/export?hours=24&format=xml")
    assert response.status_code == 422


def test_exception_handling(test_app):
    """Test exception handling in various endpoints"""
    with patch('app.api.validator_routes.get_validation_summary', 
              side_effect=Exception("Test error")):
        response = test_app.get("/metrics/summary?hours=24")
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]