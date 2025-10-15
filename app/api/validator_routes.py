"""
Validation API Routes

Provides API endpoints for monitoring and managing LLM validation
performance, metrics, and configuration.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import Response

from app.validation.metrics import (
    get_validation_summary, 
    get_metrics_collector,
    ValidationMonitor
)
from app.validation.config import load_validation_config
from app.validation.llm_validator import LLMResponseValidator

logger = logging.getLogger(__name__)

# Create router for validation endpoints
validation_router = APIRouter(tags=["validation"])  # Prefix will be added in main.py

# Global validation monitor instance
_validation_monitor = ValidationMonitor()


@validation_router.get("/health")
async def get_validation_health():
    """Get overall validation system health status"""
    try:
        # Check if validation system is functioning
        config = load_validation_config()
        validator = LLMResponseValidator(config)
        
        # Basic health check
        test_response = {"test": "data"}
        validation_result = validator.validate_response(test_response, "test")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "config_loaded": True,
            "validator_functional": True,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Validation health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation system unhealthy: {str(e)}")


@validation_router.get("/metrics/summary")
async def get_metrics_summary(
    hours: int = Query(24, description="Hours to look back for metrics", ge=1, le=168)
) -> Dict[str, Any]:
    """
    Get validation metrics summary for the specified time period
    
    Args:
        hours: Number of hours to look back (1-168)
        
    Returns:
        Validation metrics summary including success rates, confidence scores, and issue distribution
    """
    try:
        summary = get_validation_summary(hours)
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@validation_router.get("/metrics/domain/{domain}")
async def get_domain_metrics(
    domain: str,
    hours: int = Query(24, description="Hours to look back for metrics", ge=1, le=168)
) -> Dict[str, Any]:
    """
    Get validation metrics for a specific domain
    
    Args:
        domain: Domain name to get metrics for
        hours: Number of hours to look back
        
    Returns:
        Domain-specific validation metrics
    """
    try:
        collector = get_metrics_collector()
        domain_performance = collector.get_domain_performance(domain, hours)
        
        return {
            "status": "success",
            "data": domain_performance,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get domain metrics for {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve domain metrics: {str(e)}")


@validation_router.get("/performance/check")
async def check_performance(
    hours: int = Query(1, description="Hours to check performance for", ge=1, le=24)
) -> Dict[str, Any]:
    """
    Check validation performance against configured thresholds
    
    Args:
        hours: Number of hours to analyze
        
    Returns:
        Performance check results with any issues found
    """
    try:
        performance_check = _validation_monitor.check_performance(hours)
        
        return {
            "status": "success",
            "data": performance_check,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Performance check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance check failed: {str(e)}")


@validation_router.get("/config")
async def get_validation_config() -> Dict[str, Any]:
    """
    Get current validation configuration
    
    Returns:
        Current validation configuration settings
    """
    try:
        config = load_validation_config()
        
        # Convert to dict for JSON serialization
        config_dict = {
            "profile": config.profile.value,
            "max_issues_allowed": config.max_issues_allowed,
            "min_confidence_score": config.min_confidence_score,
            "enable_auto_correction": config.enable_auto_correction,
            "schema_validation_enabled": config.schema_validation_enabled,
            "rule_validation_enabled": config.rule_validation_enabled,
            "content_validation_enabled": config.content_validation_enabled,
            "domain_rules_count": len(config.domain_rules)
        }
        
        return {
            "status": "success",
            "data": config_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get validation config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve config: {str(e)}")


@validation_router.get("/metrics/export")
async def export_metrics(
    hours: int = Query(24, description="Hours of data to export", ge=1, le=168),
    format: str = Query("json", description="Export format", regex="^(json|csv)$")
) -> Response:
    """
    Export validation metrics data
    
    Args:
        hours: Number of hours of data to export
        format: Export format (json or csv)
        
    Returns:
        Exported metrics data
    """
    try:
        collector = get_metrics_collector()
        exported_data = collector.export_metrics(hours, format)
        
        if format.lower() == "json":
            media_type = "application/json"
            filename = f"validation_metrics_{hours}h.json"
        else:  # csv
            media_type = "text/csv"
            filename = f"validation_metrics_{hours}h.csv"
        
        return Response(
            content=exported_data,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@validation_router.post("/test")
async def test_validation(
    response_data: Dict[str, Any],
    response_type: str = Query("schema", description="Type of response to validate"),
    strict_mode: bool = Query(True, description="Use strict validation mode")
) -> Dict[str, Any]:
    """
    Test validation system with provided data
    
    Args:
        response_data: Data to validate
        response_type: Type of response (schema, rule, content)
        strict_mode: Whether to use strict validation
        
    Returns:
        Validation results
    """
    try:
        config = load_validation_config()
        validator = LLMResponseValidator(config)
        
        validation_result = validator.validate_response(
            response=response_data,
            response_type=response_type,
            strict_mode=strict_mode,
            auto_correct=True
        )
        
        # Convert validation result to dict for JSON response
        result_dict = {
            "is_valid": validation_result.is_valid,
            "confidence_score": validation_result.confidence_score,
            "issues": [
                {
                    "field": issue.field,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "suggestion": issue.suggestion
                }
                for issue in validation_result.issues
            ],
            "corrected_data": validation_result.corrected_data,
            "metadata": validation_result.metadata
        }
        
        return {
            "status": "success",
            "data": result_dict,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Validation test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation test failed: {str(e)}")


@validation_router.get("/stats")
async def get_validation_stats() -> Dict[str, Any]:
    """
    Get comprehensive validation system statistics
    
    Returns:
        Comprehensive validation statistics
    """
    try:
        # Get metrics for different time periods
        hourly_summary = get_validation_summary(1)
        daily_summary = get_validation_summary(24)
        weekly_summary = get_validation_summary(168)
        
        # Get current config
        config = load_validation_config()
        
        # Performance check
        performance_check = _validation_monitor.check_performance(1)
        
        stats = {
            "config_profile": config.profile.value,
            "hourly_metrics": {
                "success_rate": hourly_summary["success_rate"],
                "avg_confidence": hourly_summary["avg_confidence"],
                "total_validations": hourly_summary["total_validations"]
            },
            "daily_metrics": {
                "success_rate": daily_summary["success_rate"],
                "avg_confidence": daily_summary["avg_confidence"],
                "total_validations": daily_summary["total_validations"]
            },
            "weekly_metrics": {
                "success_rate": weekly_summary["success_rate"],
                "avg_confidence": weekly_summary["avg_confidence"],
                "total_validations": weekly_summary["total_validations"]
            },
            "performance_health": performance_check["overall_health"],
            "active_domains": list(daily_summary.get("domain_stats", {}).keys()),
            "response_types": list(daily_summary.get("response_type_stats", {}).keys())
        }
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get validation stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


# Include router in main app
def setup_validation_routes(app):
    """Setup validation routes in FastAPI app"""
    app.include_router(validation_router)
    logger.info("Validation API routes configured")


if __name__ == "__main__":
    # For testing
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Validation API Test")
    setup_validation_routes(app)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)