"""
Validation API Routes

Provides API endpoints for monitoring and managing LLM validation
performance, metrics, and configuration.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends, Body
from fastapi.responses import Response, JSONResponse
from app.auth.authentication import verify_any_scope_token, UserInfo

from app.validation.compat import (
    LLMResponseValidator,
    ValidationConfig,
    load_validation_config
)

from app.validation.metrics import (
    ValidationMetricsCollector,
    get_validation_summary,
    get_metrics_collector,
    ValidationMonitor
)

logger = logging.getLogger(__name__)

# Create router for validation endpoints
validation_router = APIRouter(prefix="/validation", tags=["validation"])

# Global validation monitor instance
_validation_monitor = ValidationMonitor(
    success_rate_threshold=0.8,
    confidence_threshold=0.7
)

@validation_router.get("/health")
async def get_validation_health():
    """Get overall validation system health status"""
    try:
        # Check if validation system is functioning
        config = load_validation_config()
        validator = LLMResponseValidator(config)
        
        # Basic health check
        test_response = {"test": "data"}
        validation_result = validator.validate_response(test_response, "schema")
        
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


@validation_router.get("/metrics")
async def get_validation_metrics():
    """Get LLM validation metrics and statistics"""
    try:
        collector = get_metrics_collector()
        metrics = collector.get_metrics_summary()
        
        return JSONResponse({
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting validation metrics: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Failed to get validation metrics: {str(e)}"
        }, status_code=500)


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
            "profile": getattr(config, "profile", "standard"),
            "max_issues_allowed": getattr(config, "max_issues_allowed", 5),
            "min_confidence_score": getattr(config, "min_confidence_score", 0.7),
            "enable_auto_correction": getattr(config, "enable_auto_correction", False),
            "schema_validation_enabled": getattr(config, "schema_validation_enabled", True),
            "rule_validation_enabled": getattr(config, "rule_validation_enabled", True),
            "content_validation_enabled": getattr(config, "content_validation_enabled", True)
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


@validation_router.post("/validate-schema")
async def validate_schema_endpoint(
    payload: Dict[str, Any] = Body(...),
    user: UserInfo = Depends(verify_any_scope_token)
):
    """Validate an LLM-generated schema response"""
    try:
        logger.info(f"Schema validation request from user: {user.email}")
        
        if "schema" not in payload:
            return JSONResponse({
                "status": "error",
                "message": "Missing required field: 'schema'"
            }, status_code=400)
        
        schema = payload["schema"]
        context = payload.get("context", {})
        
        validator = LLMResponseValidator()
        result = validator.validate_response(schema, "schema")
        
        return JSONResponse({
            "status": "success",
            "validation_result": {
                "is_valid": result.is_valid,
                "confidence_score": result.confidence_score,
                "validation_errors": [vars(issue) for issue in result.issues],
                "suggestions": [issue.suggestion for issue in result.issues if issue.suggestion],
                "metadata": result.metadata
            }
        })
        
    except Exception as e:
        logger.error(f"Error validating schema: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Schema validation failed: {str(e)}"
        }, status_code=500)


@validation_router.post("/validate-rules")
async def validate_rules_endpoint(
    payload: Dict[str, Any] = Body(...),
    user: UserInfo = Depends(verify_any_scope_token)
):
    """Validate LLM-generated rules"""
    try:
        logger.info(f"Rules validation request from user: {user.email}")
        
        if "rules" not in payload:
            return JSONResponse({
                "status": "error",
                "message": "Missing required field: 'rules'"
            }, status_code=400)
        
        rules = payload["rules"]
        context = payload.get("context", {})
        
        validator = LLMResponseValidator()
        result = validator.validate_response(rules, "rule")
        
        return JSONResponse({
            "status": "success",
            "validation_result": {
                "is_valid": result.is_valid,
                "confidence_score": result.confidence_score,
                "validation_errors": [vars(issue) for issue in result.issues],
                "suggestions": [issue.suggestion for issue in result.issues if issue.suggestion],
                "metadata": result.metadata
            }
        })
        
    except Exception as e:
        logger.error(f"Error validating rules: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Rules validation failed: {str(e)}"
        }, status_code=500)


@validation_router.post("/test")
async def test_validation_endpoint(
    payload: Dict[str, Any] = Body(...),
    response_type: str = Query("schema", description="Type of response to validate"),
    strict_mode: bool = Query(True, description="Use strict validation mode"),
    user: UserInfo = Depends(verify_any_scope_token)
):
    """Test endpoint for validation system"""
    try:
        logger.info(f"Validation test request from user: {user.email}")
        
        validator = LLMResponseValidator()
        test_data = payload.get("test_data", {})
        
        result = validator.validate_response(
            response=test_data,
            response_type=response_type,
            strict_mode=strict_mode,
            auto_correct=True
        )
        
        return JSONResponse({
            "status": "success",
            "test_type": response_type,
            "validation_result": {
                "is_valid": result.is_valid,
                "confidence_score": result.confidence_score,
                "validation_errors": [vars(issue) for issue in result.issues],
                "suggestions": [issue.suggestion for issue in result.issues if issue.suggestion],
                "execution_time": result.metadata.get("execution_time", 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Validation test failed: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Validation test failed: {str(e)}"
        }, status_code=500)


# Include router in main app
def setup_validation_routes(app):
    """Setup validation routes in FastAPI app"""
    app.include_router(validation_router)
    logger.info("Validation API routes configured")