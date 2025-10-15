#!/usr/bin/env python3
"""
Audit Logging Models
Data models for audit trail logging with SQS integration
Matches the specified JSON format exactly
"""

from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
from enum import Enum
import time
from datetime import datetime

class ActivityType(str, Enum):
    """Activity type enumeration for different operations"""
    CORE_CAMPAIGN_LIST = "Core-Campaign List"
    CORE_RULE_SUGGEST = "Core-Rule Suggest"
    CORE_SCHEMA_CREATE = "Core-Schema Create"
    CORE_SCHEMA_EXTEND = "Core-Schema Extend"
    CORE_DOMAIN_LIST = "Core-Domain List"
    CORE_DOMAIN_DETAILS = "Core-Domain Details"
    CORE_VECTOR_STATUS = "Core-Vector Status"
    AGENT_INSIGHTS = "Agent-Insights"
    VALIDATION_CHECK = "Validation-Check"
    HEALTH_CHECK = "Health-Check"
    INFO_REQUEST = "Info-Request"
    AUTHENTICATION_LOGIN = "Authentication-Login"
    AUTHENTICATION_LOGOUT = "Authentication-Logout"

class ResponseStatus(str, Enum):
    """Response status enumeration"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"

class AuditLogDTO(BaseModel):
    """
    Audit Log Data Transfer Object
    Matches the exact JSON format specified
    """
    statusCode: str = Field(..., description="HTTP status code")
    userId: str = Field(default="0000", description="User ID from token or default for public")
    username: str = Field(default="public user", description="Username from token or email")
    activityType: ActivityType = Field(..., description="Type of activity being performed")
    activityDescription: str = Field(..., description="Human-readable description of the activity")
    requestActionEndpoint: str = Field(..., description="API endpoint that was called")
    responseStatus: ResponseStatus = Field(..., description="Response status")
    requestType: str = Field(..., description="HTTP method (GET, POST, PUT, DELETE)")
    remarks: str = Field(default="", description="Additional remarks or error details")
    
    class Config:
        use_enum_values = True
    
    def to_sqs_message(self) -> Dict[str, Any]:
        """
        Convert to SQS message format
        """
        # Use model_dump() to leverage Pydantic's enum handling
        return self.model_dump()

class AuditContext(BaseModel):
    """Context information for audit logging"""
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    client_ip: str = "unknown"
    user_agent: Optional[str] = None
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Endpoint to activity mapping for automatic classification
endpoint_activity_mapping = {
    "POST /api/aips/rules/suggest": (ActivityType.CORE_RULE_SUGGEST, "Generate validation rules for domain"),
    "GET /api/aips/domains": (ActivityType.CORE_DOMAIN_LIST, "Retrieve domain list"),
    "GET /api/aips/domains": (ActivityType.CORE_DOMAIN_DETAILS, "Get domain details"),
    "POST /api/aips/domains/create": (ActivityType.CORE_SCHEMA_CREATE, "Create new domain schema"),
    "PUT /api/aips/domains/extend-schema": (ActivityType.CORE_SCHEMA_EXTEND, "Extend existing domain schema"),
    "POST /api/aips/domains/suggest-extend-schema": (ActivityType.CORE_RULE_SUGGEST, "Suggest additional columns for domain"),
    "POST /api/aips/domains/suggest-schema": (ActivityType.CORE_RULE_SUGGEST, "AI-powered domain schema suggestions"),
    "GET /api/aips/vector/status": (ActivityType.CORE_VECTOR_STATUS, "Check vector database connection and index status"),
    "POST /api/aips/validation": (ActivityType.VALIDATION_CHECK, "Validate LLM responses"),
    "GET /api/aips/agent": (ActivityType.AGENT_INSIGHTS, "Agent insights and monitoring"),
    "GET /api/aips/health": (ActivityType.HEALTH_CHECK, "System health check"),
    "GET /api/aips/info": (ActivityType.INFO_REQUEST, "System information request"),
}

def get_activity_info(endpoint: str) -> tuple[ActivityType, str]:
    """Get activity type and description for an endpoint"""
    for pattern, (activity_type, description) in endpoint_activity_mapping.items():
        if endpoint.startswith(pattern):
            return activity_type, description
    
    # Default for unknown endpoints
    return ActivityType.INFO_REQUEST, f"API request to {endpoint}"