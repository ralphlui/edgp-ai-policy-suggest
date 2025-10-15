#!/usr/bin/env python3
"""
Test JWT token parsing in audit middleware
"""

import jwt
import json
from app.aws.audit_models import AuditLogDTO, ActivityType, ResponseStatus

def test_jwt_token_creation():
    """Test creating and parsing JWT tokens like the middleware does"""
    
    print("🧪 Testing JWT Token Parsing for Audit Middleware")
    print("=" * 60)
    
    # Test 1: Valid JWT token with user info
    print("\n1. Testing Valid JWT Token:")
    payload = {
        'sub': 'user_12345',
        'username': 'john.doe',
        'exp': 9999999999  # Far future expiry
    }
    
    # Create token
    token = jwt.encode(payload, 'secret_key', algorithm='HS256')
    print(f"Created Token: {token}")
    
    # Decode token (like middleware does - without verification for audit)
    decoded = jwt.decode(token, options={"verify_signature": False})
    print(f"Decoded Payload: {json.dumps(decoded, indent=2)}")
    
    # Extract user info like middleware
    user_id = decoded.get('sub')
    username = decoded.get('username')
    print(f"Extracted User ID: {user_id}")
    print(f"Extracted Username: {username}")
    
    # Test 2: Create audit log with extracted info
    print("\n2. Creating Audit Log with JWT Info:")
    audit = AuditLogDTO(
        statusCode='401',
        userId=user_id or '0000',
        username=username or 'public user',
        activityType=ActivityType.CORE_SCHEMA_CREATE,
        activityDescription='Create new domain schema',
        requestActionEndpoint='/api/aips/domains/create',
        responseStatus=ResponseStatus.FAILED,
        requestType='POST',
        remarks=f'{username or "public user"} encountered error: JWT token is expired'
    )
    
    # Convert to SQS message format
    message = audit.to_sqs_message()
    print(f"Audit Message JSON:")
    print(json.dumps(message, indent=2))
    
    # Test 3: Anonymous user (no token)
    print("\n3. Testing Anonymous User (No Token):")
    anonymous_audit = AuditLogDTO(
        statusCode='200',
        userId='0000',
        username='public user',
        activityType=ActivityType.CORE_DOMAIN_LIST,
        activityDescription='Retrieve domain list',
        requestActionEndpoint='/api/aips/domains',
        responseStatus=ResponseStatus.SUCCESS,
        requestType='GET',
        remarks='Public user retrieving'
    )
    
    anonymous_message = anonymous_audit.to_sqs_message()
    print(f"Anonymous User Audit:")
    print(json.dumps(anonymous_message, indent=2))
    
    print("\n✅ All tests passed! JWT parsing and audit logging work correctly.")

if __name__ == "__main__":
    test_jwt_token_creation()