#!/usr/bin/env python3
"""
JWT Token Test Utility
Test JWT token extraction for audit logging
"""

import jwt
import json
from datetime import datetime, timedelta

def create_test_jwt_token(user_id: str, username: str, secret: str = "test_secret") -> str:
    """Create a test JWT token for testing"""
    
    payload = {
        "sub": user_id,  # Standard 'subject' claim - maps to user_id
        "username": username,  # Username claim
        "iat": datetime.utcnow(),  # Issued at
        "exp": datetime.utcnow() + timedelta(hours=1),  # Expires in 1 hour
        "iss": "edgp-ai-policy-suggest",  # Issuer
        "aud": "api-users"  # Audience
    }
    
    # Create JWT token
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token

def decode_test_token(token: str, secret: str = "test_secret", verify: bool = False) -> dict:
    """Decode JWT token for testing"""
    
    try:
        if verify:
            # Decode with verification
            payload = jwt.decode(token, secret, algorithms=["HS256"])
        else:
            # Decode without verification (for audit purposes)
            payload = jwt.decode(token, options={"verify_signature": False})
        
        return payload
    except jwt.ExpiredSignatureError:
        return {"error": "Token has expired"}
    except jwt.InvalidTokenError:
        return {"error": "Invalid token"}
    except Exception as e:
        return {"error": str(e)}

def test_jwt_extraction():
    """Test JWT token creation and extraction"""
    
    print(" JWT TOKEN EXTRACTION TEST")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "user_id": "user_12345",
            "username": "john.doe",
            "description": "Standard user with alphanumeric ID"
        },
        {
            "user_id": "admin_999",
            "username": "admin@company.com",
            "description": "Admin user with email as username"
        },
        {
            "user_id": "service_account_001",
            "username": "api_service",
            "description": "Service account"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print("-" * 30)
        
        user_id = test_case["user_id"]
        username = test_case["username"]
        
        # Create JWT token
        token = create_test_jwt_token(user_id, username)
        print(f"Generated JWT Token:")
        print(f"Bearer {token}")
        
        # Decode without verification (audit method)
        payload = decode_test_token(token, verify=False)
        
        if "error" in payload:
            print(f" Error: {payload['error']}")
            continue
        
        # Extract user info
        extracted_user_id = payload.get("sub")
        extracted_username = payload.get("username")
        
        print(f"\n Extracted Information:")
        print(f"  User ID (sub): {extracted_user_id}")
        print(f"  Username: {extracted_username}")
        print(f"  Issued At: {payload.get('iat')}")
        print(f"  Expires: {payload.get('exp')}")
        print(f"  Issuer: {payload.get('iss')}")
        
        # Validation
        if extracted_user_id == user_id and extracted_username == username:
            print(f" Extraction successful!")
        else:
            print(f"   Extraction failed!")
            print(f"   Expected: user_id={user_id}, username={username}")
            print(f"   Got: user_id={extracted_user_id}, username={extracted_username}")

def show_curl_examples():
    """Show curl examples with JWT tokens"""
    
    print("\n CURL EXAMPLES WITH JWT TOKENS")
    print("=" * 50)
    
    # Generate sample tokens
    tokens = [
        create_test_jwt_token("user_123", "john.doe"),
        create_test_jwt_token("admin_001", "admin@company.com"),
        create_test_jwt_token("service_001", "api_service")
    ]
    
    endpoints = [
        "POST /api/aips/rules/suggest",
        "GET /api/aips/domains",
        "POST /api/aips/domains/create"
    ]
    
    for i, (endpoint, token) in enumerate(zip(endpoints, tokens), 1):
        method, path = endpoint.split(" ", 1)
        
        print(f"\nExample {i}: {endpoint}")
        print("-" * 30)
        print(f"curl -X {method} \\")
        print(f"  -H \"Authorization: Bearer {token}\" \\")
        print(f"  -H \"Content-Type: application/json\" \\")
        
        if method == "POST":
            print(f"  -d '{{\"domain_name\":\"test\",\"column_names\":[\"email\"]}}' \\")
        
        print(f"  http://localhost:8000{path}")
        
        # Show what will be extracted
        payload = decode_test_token(token, verify=False)
        print(f"\n  → Will extract:")
        print(f"    User ID: {payload.get('sub')}")
        print(f"    Username: {payload.get('username')}")

def show_audit_log_example():
    """Show example audit log with real JWT data"""
    
    print("\n AUDIT LOG EXAMPLE WITH JWT DATA")
    print("=" * 50)
    
    # Create sample token
    token = create_test_jwt_token("user_12345", "john.doe")
    payload = decode_test_token(token, verify=False)
    
    # Simulate audit log
    audit_log = {
        "userId": payload.get("sub"),
        "userName": payload.get("username"),
        "activityType": "POLICY_SUGGESTION",
        "endPoint": "POST /api/aips/rules/suggest",
        "requestId": "req-abc-123",
        "clientIp": "192.168.1.100",
        "remarks": json.dumps({
            "transaction_type": "API_REQUEST",
            "endpoint": "POST /api/aips/rules/suggest",
            "status_code": 200,
            "processing_time_ms": 1247.83,
            "success": True,
            "jwt_claims": {
                "sub": payload.get("sub"),
                "username": payload.get("username"),
                "iss": payload.get("iss"),
                "aud": payload.get("aud")
            }
        }, indent=2),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    print("JSON sent to SQS:")
    print(json.dumps(audit_log, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_jwt_extraction()
    show_curl_examples()
    show_audit_log_example()
    
    print("\n" + "=" * 50)
    print(" IMPLEMENTATION SUMMARY")
    print("=" * 50)
    print(" JWT tokens are decoded without signature verification for audit")
    print(" User ID extracted from 'sub' claim (JWT standard)")
    print(" Username extracted from 'username' claim")
    print(" Fallback to alternative claims if standard ones not found")
    print(" Graceful handling of invalid/missing tokens")
    print(" Debug logging for troubleshooting")
    print("\n Ready for production use!")