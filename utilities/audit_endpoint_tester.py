#!/usr/bin/env python3
"""
Dynamic API Endpoint Audit Test
Tests that all API endpoints are captured dynamically in audit logs
"""

import asyncio
import json
import httpx
from datetime import datetime
from typing import List, Dict

# Test endpoints to verify dynamic capture
TEST_ENDPOINTS = [
    {
        "method": "GET",
        "path": "/api/aips/domains",
        "description": "Get all domains"
    },
    {
        "method": "POST", 
        "path": "/api/aips/rules/suggest",
        "description": "Suggest validation rules",
        "body": {
            "domain_name": "test_domain",
            "column_names": ["email", "phone"],
            "use_agent_insights": True
        }
    },
    {
        "method": "POST",
        "path": "/api/aips/domains/create", 
        "description": "Create new domain",
        "body": {
            "domain_name": "audit_test_domain",
            "columns": [
                {"name": "user_id", "data_type": "string"},
                {"name": "email", "data_type": "string"}
            ]
        }
    },
    {
        "method": "GET",
        "path": "/api/aips/domains/audit_test_domain",
        "description": "Get specific domain details"
    },
    {
        "method": "PUT",
        "path": "/api/aips/domains/extend-schema",
        "description": "Extend domain schema",
        "body": {
            "domain_name": "audit_test_domain",
            "new_columns": [
                {"name": "phone", "data_type": "string"}
            ]
        }
    },
    {
        "method": "POST",
        "path": "/api/aips/domains/suggest-extend-schema/audit_test_domain",
        "description": "Suggest schema extension",
        "body": {
            "domain_description": "Customer contact information"
        }
    },
    {
        "method": "GET",
        "path": "/api/aips/vector/status",
        "description": "Check vector database status"
    },
    {
        "method": "POST",
        "path": "/api/aips/domains/suggest-schema",
        "description": "AI-powered schema suggestion",
        "body": {
            "domain_name": "new_test_domain",
            "domain_description": "User profile management",
            "business_context": "Customer relationship management"
        }
    }
]

class AuditTestClient:
    """Test client for verifying dynamic audit logging"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
        
    async def test_dynamic_endpoint_capture(self):
        """Test that all endpoints are captured dynamically"""
        print("🔍 Testing Dynamic Endpoint Capture for Audit Logging")
        print("=" * 60)
        
        for i, endpoint in enumerate(TEST_ENDPOINTS, 1):
            await self._test_single_endpoint(i, endpoint)
            await asyncio.sleep(0.5)  # Small delay between requests
        
        await self.client.aclose()
        
        # Print summary
        self._print_test_summary()
    
    async def _test_single_endpoint(self, test_num: int, endpoint: Dict):
        """Test a single endpoint for audit capture"""
        method = endpoint["method"]
        path = endpoint["path"]
        description = endpoint["description"]
        body = endpoint.get("body")
        
        print(f"Test {test_num}: {method} {path}")
        print(f"   Description: {description}")
        
        try:
            # Add query parameters for some endpoints to test dynamic capture
            query_params = {}
            if "suggest" in path.lower():
                query_params["test_mode"] = "audit_verification"
            if "domain" in path.lower():
                query_params["include_metadata"] = "true"
            
            # Construct full URL
            url = f"{self.base_url}{path}"
            
            # Add test headers
            headers = {
                "User-Agent": "AuditTestClient/1.0",
                "X-Test-ID": f"audit-test-{test_num}",
                "X-Client-IP": "192.168.1.100"
            }
            
            # Add auth header for authenticated endpoints
            if "/rule/" in path or "/domain/" in path:
                headers["Authorization"] = "Bearer test-audit-token-123"
            
            # Make request
            start_time = datetime.now()
            
            if method == "GET":
                response = await self.client.get(url, headers=headers, params=query_params)
            elif method == "POST":
                response = await self.client.post(url, headers=headers, params=query_params, json=body)
            elif method == "PUT":
                response = await self.client.put(url, headers=headers, params=query_params, json=body)
            else:
                response = await self.client.request(method, url, headers=headers, params=query_params, json=body)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            # Record test result
            test_result = {
                "test_number": test_num,
                "method": method,
                "path": path,
                "full_url": str(response.url),
                "status_code": response.status_code,
                "processing_time_ms": round(processing_time, 2),
                "success": response.status_code < 500,  # Accept 4xx as valid responses
                "description": description,
                "has_query_params": bool(query_params),
                "has_request_body": body is not None,
                "response_size": len(response.content) if response.content else 0
            }
            
            self.test_results.append(test_result)
            
            # Print result
            status_icon = "✅" if test_result["success"] else "❌"
            print(f"   {status_icon} Status: {response.status_code} | Time: {processing_time:.2f}ms")
            
            if query_params:
                print(f"   📝 Query params: {query_params}")
            
            if body:
                print(f"   📋 Request body: {len(json.dumps(body))} bytes")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            test_result = {
                "test_number": test_num,
                "method": method,
                "path": path,
                "status_code": 0,
                "success": False,
                "error": str(e),
                "description": description
            }
            self.test_results.append(test_result)
        
        print()
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        print("📊 AUDIT CAPTURE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - successful_tests
        
        print(f"Total Endpoints Tested: {total_tests}")
        print(f"Successful Requests: {successful_tests}")
        print(f"Failed Requests: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests*100):.1f}%")
        print()
        
        # Detailed results
        print("📋 DETAILED RESULTS:")
        print("-" * 60)
        
        for result in self.test_results:
            status_icon = "✅" if result["success"] else "❌"
            method = result["method"]
            path = result["path"]
            status = result.get("status_code", "ERR")
            
            print(f"{status_icon} {method:4} {path:35} | Status: {status}")
            
            if "full_url" in result:
                print(f"     Full URL: {result['full_url']}")
            
            if "processing_time_ms" in result:
                print(f"     Processing Time: {result['processing_time_ms']}ms")
            
            if "error" in result:
                print(f"     Error: {result['error']}")
            
            print()
        
        print("🎯 AUDIT VERIFICATION NOTES:")
        print("-" * 60)
        print("✓ All above requests should appear in audit logs")
        print("✓ Each endpoint should be captured with full URL including query params")
        print("✓ Request bodies should be logged for POST/PUT requests")
        print("✓ Response status codes should match above results")
        print("✓ Processing times should be captured in milliseconds")
        print("✓ User agent and custom headers should be preserved")
        print("✓ Dynamic endpoint classification should be applied")
        print()
        
        # Provide audit log verification commands
        print("🔍 AUDIT LOG VERIFICATION:")
        print("-" * 60)
        print("Check your audit logs (SQS or local) for the following patterns:")
        print()
        
        for result in self.test_results:
            if result["success"]:
                endpoint = f"{result['method']} {result['path']}"
                print(f"  - endPoint: \"{endpoint}\"")
        
        print()
        print("💡 Each audit log should contain:")
        print("  - requestId: unique identifier")
        print("  - userId: 'anonymous' or extracted from Bearer token") 
        print("  - activityType: dynamically determined based on endpoint")
        print("  - clientIp: client IP address")
        print("  - remarks: JSON with full transaction details")
        print("  - timestamp: ISO 8601 formatted UTC timestamp")

async def main():
    """Main test function"""
    print("🚀 DYNAMIC AUDIT LOGGING TEST")
    print("=" * 60)
    print("This test verifies that ALL API transactions are captured")
    print("dynamically in the audit system with complete details.")
    print()
    
    # Initialize test client
    test_client = AuditTestClient()
    
    try:
        await test_client.test_dynamic_endpoint_capture()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    
    print("🏁 Test completed!")
    print("\nNext steps:")
    print("1. Check your audit logs (SQS queue or local logs)")
    print("2. Verify all endpoints are captured with full details")
    print("3. Confirm dynamic activity type classification")
    print("4. Validate request/response body logging as configured")

if __name__ == "__main__":
    asyncio.run(main())