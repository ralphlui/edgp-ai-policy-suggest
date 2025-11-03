#!/usr/bin/env python3
"""
Simple script to test OpenSearch connection pooling.
This script simulates multiple API calls to verify connections are reused.
"""

import logging
import time
from app.vector_db.schema_loader import get_schema_by_domain, get_store

# Set up logging to see connection messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection_reuse():
    """Test that multiple calls reuse the same connection"""
    
    print(" Testing OpenSearch connection reuse...")
    print("=" * 50)
    
    # Test 1: Multiple schema retrievals
    print("\n Test 1: Multiple schema retrievals")
    domains = ["customer", "product", "order", "customer"]  # Repeat customer to test
    
    for i, domain in enumerate(domains, 1):
        print(f"\n Call {i}: Retrieving schema for domain '{domain}'")
        start_time = time.time()
        
        schema = get_schema_by_domain(domain)
        
        elapsed = time.time() - start_time
        print(f"  Response time: {elapsed:.3f}s")
        print(f" Schema columns found: {len(schema)}")
    
    # Test 2: Direct store access
    print("\n\n Test 2: Direct store access")
    for i in range(3):
        print(f"\n Call {i+1}: Getting store instance")
        start_time = time.time()
        
        store = get_store()
        
        elapsed = time.time() - start_time
        print(f"  Store creation time: {elapsed:.3f}s")
        print(f" Store instance ID: {id(store)}")
    
    print("\n Connection pooling test completed!")
    print(" If connections are reused properly, you should see:")
    print("   - Initial connection creation messages only once")
    print("   - Subsequent calls should be faster")
    print("   - Same store instance ID for multiple calls")

if __name__ == "__main__":
    test_connection_reuse()