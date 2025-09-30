#!/usr/bin/env python3
"""
Test script to verify AWS Secrets Manager integration
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_secrets_manager():
    print(" Testing AWS Secrets Manager integration...")
    print(" This service now uses AWS Secrets Manager ONLY")
    print("=" * 60)
    
    try:
        from app.core.config import OPENAI_API_KEY, USE_AWS_SECRETS, OPENAI_SECRET_NAME
        
        print(f" Configuration:")
        print(f"   USE_AWS_SECRETS: {USE_AWS_SECRETS}")
        print(f"   OPENAI_SECRET_NAME: {OPENAI_SECRET_NAME}")
        print(f"   AWS_REGION: {os.getenv('AWS_REGION', 'not set')}")
        print(f"   AWS_ACCESS_KEY_ID: {'set' if os.getenv('AWS_ACCESS_KEY_ID') else 'not set'}")
        print()
        
        if not USE_AWS_SECRETS:
            print(" AWS Secrets Manager is disabled!")
            print("   Set USE_AWS_SECRETS=true in your environment")
            return False
        
        if OPENAI_API_KEY:
            print(" OpenAI API Key loaded successfully from AWS Secrets Manager!")
            print(f"   Key starts with: {OPENAI_API_KEY[:8]}...")
            print(f"   Key length: {len(OPENAI_API_KEY)}")
        else:
            print(" Failed to load OpenAI API Key from AWS Secrets Manager")
            print("   Please check:")
            print(f"   1. Secret '{OPENAI_SECRET_NAME}' exists in AWS Secrets Manager")
            print("   2. AWS credentials are configured")
            print("   3. IAM permissions for secretsmanager:GetSecretValue")
            return False
        
        # Test OpenAI client with the key
        print("\n🔑 Testing OpenAI client initialization...")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            print(" OpenAI client initialized successfully")
        except Exception as e:
            print(f" OpenAI client initialization failed: {e}")
            return False
        
        print("\n AWS Secrets Manager integration test passed!")
        print(" Your service is properly configured to use AWS Secrets Manager")
        return True
        
    except Exception as e:
        print(f" Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_secrets_manager()
    sys.exit(0 if success else 1)