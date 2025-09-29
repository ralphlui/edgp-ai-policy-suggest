#!/usr/bin/env python3
"""
Test script to validate environment variable configuration
This simulates how Kubernetes will inject environment variables
"""
import os
import sys

def test_env_config():
    print("🧪 Testing Environment Variable Configuration")
    print("=" * 55)
    
    # Set some test environment variables (simulating Kubernetes injection)
    test_env = {
        'ENVIRONMENT': 'production',
        'AWS_REGION': 'ap-southeast-1',
        'AOSS_HOST': 'test-host.ap-southeast-1.aoss.amazonaws.com',
        'AWS_ACCESS_KEY_ID': 'AKIA_TEST_KEY_ID',
        'AWS_SECRET_ACCESS_KEY': 'test_secret_key',
        'USE_AWS_SECRETS': 'true',
        'OPENAI_SECRET_NAME': 'test/edgp/secret2',
        'HOST': '0.0.0.0',
        'PORT': '8022'
    }
    
    print("📝 Setting test environment variables...")
    for key, value in test_env.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    print("\n🔧 Loading configuration...")
    try:
        from app.core.config import settings
        
        print(f"\n📋 Configuration loaded successfully:")
        print(f"   Environment: {settings.environment}")
        print(f"   Host: {settings.host}")
        print(f"   Port: {settings.port}")
        print(f"   AWS Region: {settings.aws_region}")
        print(f"   AOSS Host: {settings.aoss_host}")
        print(f"   AWS Access Key ID: {settings.aws_access_key_id[:8]}..." if settings.aws_access_key_id else "   AWS Access Key ID: Not set")
        print(f"   AWS Secret Key: {'Set' if settings.aws_secret_access_key else 'Not set'}")
        print(f"   Use AWS Secrets: {settings.use_aws_secrets}")
        print(f"   OpenAI Secret Name: {settings.openai_secret_name}")
        print(f"   OpenSearch Index: {settings.opensearch_index}")
        
        print("\n✅ All environment variables loaded successfully!")
        print("🐳 Your app is ready for Kubernetes deployment!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test environment variables
        for key in test_env.keys():
            if key in os.environ:
                del os.environ[key]

if __name__ == "__main__":
    success = test_env_config()
    sys.exit(0 if success else 1)