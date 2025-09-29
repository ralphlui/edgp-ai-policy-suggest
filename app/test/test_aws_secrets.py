#!/usr/bin/env python3
"""
AWS Secrets Manager Test for EDGP AI Policy Suggest
This Python script tests the AWS Secrets Manager integration
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_aws_secrets_integration():
    """Test AWS Secrets Manager integration"""
    print("🔐 Testing AWS Secrets Manager Integration")
    print("=" * 50)
    
    try:
        # Test importing the configuration
        print("📦 Importing configuration...")
        from app.core.config import (
            OPENAI_API_KEY, 
            USE_AWS_SECRETS, 
            OPENAI_SECRET_NAME,
            AWS_REGION,
            settings
        )
        
        # Display configuration status
        print(f"\n📋 Configuration Status:")
        print(f"   Environment: {settings.environment}")
        print(f"   USE_AWS_SECRETS: {USE_AWS_SECRETS}")
        print(f"   OPENAI_SECRET_NAME: {OPENAI_SECRET_NAME}")
        print(f"   AWS_REGION: {AWS_REGION}")
        
        # Check AWS credentials
        aws_access_key = settings.aws_access_key_id
        aws_secret_key = settings.aws_secret_access_key
        
        print(f"\n🔑 AWS Credentials:")
        print(f"   AWS_ACCESS_KEY_ID: {'✅ Set' if aws_access_key else '❌ Missing'}")
        print(f"   AWS_SECRET_ACCESS_KEY: {'✅ Set' if aws_secret_key else '❌ Missing'}")
        print(f"   AOSS_HOST: {'✅ Set' if settings.aoss_host else '❌ Missing'}")
        
        # Test OpenAI API Key
        print(f"\n🤖 OpenAI API Key:")
        if OPENAI_API_KEY:
            print(f"   ✅ OpenAI API Key loaded successfully")
            print(f"   Key starts with: {OPENAI_API_KEY[:8]}...")
            print(f"   Key length: {len(OPENAI_API_KEY)} characters")
            
            # Test OpenAI client initialization
            print(f"\n🧪 Testing OpenAI Client...")
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                print(f"   ✅ OpenAI client initialized successfully")
                
                # Test a simple API call (optional - uncomment if you want to test API)
                # print(f"   🌐 Testing API connection...")
                # models = client.models.list()
                # print(f"   ✅ API connection successful - {len(models.data)} models available")
                
            except Exception as e:
                print(f"   ❌ OpenAI client initialization failed: {e}")
                return False
        else:
            print(f"   ❌ OpenAI API Key not available")
            print(f"   Possible causes:")
            if USE_AWS_SECRETS:
                print(f"     - AWS Secrets Manager secret '{OPENAI_SECRET_NAME}' not found")
                print(f"     - AWS credentials not configured properly")
                print(f"     - IAM permissions missing for secretsmanager:GetSecretValue")
            else:
                print(f"     - AWS Secrets Manager is disabled")
                print(f"     - OPENAI_API_KEY_FALLBACK not set in .env file")
            return False
        
        # Test AWS Secrets Manager function directly
        if USE_AWS_SECRETS:
            print(f"\n🔍 Testing AWS Secrets Manager function directly...")
            try:
                from app.core.config import get_secret_from_aws
                test_secret = get_secret_from_aws(OPENAI_SECRET_NAME, AWS_REGION)
                if test_secret:
                    print(f"   ✅ Direct AWS Secrets Manager call successful")
                else:
                    print(f"   ⚠️ AWS Secrets Manager call returned None")
            except Exception as e:
                print(f"   ❌ AWS Secrets Manager function failed: {e}")
        
        print(f"\n🎉 AWS Secrets Manager integration test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_file_loading():
    """Test .env file loading"""
    print(f"\n📁 Testing .env file loading...")
    
    env = os.getenv("ENVIRONMENT", "development")
    env_file = f".env.{env}"
    
    if os.path.exists(env_file):
        print(f"   ✅ Environment file exists: {env_file}")
        
        # Read and display key lines
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        key_vars = ['USE_AWS_SECRETS', 'OPENAI_SECRET_NAME', 'AWS_REGION', 'AOSS_HOST']
        print(f"   📝 Key environment variables:")
        for line in lines:
            line = line.strip()
            if any(var in line for var in key_vars) and not line.startswith('#'):
                print(f"     {line}")
    else:
        print(f"   ❌ Environment file not found: {env_file}")

if __name__ == "__main__":
    print("🧪 EDGP AI Policy Suggest - AWS Secrets Manager Test")
    print("=" * 60)
    
    # Test environment file loading
    test_env_file_loading()
    
    # Test AWS Secrets Manager integration
    success = test_aws_secrets_integration()
    
    if success:
        print(f"\n✅ All tests passed! Your AWS Secrets Manager integration is working.")
    else:
        print(f"\n❌ Some tests failed. Please check the configuration.")
    
    sys.exit(0 if success else 1)