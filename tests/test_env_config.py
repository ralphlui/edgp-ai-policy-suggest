#!/usr/bin/env python3
"""
Test script to verify environment configuration loading
"""

import os
import sys
sys.path.insert(0, '.')

# Test different APP_ENV values
test_environments = ['development', 'production', 'sit', 'test']

print(" Testing Environment Configuration Loading")
print("=" * 60)

for env in test_environments:
    print(f"\n Testing APP_ENV={env}")
    print("-" * 40)
    
    # Set APP_ENV
    os.environ['APP_ENV'] = env
    
    # Remove any cached imports
    if 'app.core.config' in sys.modules:
        del sys.modules['app.core.config']
    
    try:
        # Import config to trigger loading
        from app.core.config import settings, app_env, env_file_path
        
        print(f" Successfully loaded configuration")
        print(f"    Detected APP_ENV: {app_env}")
        print(f"    Environment file: {env_file_path}")
        print(f"    Host: {settings.host}")
        print(f"    Port: {settings.port}")
        print(f"    Environment: {settings.environment}")
        print(f"    Log Level: {settings.log_level}")
        
    except Exception as e:
        print(f" Failed to load configuration: {e}")

# Test with non-existent environment
print(f"\n Testing APP_ENV=nonexistent")
print("-" * 40)
os.environ['APP_ENV'] = 'nonexistent'

if 'app.core.config' in sys.modules:
    del sys.modules['app.core.config']

try:
    from app.core.config import settings, app_env, env_file_path
    print(f" Fallback worked")
    print(f"    Detected APP_ENV: {app_env}")
    print(f"    Environment file: {env_file_path}")
    print(f"    Environment: {settings.environment}")
except Exception as e:
    print(f" Fallback failed: {e}")

print("\n" + "=" * 60)
print(" Environment configuration test completed!")