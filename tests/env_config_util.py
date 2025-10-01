#!/usr/bin/env python3
"""
Environment Configuration Utility for EDGP AI Policy Suggest

Usage:
    python env_config_util.py show                    # Show current environment
    python env_config_util.py list                    # List available environments  
    python env_config_util.py set development         # Set APP_ENV to development
    python env_config_util.py validate sit            # Validate a specific environment
"""

import os
import sys
import argparse
from pathlib import Path

def show_current_env():
    """Show current environment configuration"""
    app_env = os.getenv('APP_ENV', 'not set')
    environment = os.getenv('ENVIRONMENT', 'not set')
    
    print("🌍 Current Environment Configuration")
    print("=" * 50)
    print(f"APP_ENV: {app_env}")
    print(f"ENVIRONMENT: {environment}")
    
    if app_env != 'not set':
        env_file = f".env.{app_env}"
        if os.path.exists(env_file):
            print(f" Environment file exists: {env_file}")
        else:
            print(f" Environment file missing: {env_file}")

def list_environments():
    """List all available environment files"""
    print(" Available Environment Files")
    print("=" * 50)
    
    env_files = list(Path('.').glob('.env.*'))
    
    for env_file in sorted(env_files):
        env_name = env_file.name.replace('.env.', '')
        size = env_file.stat().st_size
        print(f" {env_name:<12} ({env_file}, {size} bytes)")
    
    if not env_files:
        print(" No environment files found")
    
    print(f"\nTotal: {len(env_files)} environment files")

def set_environment(env_name):
    """Set APP_ENV environment variable"""
    env_file = f".env.{env_name}"
    
    if not os.path.exists(env_file):
        print(f" Environment file {env_file} does not exist")
        print("Available environments:")
        list_environments()
        return False
    
    # Export to current shell (note: this only affects the Python process)
    os.environ['APP_ENV'] = env_name
    
    print(f" APP_ENV set to: {env_name}")
    print(f" Will use: {env_file}")
    print()
    print("Note: This only affects the current Python process.")
    print("To set for your shell session, run:")
    print(f"export APP_ENV={env_name}")
    
    return True

def validate_environment(env_name):
    """Validate an environment configuration"""
    print(f" Validating Environment: {env_name}")
    print("=" * 50)
    
    env_file = f".env.{env_name}"
    
    # Check if file exists
    if not os.path.exists(env_file):
        print(f" Environment file missing: {env_file}")
        return False
    
    print(f" Environment file exists: {env_file}")
    
    # Try to load and validate configuration
    os.environ['APP_ENV'] = env_name
    
    # Remove cached config if exists
    if 'app.core.config' in sys.modules:
        del sys.modules['app.core.config']
    
    try:
        sys.path.insert(0, '.')
        from app.core.config import settings, app_env, env_file_path
        
        print(f" Configuration loaded successfully")
        print(f"    APP_ENV: {app_env}")
        print(f"    File: {env_file_path}")
        print(f"    Host: {settings.host}")
        print(f"    Port: {settings.port}")
        print(f"    Environment: {settings.environment}")
        print(f"    Log Level: {settings.log_level}")
        
        # Check critical settings
        issues = []
        if not settings.jwt_public_key:
            issues.append("JWT public key not configured")
        if not settings.admin_api_url:
            issues.append("Admin API URL not configured")
        if not settings.rule_api_url:
            issues.append("Rule API URL not configured")
            
        if issues:
            print(f"\n Configuration Issues:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"\n All critical settings configured")
            
        return True
        
    except Exception as e:
        print(f" Configuration validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Environment Configuration Utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Show command
    subparsers.add_parser('show', help='Show current environment')
    
    # List command  
    subparsers.add_parser('list', help='List available environments')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set APP_ENV')
    set_parser.add_argument('environment', help='Environment name to set')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate environment')
    validate_parser.add_argument('environment', help='Environment name to validate')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'show':
        show_current_env()
    elif args.command == 'list':
        list_environments()
    elif args.command == 'set':
        set_environment(args.environment)
    elif args.command == 'validate':
        validate_environment(args.environment)

if __name__ == '__main__':
    main()