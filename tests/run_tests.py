#!/usr/bin/env python3
"""
Updated test runner for EDGP AI Policy Suggest application
Automatically loads .env.test environment and runs test suite with coverage
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and setup test environment first
from tests.test_config import setup_test_environment

def run_tests_with_coverage():
    """Run tests with coverage reporting using .env.test"""
    
    print("🧪 Setting up test environment...")
    setup_test_environment()
    
    # Change to project root directory
    os.chdir(PROJECT_ROOT)
    
    print("🧪 Running comprehensive test suite with coverage...")
    
    # Run pytest with coverage and async support
    cmd = [
        sys.executable, "-m", "pytest",
        "--verbose",
        "--asyncio-mode=auto",  # Enable async support
        "--cov=app",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=90",  # Aim for 90%+ coverage
        "tests/"
    ]
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print(f"\n❌ Tests failed with exit code: {result.returncode}")
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def run_specific_tests(test_pattern=None, markers=None):
    """Run specific tests based on pattern or markers"""
    
    setup_test_environment()
    os.chdir(PROJECT_ROOT)
    
    cmd = [sys.executable, "-m", "pytest", "--verbose"]
    
    if markers:
        cmd.extend(["-m", markers])
    
    if test_pattern:
        cmd.extend(["-k", test_pattern])
    
    cmd.append("tests/")
    
    print(f"🧪 Running tests with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for EDGP AI Policy Suggest")
    parser.add_argument("--pattern", "-k", help="Run tests matching pattern")
    parser.add_argument("--markers", "-m", help="Run tests with specific markers (e.g., 'auth', 'unit', 'integration')")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    
    args = parser.parse_args()
    
    if args.pattern or args.markers:
        return run_specific_tests(args.pattern, args.markers)
    elif args.no_coverage:
        setup_test_environment()
        os.chdir(PROJECT_ROOT)
        result = subprocess.run([sys.executable, "-m", "pytest", "--verbose", "tests/"])
        return result.returncode
    else:
        return run_tests_with_coverage()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)