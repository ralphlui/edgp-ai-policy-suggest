#!/usr/bin/env python3
"""
Test runner for EDGP AI Policy Suggest Microservice
"""
import os
import sys
import subprocess
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file"""
    print(f"🧪 Running {test_file}...")
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED")
            return True
        else:
            print(f"❌ {test_file} - FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"💥 {test_file} - ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 EDGP AI Policy Suggest Microservice - Test Runner")
    print("=" * 60)
    
    # Find all test files
    test_dir = Path("app/test")
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("❌ No test files found in app/test/")
        return 1
    
    print(f"📁 Found {len(test_files)} test files")
    
    # Run each test
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if run_test_file(str(test_file)):
            passed += 1
        else:
            failed += 1
        print("-" * 40)
    
    # Summary
    print("📊 Test Summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Total:  {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())