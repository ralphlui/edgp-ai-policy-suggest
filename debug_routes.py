#!/usr/bin/env python3
"""Debug script to test the fake app routes"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tests.test_main_app import _install_fake_pkg
from fastapi.testclient import TestClient

# Install fake package
_install_fake_pkg(include_validation=True)

# Import the fake module
import app.main as fake_main
client = TestClient(fake_main.app)

# Test available routes
print("Available routes:")
for route in fake_main.app.routes:
    print(f"  {route}")

print("\nTesting health endpoint:")
r = client.get("/api/aips/health")
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"Response: {r.json()}")

print("\nTesting info endpoint:")
r = client.get("/api/aips/info")
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print(f"Response: {r.json()}")
else:
    print(f"Error: {r.text}")