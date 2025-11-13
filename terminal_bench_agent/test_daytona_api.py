#!/usr/bin/env python3
"""Test Daytona API connectivity."""

import os
from daytona import Daytona

api_key = os.environ.get("DAYTONA_API_KEY")
print(f"Testing Daytona API with key: {api_key[:20]}...{api_key[-4:]}")

try:
    client = Daytona()
    print("\nListing sandboxes...")
    result = client.list()
    sandboxes = getattr(result, 'items', [])
    print(f"SUCCESS: Found {len(sandboxes)} sandbox(es)")
    
    for sandbox in sandboxes[:3]:  # Show first 3
        print(f"  - {getattr(sandbox, 'id', 'unknown')}: {getattr(sandbox, 'status', 'unknown')}")
        
except Exception as e:
    print(f"\nERROR: Daytona API call failed")
    print(f"Exception: {type(e).__name__}: {e}")
