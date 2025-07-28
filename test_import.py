#!/usr/bin/env python3
"""Test script to debug import issues."""

print("Starting import test...")

try:
    print("Testing config import...")
    from binning.config import get_config
    print("✅ Config import successful")
    
    print("Testing base types import...")
    from binning.base._types import ColumnId
    print("✅ Base types import successful")
    
    print("Testing base classes import...")
    from binning.base import GeneralBinningBase
    print("✅ Base classes import successful")
    
    print("Testing main import...")
    import binning
    print("✅ Main import successful")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
