#!/usr/bin/env python3
"""Debug sklearn clone issue."""

from sklearn.base import clone
from binning.methods._supervised_binning import SupervisedBinning

# Create a simple instance with tree_params=None explicitly
print("Creating SupervisedBinning instance with tree_params=None...")
original = SupervisedBinning(task_type="regression", tree_params=None)
print(f"After __init__ - tree_params: {original.tree_params} (type: {type(original.tree_params)})")

# Check what get_params returns
params = original.get_params()
print(f"get_params() returns tree_params: {params.get('tree_params')} (type: {type(params.get('tree_params'))})")

# Check if this is the issue - try setting tree_params back to None
print("\nSetting tree_params back to None...")
original.tree_params = None
print(f"After manual set - tree_params: {original.tree_params} (type: {type(original.tree_params)})")

# Try clone now
try:
    cloned = clone(original)
    print("✓ Clone successful after manual fix!")
except Exception as e:
    print(f"✗ Clone still failed: {e}")

# Let's also check what happens during set_params
print(f"\nCalling set_params with tree_params=None...")
original.set_params(tree_params=None)
print(f"After set_params(tree_params=None) - tree_params: {original.tree_params} (type: {type(original.tree_params)})")
