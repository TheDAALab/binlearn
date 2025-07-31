#!/usr/bin/env python3

print("Testing all concrete binning classes with sklearn clone...")

import sys

sys.path.insert(0, ".")

from sklearn.base import clone
import numpy as np

# Test classes
tests = [
    {
        "name": "EqualWidthBinning",
        "class": "binning.methods._equal_width_binning.EqualWidthBinning",
        "params": {"n_bins": 5},
    },
    {
        "name": "ManualIntervalBinning",
        "class": "binning.methods._manual_interval_binning.ManualIntervalBinning",
        "params": {"bin_edges": {0: [0, 10, 20, 30]}},
    },
    {
        "name": "OneHotBinning",
        "class": "binning.methods._onehot_binning.OneHotBinning",
        "params": {"max_unique_values": 10},
    },
    {
        "name": "SupervisedBinning",
        "class": "binning.methods._supervised_binning.SupervisedBinning",
        "params": {"task_type": "classification"},
    },
]

results = []

for test in tests:
    print(f"\n=== Testing {test['name']} ===")
    try:
        # Import the class
        module_path, class_name = test["class"].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)

        # Create instance
        instance = cls(**test["params"])
        print(f"✓ {test['name']} created successfully")

        # Get original params
        orig_params = instance.get_params()
        print(f"✓ get_params() works")

        # Test sklearn clone
        cloned = clone(instance)
        print(f"✓ sklearn clone succeeded")

        # Check params are identical
        cloned_params = cloned.get_params()
        if orig_params == cloned_params:
            print(f"✓ Parameters identical")
        else:
            print(f"✗ Parameters differ!")
            print(f"  Original: {orig_params}")
            print(f"  Cloned:   {cloned_params}")

        # Check they're different objects
        if instance is not cloned:
            print(f"✓ Different objects")
        else:
            print(f"✗ Same object!")

        results.append({"name": test["name"], "success": True})
        print(f"✓ {test['name']} clone test PASSED")

    except Exception as e:
        print(f"✗ {test['name']} clone test FAILED: {e}")
        results.append({"name": test["name"], "success": False, "error": str(e)})

print(f"\n=== SUMMARY ===")
passed = sum(1 for r in results if r["success"])
total = len(results)
print(f"Passed: {passed}/{total}")

for result in results:
    status = "✓" if result["success"] else "✗"
    error_msg = f" - {result['error']}" if not result["success"] else ""
    print(f"{status} {result['name']}{error_msg}")

if passed == total:
    print(f"\n🎉 ALL TESTS PASSED! sklearn cloning works for all binning classes")
else:
    print(f"\n❌ {total - passed} tests failed")
