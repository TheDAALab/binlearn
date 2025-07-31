#!/usr/bin/env python3

print("Testing SupervisedBinning attribute issue...")

import sys

sys.path.insert(0, ".")

try:
    from binning.methods import SupervisedBinning

    # Try to create instance
    print("Creating SupervisedBinning instance...")
    binner = SupervisedBinning(task_type="classification")

    print(f"task_type attribute: {hasattr(binner, 'task_type')}")
    if hasattr(binner, "task_type"):
        print(f"task_type value: {binner.task_type}")
    else:
        print("task_type attribute is missing!")

    print(f"tree_params attribute: {hasattr(binner, 'tree_params')}")
    if hasattr(binner, "tree_params"):
        print(f"tree_params value: {binner.tree_params}")
    else:
        print("tree_params attribute is missing!")

    # Check if initialization completed
    print(f"Initialization successful: {isinstance(binner, SupervisedBinning)}")

except Exception as e:
    print(f"Error creating SupervisedBinning: {e}")
    import traceback

    traceback.print_exc()
