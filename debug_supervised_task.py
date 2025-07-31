#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

from binning.methods import SupervisedBinning

print("Testing SupervisedBinning task_type parameter...")

try:
    # Test with regression
    binner = SupervisedBinning(task_type="regression")
    print(f"task_type set to 'regression': {binner.task_type}")

    # Test with classification
    binner2 = SupervisedBinning(task_type="classification")
    print(f"task_type set to 'classification': {binner2.task_type}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
