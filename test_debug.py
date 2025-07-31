import numpy as np
from binning.methods import EqualWidthMinimumWeightBinning

# Test with simple data
X = np.array([[1], [2], [3], [4]]).astype(float)
weights = np.array([1.0, 1.0, 1.0, 1.0])

print("Testing EqualWidthMinimumWeightBinning...")

# Test without explicit guidance_columns but with external data
ewmwb = EqualWidthMinimumWeightBinning(n_bins=2, minimum_weight=1.5)
try:
    print("Fitting with guidance_data...")
    ewmwb.fit(X, guidance_data=weights)
    print("Success without guidance_columns")
    result = ewmwb.transform(X)
    print(f"Transform result: {result}")
except Exception as e:
    print(f"Error without guidance_columns: {e}")
    import traceback

    traceback.print_exc()
