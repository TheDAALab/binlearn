#!/usr/bin/env python3
"""Test to show bin_representatives flow in EqualWidthBinning parameter reconstruction."""

import numpy as np
from binlearn.methods import EqualWidthBinning


def explain_bin_representatives_flow():
    """Explain how bin_representatives flow through constructors in parameter reconstruction."""
    print("=" * 80)
    print("How bin_representatives Flow Through EqualWidthBinning Constructors")
    print("=" * 80)

    # Create sample data
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

    print("Sample data:")
    print(X)
    print()

    # Step 1: Fit original instance
    print("STEP 1: Create and fit original EqualWidthBinning")
    print("-" * 50)
    binner_original = EqualWidthBinning(n_bins=3)
    binner_original.fit(X)

    print("After fitting, the original instance has:")
    print(f"- bin_edges_: {getattr(binner_original, 'bin_edges_', 'Not found')}")
    print(
        f"- bin_representatives_: {getattr(binner_original, 'bin_representatives_', 'Not found')}"
    )
    print()

    # Step 2: Get parameters
    print("STEP 2: Extract parameters using get_params()")
    print("-" * 50)
    params = binner_original.get_params()
    print(f"Parameters returned: {list(params.keys())}")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    # Step 3: Parameter flow analysis
    print("STEP 3: Parameter Flow Analysis")
    print("-" * 50)
    print("When you call:")
    print("  new_binner = EqualWidthBinning(**params)")
    print()
    print("The constructor receives these fitted parameters:")
    print(f"  - bin_edges: {params.get('bin_edges', 'Missing')}")
    print(f"  - bin_representatives: {params.get('bin_representatives', 'Missing')}")
    print()

    # Step 4: Create reconstructed instance
    print("STEP 4: Create reconstructed instance")
    print("-" * 50)
    print("Constructor flow:")
    print("1. EqualWidthBinning.__init__() receives fitted parameters")
    print("2. Stores bin_edges and bin_representatives as instance attributes")
    print("3. If bin_edges provided, marks instance as fitted (_fitted = True)")
    print()

    binner_reconstructed = EqualWidthBinning(**params)

    print("After construction, reconstructed instance has:")
    print(f"- bin_edges (param): {getattr(binner_reconstructed, 'bin_edges', 'Not found')}")
    print(
        f"- bin_representatives (param): {getattr(binner_reconstructed, 'bin_representatives', 'Not found')}"
    )
    print(f"- bin_edges_: {getattr(binner_reconstructed, 'bin_edges_', 'Not found')}")
    print(
        f"- bin_representatives_: {getattr(binner_reconstructed, 'bin_representatives_', 'Not found')}"
    )
    print(f"- _fitted: {getattr(binner_reconstructed, '_fitted', False)}")
    print()

    # Step 5: Test functionality
    print("STEP 5: Test reconstructed instance functionality")
    print("-" * 50)
    try:
        X_original = binner_original.transform(X)
        X_reconstructed = binner_reconstructed.transform(X)
        print("Transform results match:", np.array_equal(X_original, X_reconstructed))

        X_inv_original = binner_original.inverse_transform(X_original)
        X_inv_reconstructed = binner_reconstructed.inverse_transform(X_reconstructed)
        print("Inverse transform results match:", np.allclose(X_inv_original, X_inv_reconstructed))

    except Exception as e:
        print(f"Error: {e}")

    print()

    # Step 6: Key insights
    print("STEP 6: Key Insights About bin_representatives Flow")
    print("-" * 50)
    print("🔑 KEY INSIGHT 1: Parameter vs. Fitted Attribute Storage")
    print(
        "   - Constructor parameters (bin_edges, bin_representatives) are stored as instance attributes"
    )
    print(
        "   - Fitted attributes (bin_edges_, bin_representatives_) are created during fitting or parameter reconstruction"
    )
    print()
    print("🔑 KEY INSIGHT 2: Parameter Reconstruction Workflow")
    print("   1. get_params() extracts BOTH constructor params AND fitted attributes")
    print("   2. Constructor receives fitted attributes as constructor parameters")
    print("   3. Constructor logic copies them to proper fitted attribute names")
    print("   4. Instance is marked as fitted, ready for transform/inverse_transform")
    print()
    print("🔑 KEY INSIGHT 3: Dual Storage Strategy")
    print("   - bin_representatives flows through constructor as a parameter")
    print("   - Gets stored as bin_representatives_ (fitted attribute) for actual use")
    print("   - This enables seamless parameter reconstruction without refitting")


if __name__ == "__main__":
    explain_bin_representatives_flow()
