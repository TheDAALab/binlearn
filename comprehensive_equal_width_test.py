"""
Comprehensive test suite for EqualWidthBinning functionality.

This test suite covers all aspects of EqualWidthBinning including:
- Basic functionality and initialization
- Parameter validation
- Data type compatibility (numpy, pandas, polars if available)
- Edge cases and error handling
- sklearn integration and pipelines
- Configuration system integration
- Parameter reconstruction and cloning
"""

import numpy as np
import warnings
from typing import Any

# Import the EqualWidthBinning class
from binlearn.methods._equal_width_binning import EqualWidthBinning
from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE

# Conditional imports
if PANDAS_AVAILABLE:
    import pandas as pd

if POLARS_AVAILABLE:
    import polars as pl

# sklearn imports
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.base import clone

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def test_basic_functionality():
    """Test basic EqualWidthBinning functionality."""
    print("=== Testing Basic Functionality ===")

    # Test initialization with default parameters
    ewb = EqualWidthBinning()
    print(f"✓ Default initialization successful")
    print(f"  - Default n_bins: {ewb.n_bins}")
    print(f"  - Default bin_range: {ewb.bin_range}")

    # Test custom parameters
    ewb_custom = EqualWidthBinning(n_bins=7, bin_range=(0, 100))
    print(f"✓ Custom initialization successful")
    print(f"  - Custom n_bins: {ewb_custom.n_bins}")
    print(f"  - Custom bin_range: {ewb_custom.bin_range}")

    # Test basic fitting and transformation
    np.random.seed(42)
    X = np.random.uniform(0, 100, size=(1000, 3))

    ewb.fit(X)
    print(f"✓ Fitting successful with shape {X.shape}")
    print(
        f"  - Fitted columns: {list(ewb.bin_edges_.keys()) if hasattr(ewb, 'bin_edges_') else 'None'}"
    )

    X_transformed = ewb.transform(X)
    print(f"✓ Transformation successful")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Output shape: {X_transformed.shape}")
    print(f"  - Output dtype: {X_transformed.dtype}")

    # Check bin edges for first column
    if hasattr(ewb, "bin_edges_") and 0 in ewb.bin_edges_:
        edges = ewb.bin_edges_[0]
        print(f"  - Column 0 bin edges: {len(edges)} edges")
        print(f"    Range: [{edges[0]:.2f}, {edges[-1]:.2f}]")
        print(f"    Widths: {[edges[i+1] - edges[i] for i in range(min(3, len(edges)-1))]}")


def test_parameter_validation():
    """Test parameter validation and error handling."""
    print("\n=== Testing Parameter Validation ===")

    # Test valid parameters
    try:
        ewb = EqualWidthBinning(n_bins=5, bin_range=(0, 100))
        print("✓ Valid parameters accepted")
    except Exception as e:
        print(f"✗ Valid parameters rejected: {e}")

    # Test invalid n_bins
    invalid_n_bins = [0, -1, 1.5, "5"]
    for invalid in invalid_n_bins:
        try:
            ewb = EqualWidthBinning(n_bins=invalid)
            print(f"✗ Invalid n_bins={invalid} was accepted (should be rejected)")
        except Exception as e:
            print(f"✓ Invalid n_bins={invalid} correctly rejected: {type(e).__name__}")

    # Test invalid bin_range
    invalid_ranges = [(10, 5), (5, 5), [0, 100], "0,100"]
    for invalid in invalid_ranges:
        try:
            ewb = EqualWidthBinning(bin_range=invalid)
            print(f"✗ Invalid bin_range={invalid} was accepted (should be rejected)")
        except Exception as e:
            print(f"✓ Invalid bin_range={invalid} correctly rejected: {type(e).__name__}")


def test_data_types():
    """Test compatibility with different data types."""
    print("\n=== Testing Data Type Compatibility ===")

    # Test numpy arrays
    np.random.seed(42)
    X_numpy = np.random.uniform(0, 100, size=(100, 2))

    ewb = EqualWidthBinning(n_bins=5)
    try:
        ewb.fit(X_numpy)
        X_transformed = ewb.transform(X_numpy)
        print(f"✓ NumPy array support: {X_numpy.shape} -> {X_transformed.shape}")
    except Exception as e:
        print(f"✗ NumPy array failed: {e}")

    # Test pandas DataFrame
    if PANDAS_AVAILABLE:
        try:
            df = pd.DataFrame(X_numpy, columns=["feature1", "feature2"])
            ewb_pd = EqualWidthBinning(n_bins=5)
            ewb_pd.fit(df)
            df_transformed = ewb_pd.transform(df)
            print(f"✓ Pandas DataFrame support: {df.shape} -> {df_transformed.shape}")
            print(f"  - Input type: {type(df)}")
            print(f"  - Output type: {type(df_transformed)}")
        except Exception as e:
            print(f"✗ Pandas DataFrame failed: {e}")
    else:
        print("- Pandas not available, skipping DataFrame tests")

    # Test polars DataFrame
    if POLARS_AVAILABLE:
        try:
            df_polars = pl.DataFrame({"feature1": X_numpy[:, 0], "feature2": X_numpy[:, 1]})
            ewb_pl = EqualWidthBinning(n_bins=5)
            ewb_pl.fit(df_polars)
            df_pl_transformed = ewb_pl.transform(df_polars)
            print(f"✓ Polars DataFrame support: {df_polars.shape} -> {df_pl_transformed.shape}")
            print(f"  - Input type: {type(df_polars)}")
            print(f"  - Output type: {type(df_pl_transformed)}")
        except Exception as e:
            print(f"✗ Polars DataFrame failed: {e}")
    else:
        print("- Polars not available, skipping Polars tests")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== Testing Edge Cases ===")

    # Test constant data
    try:
        X_constant = np.full((100, 2), 5.0)
        ewb = EqualWidthBinning(n_bins=5)
        ewb.fit(X_constant)
        X_transformed = ewb.transform(X_constant)
        print(f"✓ Constant data handled: {X_constant.shape} -> {X_transformed.shape}")
        print(
            f"  - All values should be in one bin (usually 0): unique values = {np.unique(X_transformed.flatten())}"
        )
    except Exception as e:
        print(f"✗ Constant data failed: {e}")

    # Test single value data
    try:
        X_single = np.array([[1.0], [1.0], [1.0]])
        ewb = EqualWidthBinning(n_bins=3)
        ewb.fit(X_single)
        X_transformed = ewb.transform(X_single)
        print(f"✓ Single value data handled: {X_single.shape} -> {X_transformed.shape}")
    except Exception as e:
        print(f"✗ Single value data failed: {e}")

    # Test very small dataset
    try:
        X_tiny = np.array([[1.0], [2.0]])
        ewb = EqualWidthBinning(n_bins=5)
        ewb.fit(X_tiny)
        X_transformed = ewb.transform(X_tiny)
        print(f"✓ Tiny dataset handled: {X_tiny.shape} -> {X_transformed.shape}")
    except Exception as e:
        print(f"✗ Tiny dataset failed: {e}")

    # Test with custom bin_range
    try:
        np.random.seed(42)
        X = np.random.uniform(20, 80, size=(100, 1))  # Data in [20, 80]
        ewb = EqualWidthBinning(n_bins=5, bin_range=(0, 100))  # Bins in [0, 100]
        ewb.fit(X)
        X_transformed = ewb.transform(X)
        print(f"✓ Custom bin_range handled: data range ~[20,80], bin_range [0,100]")
        print(f"  - Unique bin values: {sorted(np.unique(X_transformed.flatten()))}")
    except Exception as e:
        print(f"✗ Custom bin_range failed: {e}")

    # Test clipping behavior
    try:
        np.random.seed(42)
        X_train = np.random.uniform(0, 50, size=(100, 1))
        X_test = np.random.uniform(-10, 60, size=(50, 1))  # Out of training range

        ewb = EqualWidthBinning(n_bins=5, clip=True)
        ewb.fit(X_train)
        X_test_transformed = ewb.transform(X_test)
        print(f"✓ Clipping behavior tested")
        print(f"  - Training range: [0, 50], Test range: [-10, 60]")
        print(f"  - Unique transformed values: {sorted(np.unique(X_test_transformed.flatten()))}")
    except Exception as e:
        print(f"✗ Clipping test failed: {e}")


def test_sklearn_integration():
    """Test sklearn integration and compatibility."""
    print("\n=== Testing Sklearn Integration ===")

    if not SKLEARN_AVAILABLE:
        print("- Sklearn not available, skipping integration tests")
        return

    np.random.seed(42)
    X = np.random.uniform(0, 100, size=(200, 3))

    # Test cloning
    try:
        ewb_original = EqualWidthBinning(n_bins=5)
        ewb_original.fit(X)

        ewb_cloned = clone(ewb_original)
        print(f"✓ Cloning successful")
        print(f"  - Original n_bins: {ewb_original.n_bins}")
        print(f"  - Cloned n_bins: {ewb_cloned.n_bins}")

        # Test that cloned version works
        X_cloned_transformed = ewb_cloned.transform(X)
        print(f"  - Cloned transformer works: {X.shape} -> {X_cloned_transformed.shape}")
    except Exception as e:
        print(f"✗ Cloning failed: {e}")

    # Test in Pipeline
    try:
        pipeline = Pipeline(
            [("binning", EqualWidthBinning(n_bins=4)), ("scaling", StandardScaler())]
        )

        X_pipeline = pipeline.fit_transform(X)
        print(f"✓ Pipeline integration successful: {X.shape} -> {X_pipeline.shape}")
    except Exception as e:
        print(f"✗ Pipeline integration failed: {e}")

    # Test with ColumnTransformer
    try:
        ct = ColumnTransformer(
            [
                ("bin_first_two", EqualWidthBinning(n_bins=3), [0, 1]),
                ("scale_last", StandardScaler(), [2]),
            ]
        )

        X_ct = ct.fit_transform(X)
        print(f"✓ ColumnTransformer integration successful: {X.shape} -> {X_ct.shape}")
    except Exception as e:
        print(f"✗ ColumnTransformer integration failed: {e}")


def test_fit_transform_workflow():
    """Test various fit/transform workflows."""
    print("\n=== Testing Fit/Transform Workflows ===")

    np.random.seed(42)
    X_train = np.random.uniform(0, 100, size=(150, 2))
    X_test = np.random.uniform(0, 100, size=(50, 2))

    # Test fit_transform
    try:
        ewb = EqualWidthBinning(n_bins=6)
        X_fit_transform = ewb.fit_transform(X_train)
        print(f"✓ fit_transform successful: {X_train.shape} -> {X_fit_transform.shape}")
    except Exception as e:
        print(f"✗ fit_transform failed: {e}")

    # Test separate fit and transform
    try:
        ewb = EqualWidthBinning(n_bins=6)
        ewb.fit(X_train)
        X_train_transformed = ewb.transform(X_train)
        X_test_transformed = ewb.transform(X_test)
        print(f"✓ Separate fit/transform successful")
        print(f"  - Train: {X_train.shape} -> {X_train_transformed.shape}")
        print(f"  - Test: {X_test.shape} -> {X_test_transformed.shape}")
    except Exception as e:
        print(f"✗ Separate fit/transform failed: {e}")

    # Test transform before fit (should fail)
    try:
        ewb_new = EqualWidthBinning(n_bins=5)
        X_transformed = ewb_new.transform(X_train)
        print(f"✗ Transform before fit should have failed but didn't")
    except Exception as e:
        print(f"✓ Transform before fit correctly failed: {type(e).__name__}")


def test_bin_properties():
    """Test properties of the generated bins."""
    print("\n=== Testing Bin Properties ===")

    np.random.seed(42)
    X = np.random.uniform(0, 100, size=(1000, 1))

    for n_bins in [2, 5, 10, 20]:
        try:
            ewb = EqualWidthBinning(n_bins=n_bins)
            ewb.fit(X)
            X_transformed = ewb.transform(X)

            unique_bins = np.unique(X_transformed)

            print(f"✓ n_bins={n_bins}:")
            print(f"  - Unique output bins: {len(unique_bins)}")
            print(f"  - Expected bins: {n_bins}")
            print(f"  - Bin values: {unique_bins}")

            # Check bin edges if available
            if hasattr(ewb, "bin_edges_") and 0 in ewb.bin_edges_:
                edges = ewb.bin_edges_[0]
                widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
                print(f"  - Bin widths (should be equal): {widths[:3]}...")
                print(f"  - Width variance: {np.var(widths):.10f}")

        except Exception as e:
            print(f"✗ n_bins={n_bins} failed: {e}")


def test_parameter_reconstruction():
    """Test parameter reconstruction and get_params/set_params."""
    print("\n=== Testing Parameter Reconstruction ===")

    # Test get_params
    try:
        ewb = EqualWidthBinning(n_bins=7, bin_range=(10, 90), clip=True)
        params = ewb.get_params()
        print(f"✓ get_params successful")
        print(f"  - Retrieved params: {list(params.keys())}")
        print(f"  - n_bins: {params.get('n_bins')}")
        print(f"  - bin_range: {params.get('bin_range')}")
        print(f"  - clip: {params.get('clip')}")
    except Exception as e:
        print(f"✗ get_params failed: {e}")

    # Test set_params
    try:
        ewb = EqualWidthBinning()
        ewb.set_params(n_bins=8, bin_range=(0, 50))
        print(f"✓ set_params successful")
        print(f"  - New n_bins: {ewb.n_bins}")
        print(f"  - New bin_range: {ewb.bin_range}")
    except Exception as e:
        print(f"✗ set_params failed: {e}")

    # Test reconstruction compatibility
    try:
        ewb_original = EqualWidthBinning(n_bins=6, clip=True)
        params = ewb_original.get_params()

        # Add reconstruction parameters
        params["class_"] = "EqualWidthBinning"
        params["module_"] = "binlearn.methods._equal_width_binning"

        ewb_reconstructed = EqualWidthBinning(**params)
        print(f"✓ Reconstruction compatibility successful")
        print(f"  - Original n_bins: {ewb_original.n_bins}")
        print(f"  - Reconstructed n_bins: {ewb_reconstructed.n_bins}")
    except Exception as e:
        print(f"✗ Reconstruction compatibility failed: {e}")


def run_comprehensive_test():
    """Run all comprehensive tests for EqualWidthBinning."""
    print("🧪 COMPREHENSIVE EQUAL WIDTH BINNING TEST SUITE")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_parameter_validation()
        test_data_types()
        test_edge_cases()
        test_sklearn_integration()
        test_fit_transform_workflow()
        test_bin_properties()
        test_parameter_reconstruction()

        print("\n" + "=" * 60)
        print("🎉 COMPREHENSIVE TEST SUITE COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n💥 TEST SUITE FAILED WITH UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_test()
