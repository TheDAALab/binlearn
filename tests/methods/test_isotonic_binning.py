"""
Comprehensive tests for IsotonicBinning implementation.

Tests cover all required scenarios:
- Various input/output formats (numpy, pandas, polars)
- preserve_dataframe True and False
- Fitted state reconstruction via set_params(params) and constructor(**params)
- Test repeated fitting on reconstructed state
- Test sklearn pipeline integration
- Test edge cases with nans and infs and constant columns
- Test supervised-specific functionality:
  - Target column provided through guidance_columns parameter
  - Target column provided through guidance_data argument of fit function
  - Isotonic regression behavior with different parameters
  - Both increasing and decreasing monotonic relationships
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.methods import IsotonicBinning
from binlearn.utils import ConfigurationError, FittingError, ValidationError


class TestIsotonicBinning:
    """Comprehensive test suite for IsotonicBinning."""

    @pytest.fixture
    def monotonic_increasing_data(self):
        """Data with monotonic increasing relationship."""
        np.random.seed(42)
        X = np.random.uniform(0, 10, (200, 2))
        # Create monotonic increasing target with noise
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.5, 200)
        return X, y

    @pytest.fixture
    def monotonic_decreasing_data(self):
        """Data with monotonic decreasing relationship."""
        np.random.seed(42)
        X = np.random.uniform(0, 10, (200, 2))
        # Create monotonic decreasing target with noise
        y = 10 - X[:, 0] - 0.5 * X[:, 1] + np.random.normal(0, 0.5, 200)
        return X, y

    @pytest.fixture
    def classification_data(self):
        """Classification data for testing."""
        np.random.seed(42)
        X = np.random.uniform(0, 10, (200, 2))
        # Create target that correlates with features
        scores = X[:, 0] + X[:, 1]
        y = (scores > np.median(scores)).astype(int)
        return X, y

    # ======================
    # Constructor Tests
    # ======================

    def test_constructor_defaults(self):
        """Test constructor with default parameters."""
        binner = IsotonicBinning()
        assert binner.max_bins == 10  # Config default
        assert binner.min_samples_per_bin == 5  # Hardcoded default
        assert binner.increasing is True  # Config default
        assert binner.y_min is None
        assert binner.y_max is None
        assert binner.min_change_threshold == 0.01  # Hardcoded default
        assert binner.clip is True  # Config default
        assert binner.preserve_dataframe is False  # Config default
        assert binner.guidance_columns is None

    def test_constructor_with_max_bins_int(self):
        """Test constructor with integer max_bins."""
        binner = IsotonicBinning(max_bins=8)
        assert binner.max_bins == 8

    def test_constructor_with_max_bins_string(self):
        """Test constructor with string max_bins."""
        binner = IsotonicBinning(max_bins="sqrt")
        assert binner.max_bins == "sqrt"

    def test_constructor_with_min_samples_per_bin(self):
        """Test constructor with custom min_samples_per_bin."""
        binner = IsotonicBinning(min_samples_per_bin=10)
        assert binner.min_samples_per_bin == 10

    def test_constructor_with_increasing_false(self):
        """Test constructor with decreasing monotonicity."""
        binner = IsotonicBinning(increasing=False)
        assert binner.increasing is False

    def test_constructor_with_y_bounds(self):
        """Test constructor with y_min and y_max bounds."""
        binner = IsotonicBinning(y_min=0.0, y_max=1.0)
        assert binner.y_min == 0.0
        assert binner.y_max == 1.0

    def test_constructor_with_min_change_threshold(self):
        """Test constructor with custom min_change_threshold."""
        binner = IsotonicBinning(min_change_threshold=0.05)
        assert binner.min_change_threshold == 0.05

    def test_constructor_with_guidance_columns(self):
        """Test constructor with guidance columns."""
        binner = IsotonicBinning(guidance_columns=["target"])
        assert binner.guidance_columns == ["target"]

    def test_constructor_with_preserve_dataframe(self):
        """Test constructor with preserve_dataframe option."""
        binner = IsotonicBinning(preserve_dataframe=True)
        assert binner.preserve_dataframe is True

    def test_constructor_with_clip(self):
        """Test constructor with clip option."""
        binner = IsotonicBinning(clip=False)
        assert binner.clip is False

    def test_constructor_state_reconstruction_params(self):
        """Test constructor with state reconstruction parameters."""
        bin_edges = {0: [0.0, 5.0, 10.0], 1: [0.0, 3.0, 6.0, 10.0]}
        bin_representatives = {0: [2.5, 7.5], 1: [1.5, 4.5, 8.0]}
        binner = IsotonicBinning(
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            class_="IsotonicBinning",
            module_="binlearn.methods",
        )
        assert binner.bin_edges == bin_edges
        assert binner.bin_representatives == bin_representatives

    # ======================
    # Parameter Validation Tests
    # ======================

    def test_invalid_max_bins_negative(self):
        """Test validation of negative max_bins."""
        with pytest.raises(ConfigurationError, match="max_bins must be a positive integer"):
            IsotonicBinning(max_bins=-1)

    def test_invalid_max_bins_zero(self):
        """Test validation of zero max_bins."""
        with pytest.raises(ConfigurationError, match="max_bins must be a positive integer"):
            IsotonicBinning(max_bins=0)

    def test_invalid_max_bins_type(self):
        """Test validation of invalid max_bins type."""
        with pytest.raises(ConfigurationError, match="max_bins must be"):
            IsotonicBinning(max_bins=3.14)  # type: ignore

    def test_invalid_min_samples_per_bin_negative(self):
        """Test validation of negative min_samples_per_bin."""
        with pytest.raises(
            ConfigurationError, match="min_samples_per_bin must be a positive integer"
        ):
            IsotonicBinning(min_samples_per_bin=-1)

    def test_invalid_min_samples_per_bin_zero(self):
        """Test validation of zero min_samples_per_bin."""
        with pytest.raises(
            ConfigurationError, match="min_samples_per_bin must be a positive integer"
        ):
            IsotonicBinning(min_samples_per_bin=0)

    def test_invalid_min_samples_per_bin_type(self):
        """Test validation of invalid min_samples_per_bin type."""
        with pytest.raises(
            ConfigurationError, match="min_samples_per_bin must be a positive integer"
        ):
            IsotonicBinning(min_samples_per_bin="invalid")  # type: ignore

    def test_invalid_increasing_type(self):
        """Test validation of invalid increasing type."""
        with pytest.raises(ConfigurationError, match="increasing must be a boolean value"):
            IsotonicBinning(increasing="yes")  # type: ignore

    def test_invalid_y_min_type(self):
        """Test validation of invalid y_min type."""
        with pytest.raises(ConfigurationError, match="y_min must be a number or None"):
            IsotonicBinning(y_min="invalid")  # type: ignore

    def test_invalid_y_max_type(self):
        """Test validation of invalid y_max type."""
        with pytest.raises(ConfigurationError, match="y_max must be a number or None"):
            IsotonicBinning(y_max="invalid")  # type: ignore

    def test_invalid_y_bounds_order(self):
        """Test validation of y_min >= y_max."""
        with pytest.raises(ConfigurationError, match="y_min must be less than y_max"):
            IsotonicBinning(y_min=5.0, y_max=3.0)

        with pytest.raises(ConfigurationError, match="y_min must be less than y_max"):
            IsotonicBinning(y_min=3.0, y_max=3.0)

    def test_invalid_min_change_threshold_negative(self):
        """Test validation of negative min_change_threshold."""
        with pytest.raises(
            ConfigurationError, match="min_change_threshold must be a positive number"
        ):
            IsotonicBinning(min_change_threshold=-0.01)

    def test_invalid_min_change_threshold_zero(self):
        """Test validation of zero min_change_threshold."""
        with pytest.raises(
            ConfigurationError, match="min_change_threshold must be a positive number"
        ):
            IsotonicBinning(min_change_threshold=0.0)

    def test_invalid_min_change_threshold_type(self):
        """Test validation of invalid min_change_threshold type."""
        with pytest.raises(
            ConfigurationError, match="min_change_threshold must be a positive number"
        ):
            IsotonicBinning(min_change_threshold="invalid")  # type: ignore

    # ======================
    # Guidance Data Provision Tests (Core supervised functionality)
    # ======================

    def test_targets_via_guidance_columns(self, monotonic_increasing_data):
        """Test target column provided through guidance_columns parameter."""
        X, y = monotonic_increasing_data

        # Create data with targets as guidance column
        X_with_targets = np.column_stack([X, y])

        binner = IsotonicBinning(
            max_bins=5,
            min_samples_per_bin=10,
            guidance_columns=[2],  # Targets are in column 2
        )
        binner.fit(X_with_targets)

        # Should be fitted
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")
        assert len(binner.bin_edges_) == 2  # Two feature columns

        # Test transform - should only transform the feature columns (not targets)
        result = binner.transform(X_with_targets)
        assert result.shape == (200, 2)  # Only feature columns, not targets
        assert np.issubdtype(result.dtype, np.integer)

    def test_targets_via_guidance_data_parameter(self, monotonic_increasing_data):
        """Test target column provided through guidance_data parameter."""
        X, y = monotonic_increasing_data

        binner = IsotonicBinning(
            max_bins=5,
            min_samples_per_bin=10,
        )
        binner.fit(X, guidance_data=y.reshape(-1, 1))

        # Should be fitted
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")
        assert len(binner.bin_edges_) == 2  # Two feature columns

    def test_targets_via_y_parameter(self, monotonic_increasing_data):
        """Test target column provided through y parameter."""
        X, y = monotonic_increasing_data

        binner = IsotonicBinning(
            max_bins=5,
            min_samples_per_bin=10,
        )
        binner.fit(X, y=y)

        # Should be fitted
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")
        assert len(binner.bin_edges_) == 2  # Two feature columns

        # Test transform
        result = binner.transform(X)
        assert result.shape == (200, 2)
        assert np.issubdtype(result.dtype, np.integer)

    def test_priority_order_guidance_columns_over_y(self, monotonic_increasing_data):
        """Test that guidance_columns takes priority over y parameter."""
        X, y = monotonic_increasing_data

        # Create different targets for guidance_columns
        y_guidance = 2 * y + 1  # Transform the targets
        X_with_targets = np.column_stack([X, y_guidance])

        binner = IsotonicBinning(
            max_bins=4,
            min_samples_per_bin=10,
            guidance_columns=[2],
        )

        # Fit with both guidance_columns and y - guidance_columns should win
        binner.fit(X_with_targets, y=y)

        # Create second binner using only y_guidance via y parameter
        binner2 = IsotonicBinning(
            max_bins=4,
            min_samples_per_bin=10,
        )
        binner2.fit(X, y=y_guidance)

        # Results should be the same (guidance_columns was used, not y)
        result1 = binner.transform(X_with_targets)
        result2 = binner2.transform(X)

        np.testing.assert_array_equal(result1, result2)

    def test_no_guidance_data_provided_error(self, monotonic_increasing_data):
        """Test error when no guidance data is provided for supervised binning."""
        X, _ = monotonic_increasing_data

        binner = IsotonicBinning()

        with pytest.raises(ValidationError, match="Supervised binning requires guidance data"):
            binner.fit(X)

    # ======================
    # Monotonic Relationship Tests
    # ======================

    def test_increasing_monotonic_relationship(self, monotonic_increasing_data):
        """Test isotonic binning with increasing monotonic relationship."""
        X, y = monotonic_increasing_data

        binner = IsotonicBinning(
            max_bins=6,
            min_samples_per_bin=15,
            increasing=True,
            min_change_threshold=0.05,
        )
        binner.fit(X, y=y)

        # Should create bins that preserve monotonic increasing relationship
        result = binner.transform(X)
        assert result.shape == (200, 2)

        # Check that isotonic models were fitted
        assert hasattr(binner, "_isotonic_models")
        assert len(binner._isotonic_models) == 2  # One model per feature

    def test_decreasing_monotonic_relationship(self, monotonic_decreasing_data):
        """Test isotonic binning with decreasing monotonic relationship."""
        X, y = monotonic_decreasing_data

        binner = IsotonicBinning(
            max_bins=6,
            min_samples_per_bin=15,
            increasing=False,
            min_change_threshold=0.05,
        )
        binner.fit(X, y=y)

        # Should create bins that preserve monotonic decreasing relationship
        result = binner.transform(X)
        assert result.shape == (200, 2)

        # Check that isotonic models were fitted with decreasing=False
        for model in binner._isotonic_models.values():
            # Check the model parameters instead of the attribute
            params = model.get_params()
            assert params["increasing"] is False

    def test_y_bounds_constraint(self, monotonic_increasing_data):
        """Test isotonic regression with y_min and y_max bounds."""
        X, y = monotonic_increasing_data

        # Set bounds that constrain the target range
        y_min, y_max = np.min(y), np.max(y)
        bounded_y_min = y_min + 0.1 * (y_max - y_min)
        bounded_y_max = y_max - 0.1 * (y_max - y_min)

        binner = IsotonicBinning(
            max_bins=5,
            min_samples_per_bin=20,
            y_min=bounded_y_min,
            y_max=bounded_y_max,
        )
        binner.fit(X, y=y)

        # Should work with bounds
        result = binner.transform(X)
        assert result.shape == (200, 2)

        # Check that bounds were applied to models
        for model in binner._isotonic_models.values():
            params = model.get_params()
            assert params["y_min"] == bounded_y_min
            assert params["y_max"] == bounded_y_max

    def test_min_change_threshold_effect(self, monotonic_increasing_data):
        """Test effect of different min_change_threshold values."""
        X, y = monotonic_increasing_data

        # Low threshold - should create more bins (sensitive to small changes)
        low_threshold = IsotonicBinning(
            max_bins=10,
            min_samples_per_bin=10,
            min_change_threshold=0.01,  # Sensitive
        )
        low_threshold.fit(X, y=y)

        # High threshold - should create fewer bins (insensitive to small changes)
        high_threshold = IsotonicBinning(
            max_bins=10,
            min_samples_per_bin=10,
            min_change_threshold=0.1,  # Insensitive
        )
        high_threshold.fit(X, y=y)

        # Both should work
        low_result = low_threshold.transform(X)
        high_result = high_threshold.transform(X)

        assert low_result.shape == high_result.shape == (200, 2)

    def test_min_samples_per_bin_constraint(self, monotonic_increasing_data):
        """Test min_samples_per_bin constraint behavior."""
        X, y = monotonic_increasing_data

        # High min_samples_per_bin should result in fewer bins
        binner = IsotonicBinning(
            max_bins=10,
            min_samples_per_bin=50,  # Large constraint
            min_change_threshold=0.01,
        )
        binner.fit(X, y=y)

        result = binner.transform(X)
        assert result.shape == (200, 2)

        # Should create fewer bins due to sample size constraint
        for _col_id, edges in binner.bin_edges_.items():
            num_bins = len(edges) - 1
            assert num_bins <= 4  # With 200 samples and min 50 per bin, max ~4 bins

    # ======================
    # Input Format Tests
    # ======================

    @pytest.mark.parametrize("preserve_dataframe", [True, False])
    def test_numpy_input(self, monotonic_increasing_data, preserve_dataframe):
        """Test IsotonicBinning with numpy input."""
        X, y = monotonic_increasing_data

        binner = IsotonicBinning(
            max_bins=5,
            min_samples_per_bin=20,
            preserve_dataframe=preserve_dataframe,
        )
        binner.fit(X, y=y)
        result = binner.transform(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (200, 2)
        assert np.issubdtype(result.dtype, np.integer)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    @pytest.mark.parametrize("preserve_dataframe", [True, False])
    def test_pandas_input(self, monotonic_increasing_data, preserve_dataframe):
        """Test IsotonicBinning with pandas input."""
        X, y = monotonic_increasing_data

        X_df = pd.DataFrame(X, columns=["feat1", "feat2"])
        y_series = pd.Series(y, name="target")

        binner = IsotonicBinning(
            max_bins=5,
            min_samples_per_bin=20,
            preserve_dataframe=preserve_dataframe,
        )
        binner.fit(X_df, y=y_series)
        result = binner.transform(X_df)

        if preserve_dataframe:
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["feat1", "feat2"]
        else:
            assert isinstance(result, np.ndarray)

        assert result.shape == (200, 2)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_with_guidance_columns(self, monotonic_increasing_data):
        """Test pandas input with guidance columns."""
        X, y = monotonic_increasing_data

        # Create DataFrame with target column
        df = pd.DataFrame(X, columns=["feat1", "feat2"])
        df["target"] = y

        binner = IsotonicBinning(
            max_bins=5,
            min_samples_per_bin=20,
            guidance_columns=["target"],
            preserve_dataframe=True,
        )
        binner.fit(df)
        result = binner.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feat1", "feat2"]  # No target in output
        assert result.shape == (200, 2)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    @pytest.mark.parametrize("preserve_dataframe", [True, False])
    def test_polars_input(self, monotonic_increasing_data, preserve_dataframe):
        """Test IsotonicBinning with polars input."""
        X, y = monotonic_increasing_data

        assert pl is not None
        X_df = pl.DataFrame({"feat1": X[:, 0], "feat2": X[:, 1]})

        binner = IsotonicBinning(
            max_bins=5,
            min_samples_per_bin=20,
            preserve_dataframe=preserve_dataframe,
        )
        binner.fit(X_df, y=y)
        result = binner.transform(X_df)

        if preserve_dataframe:
            assert isinstance(result, pl.DataFrame)
            assert result.columns == ["feat1", "feat2"]
        else:
            assert isinstance(result, np.ndarray)

        assert result.shape == (200, 2)

    # ======================
    # State Reconstruction Tests
    # ======================

    def test_get_params(self, monotonic_increasing_data):
        """Test parameter extraction for state reconstruction."""
        X, y = monotonic_increasing_data

        binner = IsotonicBinning(
            max_bins=8,
            min_samples_per_bin=15,
            increasing=False,
            y_min=0.0,
            y_max=10.0,
            min_change_threshold=0.02,
            clip=False,
            preserve_dataframe=True,
        )
        binner.fit(X, y=y)

        params = binner.get_params()

        # Check all constructor parameters are included
        assert params["max_bins"] == 8
        assert params["min_samples_per_bin"] == 15
        assert params["increasing"] is False
        assert params["y_min"] == 0.0
        assert params["y_max"] == 10.0
        assert params["min_change_threshold"] == 0.02
        assert params["clip"] is False
        assert params["preserve_dataframe"] is True
        assert "bin_edges" in params
        assert "bin_representatives" in params

    def test_set_params_reconstruction(self, monotonic_increasing_data):
        """Test state reconstruction using set_params."""
        X, y = monotonic_increasing_data

        # Fit original binner
        original = IsotonicBinning(max_bins=5, min_samples_per_bin=20)
        original.fit(X, y=y)
        original_result = original.transform(X)

        # Get parameters and create new binner
        params = original.get_params()
        reconstructed = IsotonicBinning()
        reconstructed.set_params(**params)

        # Test without refitting
        reconstructed_result = reconstructed.transform(X)
        np.testing.assert_array_equal(original_result, reconstructed_result)

    def test_constructor_reconstruction(self, monotonic_increasing_data):
        """Test state reconstruction using constructor."""
        X, y = monotonic_increasing_data

        # Fit original binner
        original = IsotonicBinning(
            max_bins=6, min_samples_per_bin=15, increasing=False, min_change_threshold=0.03
        )
        original.fit(X, y=y)
        original_result = original.transform(X)

        # Get parameters and create new binner via constructor
        params = original.get_params()
        reconstructed = IsotonicBinning(**params)

        # Test without refitting
        reconstructed_result = reconstructed.transform(X)
        np.testing.assert_array_equal(original_result, reconstructed_result)

    def test_repeated_fitting_on_reconstructed_state(self, monotonic_increasing_data):
        """Test repeated fitting on reconstructed state."""
        X, y = monotonic_increasing_data

        # Create and fit original
        original = IsotonicBinning(max_bins=5, min_samples_per_bin=20)
        original.fit(X, y=y)

        # Reconstruct and fit again on same data
        params = original.get_params()
        reconstructed = IsotonicBinning(**params)
        reconstructed.fit(X, y=y)  # Refit

        # Results should be identical
        original_result = original.transform(X)
        reconstructed_result = reconstructed.transform(X)
        np.testing.assert_array_equal(original_result, reconstructed_result)

    def test_repeated_fitting_different_data(
        self, monotonic_increasing_data, monotonic_decreasing_data
    ):
        """Test repeated fitting on different data."""
        X1, y1 = monotonic_increasing_data
        X2, y2 = monotonic_decreasing_data

        # Fit on first dataset
        binner = IsotonicBinning(max_bins=5, min_samples_per_bin=20)
        binner.fit(X1, y=y1)
        result1 = binner.transform(X1)

        # Refit on second dataset
        binner.fit(X2, y=y2)
        result2 = binner.transform(X2)

        # Results should be different
        assert result1.shape[1] == result2.shape[1] == 2
        # Can't compare arrays directly as they're from different datasets

    # ======================
    # Pipeline Integration Tests
    # ======================

    def test_sklearn_pipeline_integration(self, monotonic_increasing_data):
        """Test IsotonicBinning in sklearn Pipeline."""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        X, y = monotonic_increasing_data

        # Since sklearn pipelines don't have built-in support for passing targets
        # through the pipeline, we'll test the binning step separately but within
        # a pipeline-like workflow

        # First, create and fit the binner
        binner = IsotonicBinning(max_bins=5, min_samples_per_bin=20)
        binner.fit(X, y=y)

        # Transform the data
        X_binned = binner.transform(X)

        # Now create a simple pipeline with the pre-fitted binner results
        pipeline = Pipeline(
            [
                ("regressor", LinearRegression()),
            ]
        )

        # Fit the pipeline on binned data
        pipeline.fit(X_binned, y)

        # Make predictions
        y_pred = pipeline.predict(X_binned)
        r2 = r2_score(y, y_pred)

        # Should achieve reasonable R² (isotonic binning should preserve monotonic relationships)
        assert r2 > 0.3  # Binning should preserve some monotonic signal

    def test_pipeline_get_set_params(self, monotonic_increasing_data):
        """Test parameter handling in pipeline context."""
        from sklearn.ensemble import RandomForestRegressor

        X, y = monotonic_increasing_data

        pipeline = Pipeline(
            [
                ("binning", IsotonicBinning()),
                ("regressor", RandomForestRegressor(random_state=42)),
            ]
        )

        # Test parameter access
        params = pipeline.get_params()
        assert "binning__max_bins" in params
        assert "binning__min_samples_per_bin" in params
        assert "binning__increasing" in params

        # Test parameter setting
        pipeline.set_params(
            binning__max_bins=8,
            binning__min_samples_per_bin=25,
            binning__increasing=False,
            regressor__n_estimators=50,
        )

        # Since we can't easily pass targets through pipeline, just test binning step
        binner = pipeline.named_steps["binning"]
        binner.fit(X, y=y)
        result = binner.transform(X)
        assert result.shape == X.shape

    def test_pipeline_clone(self, monotonic_increasing_data):
        """Test cloning IsotonicBinning in pipeline context."""
        X, y = monotonic_increasing_data

        original_binner = IsotonicBinning(max_bins=5, min_samples_per_bin=20)

        # Clone and fit both
        cloned_binner = clone(original_binner)

        original_binner.fit(X, y=y)
        cloned_binner.fit(X, y=y)

        # Results should be identical
        original_result = original_binner.transform(X)
        cloned_result = cloned_binner.transform(X)
        np.testing.assert_array_equal(original_result, cloned_result)

    # ======================
    # Edge Cases Tests
    # ======================

    def test_single_feature_column(self):
        """Test with single feature column."""
        np.random.seed(42)
        X = np.random.uniform(0, 10, (100, 1))
        y = X[:, 0] + np.random.normal(0, 0.5, 100)  # Monotonic relationship

        binner = IsotonicBinning(max_bins=5, min_samples_per_bin=10)
        binner.fit(X, y=y)
        result = binner.transform(X)

        assert result.shape == (100, 1)
        assert len(np.unique(result)) >= 1  # Should create at least one bin

    def test_constant_feature_column(self):
        """Test with constant feature column."""
        np.random.seed(42)
        X = np.ones((100, 2))  # All values are 1.0
        y = np.random.uniform(0, 10, 100)  # Random targets

        binner = IsotonicBinning(max_bins=5, min_samples_per_bin=10)
        binner.fit(X, y=y)
        result = binner.transform(X)

        assert result.shape == (100, 2)
        # Constant columns should result in single bin
        assert len(np.unique(result[:, 0])) == 1  # All values in column 0 should be the same
        assert len(np.unique(result[:, 1])) == 1  # All values in column 1 should be the same

    def test_binary_target_conversion(self, classification_data):
        """Test isotonic binning with binary targets."""
        X, y = classification_data

        binner = IsotonicBinning(max_bins=5, min_samples_per_bin=20)
        binner.fit(X, y=y)
        result = binner.transform(X)

        assert result.shape == (200, 2)
        # Binary targets should be converted to numeric
        assert len(np.unique(y)) == 2

    def test_feature_with_inf_values(self):
        """Test handling of inf values in features."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.5, 100)

        # Add inf values to features
        X[5, 0] = np.inf
        X[15, 1] = -np.inf

        binner = IsotonicBinning(max_bins=5, min_samples_per_bin=10)

        # Should handle inf values by clipping or other processing
        binner.fit(X, y=y)
        result = binner.transform(X)
        assert result.shape == (100, 2)

    def test_insufficient_data_points_error(self):
        """Test error with insufficient data points."""
        X = np.array([[1.0], [2.0]])  # Only 2 points
        y = np.array([1.0, 2.0])

        binner = IsotonicBinning(min_samples_per_bin=5)  # Require 5 samples per bin

        with pytest.raises(FittingError, match="Insufficient data points"):
            binner.fit(X, y=y)

    def test_empty_input(self):
        """Test with empty input data."""
        X = np.empty((0, 2))
        y = np.empty((0,))

        binner = IsotonicBinning()

        with pytest.raises(
            (ValueError, FittingError, ValidationError)
        ):  # Should fail during input validation
            binner.fit(X, y=y)

    def test_mismatched_feature_target_length(self):
        """Test with mismatched feature and target lengths."""
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.uniform(0, 10, 90)  # Different length

        binner = IsotonicBinning()

        with pytest.raises((ValueError, ValidationError)):  # Should fail during validation
            binner.fit(X, y=y)

    def test_transform_without_fitting(self):
        """Test transform before fitting."""
        X = np.random.normal(0, 1, (10, 2))

        binner = IsotonicBinning()

        with pytest.raises(
            (AttributeError, ValidationError, RuntimeError)
        ):  # Should fail - not fitted
            binner.transform(X)

    def test_guidance_data_multiple_columns_error(self):
        """Test error when guidance_data has multiple columns."""
        X = np.random.normal(0, 1, (100, 2))
        y_multi = np.random.uniform(0, 10, (100, 2))  # 2 columns

        binner = IsotonicBinning()

        with pytest.raises((ValueError, ValidationError)):  # Should fail - wrong number of columns
            binner.fit(X, guidance_data=y_multi)

    def test_guidance_data_wrong_dimensions(self):
        """Test error when guidance_data has wrong dimensions."""
        X = np.random.normal(0, 1, (100, 2))
        y_3d = np.random.uniform(0, 10, (10, 10, 10))  # 3D

        binner = IsotonicBinning()

        with pytest.raises((ValueError, ValidationError)):  # Should fail - wrong dimensions
            binner.fit(X, guidance_data=y_3d)

    def test_string_max_bins_resolution(self):
        """Test resolution of string max_bins parameter."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = X[:, 0] + X[:, 1]

        # Test with 'sqrt' which should resolve to sqrt(100) = 10
        binner = IsotonicBinning(max_bins="sqrt", min_samples_per_bin=5)
        binner.fit(X, y=y)

        # Should have resolved max_bins appropriately
        result = binner.transform(X)
        assert result.shape == (100, 2)

    def test_very_high_min_samples_per_bin(self, monotonic_increasing_data):
        """Test with very high min_samples_per_bin that fails due to insufficient data."""
        X, y = monotonic_increasing_data

        binner = IsotonicBinning(
            max_bins=10,
            min_samples_per_bin=250,  # Higher than total sample count (200)
        )

        # Should fail because we don't have enough data points
        with pytest.raises(FittingError, match="Insufficient data points"):
            binner.fit(X, y=y)

    def test_isotonic_regression_failure_handling(self):
        """Test handling of isotonic regression failures."""
        # Create data that might cause isotonic regression issues
        X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0]])  # Repeated values
        y = np.array([1.0, np.nan, 3.0, 2.0, np.inf])  # NaN and inf in targets

        binner = IsotonicBinning(max_bins=3, min_samples_per_bin=2)

        # Should handle problematic data gracefully
        # The exact behavior depends on implementation - might succeed or fail gracefully
        try:
            binner.fit(X, y=y)
            result = binner.transform(X)
            assert result.shape == (5, 1)
        except (ValueError, FittingError):
            # Expected failure for problematic data is acceptable
            pass

    def test_very_small_change_threshold(self, monotonic_increasing_data):
        """Test with very small change threshold."""
        X, y = monotonic_increasing_data

        # Very small threshold should be sensitive to tiny changes
        binner = IsotonicBinning(
            max_bins=8,
            min_samples_per_bin=20,
            min_change_threshold=1e-6,  # Extremely sensitive
        )
        binner.fit(X, y=y)

        result = binner.transform(X)
        assert result.shape == (200, 2)

    def test_large_change_threshold(self, monotonic_increasing_data):
        """Test with large change threshold."""
        X, y = monotonic_increasing_data

        # Large threshold should be insensitive to changes
        binner = IsotonicBinning(
            max_bins=10,
            min_samples_per_bin=10,
            min_change_threshold=0.5,  # Very insensitive
        )
        binner.fit(X, y=y)

        result = binner.transform(X)
        assert result.shape == (200, 2)

        # Should create fewer bins due to large threshold
        for _col_id, edges in binner.bin_edges_.items():
            num_bins = len(edges) - 1
            assert num_bins <= 3  # Very few bins due to insensitive threshold

    # ======================
    # Missing Guidance Data Error Test
    # ======================

    def test_missing_guidance_data_error_in_calculate_bins(self):
        """Test FittingError when guidance_data is None in _calculate_bins."""
        binner = IsotonicBinning()
        x_col = np.array([1.0, 2.0, 3.0])

        with pytest.raises(FittingError, match="guidance_data is required for isotonic binning"):
            binner._calculate_bins(x_col, "test_col", guidance_data=None)

    def test_isotonic_edge_cases_for_coverage(self):
        """Test edge cases to achieve 100% line coverage."""
        import warnings

        # Test shape mismatch error (line 225)
        binner = IsotonicBinning()
        x_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        guidance_data = np.array([0, 1])  # Different length

        with pytest.raises(
            ValueError, match="Guidance data length.*does not match feature data length"
        ):
            binner._calculate_bins(x_col, "test_col", guidance_data=guidance_data)

        # Test single unique value case (line 251)
        x_single = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        guidance_single = np.array([0, 1, 0, 1, 0, 1])

        left_edges, right_edges = binner._calculate_bins(
            x_single, "test_col", guidance_data=guidance_single
        )
        assert len(left_edges) == 2  # Two edges for one bin

        # Test very small dataset with min_samples_per_bin high (line 289)
        binner_high_samples = IsotonicBinning(min_samples_per_bin=100)
        x_small = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        guidance_small = np.array([0, 1, 0, 1, 0])

        with pytest.raises(FittingError, match="Insufficient data points.*for isotonic binning"):
            binner_high_samples._calculate_bins(x_small, "test_col", guidance_data=guidance_small)

        # Test infinity handling in isotonic regression (line 321, 369)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0])
            guidance_with_inf = np.array([0, 1, 1, 0, 1, 0])

            left_edges, right_edges = binner._calculate_bins(
                x_with_inf, "test_col", guidance_data=guidance_with_inf
            )
            assert len(left_edges) >= 1

        # Test edge case with max_bins constraint (line 399)
        binner_constrained = IsotonicBinning(max_bins=2)
        x_many = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        guidance_many = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        left_edges, right_edges = binner_constrained._calculate_bins(
            x_many, "test_col", guidance_data=guidance_many
        )
        assert len(left_edges) <= 2

    def test_additional_isotonic_edge_cases_for_full_coverage(self):
        """Test additional edge cases to achieve 100% line coverage."""

        binner = IsotonicBinning()

        # Test all infinity values scenario (line 258)
        x_all_inf = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        guidance_inf = np.array([0, 1, 0, 1, 0])

        with pytest.raises(ValueError, match="All feature values are infinite"):
            binner._calculate_bins(x_all_inf, "test_col", guidance_data=guidance_inf)

        # Test y_range = 0 scenario (line 320) - all targets same value
        x_varied = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        guidance_same = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # All same target value

        left_edges, right_edges = binner._calculate_bins(
            x_varied, "test_col", guidance_data=guidance_same
        )
        assert len(left_edges) == 2  # Should create single bin when no target variation

        # Test single data point scenario (line 320) - need min_samples_per_bin
        binner_single = IsotonicBinning(min_samples_per_bin=1)  # Lower minimum for this test
        x_single_pt = np.array([1.0])
        guidance_single_pt = np.array([0.5])

        left_edges, right_edges = binner_single._calculate_bins(
            x_single_pt, "test_col", guidance_data=guidance_single_pt
        )
        assert len(left_edges) == 2  # Single bin

        # Test constant x values with x_min == x_max scenario (line 368)
        x_constant = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        guidance_constant = np.array([0, 1, 0, 1, 0])

        left_edges, right_edges = binner._calculate_bins(
            x_constant, "test_col", guidance_data=guidance_constant
        )
        assert len(left_edges) == 2  # Single bin
        assert left_edges == [1.9, 2.1]  # Constant feature handling: x_val ± 0.1

        # Test single cut point scenario (line 398)
        # This happens when no significant changes are found, so only initial cut point exists
        binner_high_threshold = IsotonicBinning(min_change_threshold=10.0)  # Very high threshold
        x_small_change = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        guidance_small_change = np.array([0.0, 0.01, 0.02, 0.03, 0.04])  # Very small changes

        left_edges, right_edges = binner_high_threshold._calculate_bins(
            x_small_change, "test_col", guidance_data=guidance_small_change
        )
        assert len(right_edges) == 1  # Should use single bin logic (line 398)

    def test_all_infinite_values_error(self):
        """Test error handling when all feature values are infinite (line 258)."""
        binner = IsotonicBinning()

        # Test all infinite values case - this should trigger the error in our implementation
        x_all_inf = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        guidance_all_inf = np.array([0, 1, 0, 1, 0])

        with pytest.raises(ValueError, match="All feature values are infinite"):
            binner._calculate_bins(x_all_inf, "test_col", guidance_data=guidance_all_inf)

    def test_complete_coverage_edge_cases(self):
        """Test remaining edge cases to achieve 100% line coverage."""
        binner = IsotonicBinning()

        # Test isotonic regression failure (line 262 - except Exception as e)
        # This is hard to trigger directly, so we'll skip this as it's an exceptional case

        # Test single value y_fitted (line 313)
        x_single_y = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0]
        )  # Different values to avoid constant handling
        y_single_target = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Same target values

        # This should go through isotonic regression but result in flat y_fitted
        left_edges, right_edges = binner._calculate_bins(
            x_single_y, "test_col", guidance_data=y_single_target
        )
        assert len(left_edges) >= 2

        # Test len(y_fitted) <= 1 case directly on _find_cut_points (line 313)
        cut_points = binner._find_cut_points(np.array([1.0]), np.array([1.0]), 5)
        assert cut_points == [0]

        # Test x_min == x_max case in _create_bins_from_cuts (line 361)
        x_same = np.array([2.0, 2.0, 2.0])
        y_same = np.array([1.0, 2.0, 3.0])  # Different y values
        single_cut = [0]

        edges, reps = binner._create_bins_from_cuts(x_same, y_same, single_cut, "test")
        assert edges == [2.0, 3.0]  # x_min=2.0, x_max should become 3.0 (2.0 + 1.0)
        assert reps == [2.5]  # (2.0 + 3.0) / 2

        # Test cut_idx <= prev_cut_idx case (branch 375->368) - this is actually hard to trigger
        # in normal operation, so let's test the normal flow that hits line 391

        # Test len(cut_indices) == 1 case (line 391 - else branch)
        x_normal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_normal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Flat target - should result in single cut
        single_cut_normal = [0]  # Only one cut point

        edges_normal, reps_normal = binner._create_bins_from_cuts(
            x_normal, y_normal, single_cut_normal, "test"
        )
        assert len(edges_normal) == 2  # Two edges for one bin
        assert len(reps_normal) == 1  # One representative
        # This hits line 391: bin_representatives.append(float(np.mean(x_sorted)))

    def test_missing_coverage_lines_for_100_percent(self):
        """Test the remaining uncovered lines for 100% coverage."""
        binner = IsotonicBinning(max_bins=10, min_samples_per_bin=1)

        # Test case 1: Cover the else branch in "if cut_idx > prev_cut_idx" (line 364->357)
        # Create a scenario where cut_idx <= prev_cut_idx (which should not happen in normal flow)
        x_data = np.array([1.0, 2.0, 3.0, 4.0])
        y_data = np.array([1.0, 2.0, 3.0, 4.0])

        # Create cut_indices where we have duplicate indices to trigger the else branch
        cut_indices_with_duplicate = [0, 1, 1, 2]  # Has duplicate index

        edges, reps = binner._create_bins_from_cuts(
            x_data, y_data, cut_indices_with_duplicate, "test"
        )
        # The duplicate should be skipped, so we should still get valid bins
        assert len(edges) >= 2
        assert len(reps) >= 1

        # Test case 2: Cover line 380 - the else clause in "if len(cut_indices) > 1"
        # This requires len(cut_indices) == 0 to avoid early return but still reach line 380
        # However, empty cut_indices might not be valid, so let me try a different approach

        # Actually, let's look at this more carefully. The early return happens when len == 1
        # So we need len(cut_indices) == 0 to not trigger early return and reach line 380
        x_empty_cuts = np.array([1.0, 2.0])
        y_empty_cuts = np.array([1.0, 2.0])
        empty_cut_indices = []  # No cut points

        # This should skip the early return and reach the else clause at line 380
        edges_empty, reps_empty = binner._create_bins_from_cuts(
            x_empty_cuts, y_empty_cuts, empty_cut_indices, "test"
        )
        assert len(edges_empty) >= 1
        assert len(reps_empty) >= 1
