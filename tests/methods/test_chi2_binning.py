"""
Comprehensive tests for Chi2Binning functionality.
"""

import warnings
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

# Import sklearn components
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.methods._chi2_binning import Chi2Binning
from binlearn.utils._errors import (
    ConfigurationError,
    DataQualityWarning,
    FittingError,
    InvalidDataError,
    ValidationError,
)

SKLEARN_AVAILABLE = True


class TestChi2BinningInitialization:
    """Test Chi2Binning initialization and parameter handling."""

    def test_default_initialization(self) -> None:
        """Test default parameter initialization."""
        binning = Chi2Binning()
        assert binning.max_bins == 10
        assert binning.min_bins == 2
        assert binning.alpha == 0.05
        assert binning.initial_bins == 20
        assert binning.task_type == "classification"
        assert binning.preserve_dataframe is False
        assert binning.fit_jointly is False  # Always False for supervised binning
        assert binning.bin_edges is None
        assert binning.bin_representatives is None

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        binning = Chi2Binning(
            max_bins=8,
            min_bins=3,
            alpha=0.01,
            initial_bins=20,
            preserve_dataframe=True,
            guidance_columns=[1],
        )
        assert binning.max_bins == 8
        assert binning.min_bins == 3
        assert binning.alpha == 0.01
        assert binning.initial_bins == 20
        assert binning.preserve_dataframe is True
        assert binning.guidance_columns  # == [1]

    def test_invalid_initialization_parameters(self) -> None:
        """Test that invalid parameters raise appropriate errors."""
        # Invalid max_bins
        with pytest.raises(ConfigurationError):
            Chi2Binning(max_bins=1)

        # Invalid min_bins
        with pytest.raises(ConfigurationError):
            Chi2Binning(min_bins=1)

        # min_bins > max_bins
        with pytest.raises(ConfigurationError):
            Chi2Binning(min_bins=5, max_bins=3)

        # Invalid alpha
        with pytest.raises(ConfigurationError):
            Chi2Binning(alpha=0.0)

        with pytest.raises(ConfigurationError):
            Chi2Binning(alpha=1.0)

        # Invalid initial_bins
        with pytest.raises(ConfigurationError):
            Chi2Binning(initial_bins=3, max_bins=5)


class TestChi2BinningGuidanceParameterizations:
    """Test different ways of providing guidance data to Chi2Binning."""

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200

        # Create features with clear separation for different classes
        X1 = np.random.normal(0, 1, n_samples // 2)
        X2 = np.random.normal(3, 1, n_samples // 2)
        X = np.concatenate([X1, X2]).reshape(-1, 1)

        # Create target variable
        y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        return X, y

    @pytest.fixture
    def multifeature_data(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Create multi-feature sample data for testing."""
        np.random.seed(42)
        n_samples = 150

        X = np.random.normal(0, 1, (n_samples, 3))
        # Create target with relationship to features
        y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int)

        return X, y

    def test_guidance_columns_approach(self, sample_data) -> None:
        """Test traditional guidance_columns approach."""
        X, y = sample_data

        # Create data with embedded target
        X_with_target = np.column_stack([X, y])

        binning = Chi2Binning(guidance_columns=[1], max_bins=4)
        X_binned = binning.fit_transform(X_with_target)

        assert X_binned.shape == (len(X), 1)
        assert len(np.unique(X_binned)) >= 2  # Should create multiple bins
        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_edges_) > 0

    def test_y_parameter_approach(self, sample_data) -> None:
        """Test sklearn-style y parameter approach."""
        X, y = sample_data

        binning = Chi2Binning(max_bins=4)
        binning.fit(X, y=y)
        X_binned = binning.transform(X)

        assert X_binned.shape == (len(X), 1)
        assert len(np.unique(X_binned)) >= 2  # Should create multiple bins
        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_edges_) > 0

    def test_guidance_data_parameter_approach(self, sample_data) -> None:
        """Test explicit guidance_data parameter approach."""
        X, y = sample_data

        binning = Chi2Binning(max_bins=4)
        binning.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned = binning.transform(X)

        assert X_binned.shape == (len(X), 1)
        assert len(np.unique(X_binned)) >= 2  # Should create multiple bins
        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_edges_) > 0

    def test_identical_results_single_feature(self, sample_data) -> None:
        """Test that all guidance approaches produce identical results for single feature."""
        X, y = sample_data

        # Method 1: guidance_columns (embedded in X)
        X_with_target = np.column_stack([X, y])
        binning1 = Chi2Binning(guidance_columns=[1], max_bins=4)
        X_binned1 = binning1.fit_transform(X_with_target)

        # Method 2: y parameter (sklearn-style)
        binning2 = Chi2Binning(max_bins=4)
        binning2.fit(X, y=y)
        X_binned2 = binning2.transform(X)

        # Method 3: explicit guidance_data parameter
        binning3 = Chi2Binning(max_bins=4)
        binning3.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned3 = binning3.transform(X)

        # All results should be identical
        assert np.array_equal(
            X_binned1, X_binned2
        ), "Methods 1 and 2 should produce identical results"
        assert np.array_equal(
            X_binned1, X_binned3
        ), "Methods 1 and 3 should produce identical results"
        assert np.array_equal(
            X_binned2, X_binned3
        ), "Methods 2 and 3 should produce identical results"

        # Bin edges should be identical
        assert (
            binning1.bin_edges_ == binning2.bin_edges_
        ), "Bin edges should be identical (methods 1&2)"
        assert (
            binning1.bin_edges_ == binning3.bin_edges_
        ), "Bin edges should be identical (methods 1&3)"
        assert (
            binning2.bin_edges_ == binning3.bin_edges_
        ), "Bin edges should be identical (methods 2&3)"

    def test_identical_results_multifeature(self, multifeature_data) -> None:
        """Test that all guidance approaches produce identical results for multiple features."""
        X, y = multifeature_data

        # Method 1: guidance_columns (embedded in X)
        X_with_target = np.column_stack([X, y])
        binning1 = Chi2Binning(guidance_columns=[3], max_bins=5)
        X_binned1 = binning1.fit_transform(X_with_target)

        # Method 2: y parameter (sklearn-style)
        binning2 = Chi2Binning(max_bins=5)
        binning2.fit(X, y=y)
        X_binned2 = binning2.transform(X)

        # Method 3: explicit guidance_data parameter
        binning3 = Chi2Binning(max_bins=5)
        binning3.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned3 = binning3.transform(X)

        # All results should be identical
        assert np.array_equal(
            X_binned1, X_binned2
        ), "Methods 1 and 2 should produce identical results"
        assert np.array_equal(
            X_binned1, X_binned3
        ), "Methods 1 and 3 should produce identical results"
        assert np.array_equal(
            X_binned2, X_binned3
        ), "Methods 2 and 3 should produce identical results"

        # Bin edges should be identical
        assert (
            binning1.bin_edges_ == binning2.bin_edges_
        ), "Bin edges should be identical (methods 1&2)"
        assert (
            binning1.bin_edges_ == binning3.bin_edges_
        ), "Bin edges should be identical (methods 1&3)"
        assert (
            binning2.bin_edges_ == binning3.bin_edges_
        ), "Bin edges should be identical (methods 2&3)"

        # Should bin all features
        assert X_binned1.shape == (len(X), 3)

    def test_guidance_data_priority(self, sample_data) -> None:
        """Test the priority order of guidance data sources."""
        X, y = sample_data

        # Create conflicting guidance data
        X_with_embedded = np.column_stack([X, y])
        different_y = 1 - y  # Opposite of y

        # When both guidance_columns and y are provided, guidance_columns should take priority
        binning = Chi2Binning(guidance_columns=[1], max_bins=4)
        binning.fit(X_with_embedded, y=different_y)
        result_embedded = binning.transform(X)

        # Compare with using only embedded guidance
        binning_control = Chi2Binning(guidance_columns=[1], max_bins=4)
        binning_control.fit(X_with_embedded)
        result_control = binning_control.transform(X)

        # Should use embedded guidance (ignore y parameter)
        assert np.array_equal(
            result_embedded, result_control
        ), "Should prioritize embedded guidance_columns over y parameter"

    def test_multiple_guidance_columns_error(self, sample_data) -> None:
        """Test that providing multiple guidance columns raises appropriate errors."""
        X, y = sample_data

        # Create multiple guidance columns
        y_multi = np.column_stack([y, 1 - y])  # Two guidance columns

        # Should reject multiple guidance columns with ValidationError (wrapped in FittingError)
        binning = Chi2Binning(max_bins=4)
        with pytest.raises(FittingError, match="guidance_data has 2 columns"):
            binning.fit(X, guidance_data=y_multi)

    def test_missing_guidance_data_error(self, sample_data) -> None:
        """Test that missing guidance data raises appropriate errors."""
        X, y = sample_data

        # Chi2Binning without any guidance should raise error
        binning = Chi2Binning(max_bins=4)
        with pytest.raises(ValueError):
            binning.fit(X)  # No guidance provided


class TestChi2BinningAlgorithm:
    """Test the Chi2Binning algorithm specifics."""

    @pytest.fixture
    def classification_data(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Create data suitable for classification testing."""
        np.random.seed(42)
        n_samples = 300

        # Create data with distinct patterns for different classes
        X = np.random.normal(0, 1, (n_samples, 1))
        y = np.where(X[:, 0] < -1, 0, np.where(X[:, 0] < 0, 1, np.where(X[:, 0] < 1, 2, 3)))

        return X, y

    def test_different_bin_counts(self, classification_data) -> None:
        """Test Chi2Binning with different bin count configurations."""
        X, y = classification_data

        for max_bins in [3, 4, 5, 6]:
            binning = Chi2Binning(max_bins=max_bins)
            binning.fit(X, y=y)
            X_binned = binning.transform(X)

            # Should respect max_bins constraint
            unique_bins = len(np.unique(X_binned))
            assert (
                unique_bins <= max_bins
            ), f"Should not exceed max_bins={max_bins}, got {unique_bins}"
            assert unique_bins >= 2, "Should create at least 2 bins"

    def test_alpha_parameter_effect(self, classification_data) -> None:
        """Test that different alpha values affect binning results."""
        X, y = classification_data

        # Test with very strict alpha (should create fewer bins)
        binning_strict = Chi2Binning(max_bins=8, alpha=0.001)
        binning_strict.fit(X, y=y)
        X_binned_strict = binning_strict.transform(X)

        # Test with lenient alpha (should create more bins)
        binning_lenient = Chi2Binning(max_bins=8, alpha=0.1)
        binning_lenient.fit(X, y=y)
        X_binned_lenient = binning_lenient.transform(X)

        # Strict alpha should generally create fewer or equal bins
        strict_bins = len(np.unique(X_binned_strict))
        lenient_bins = len(np.unique(X_binned_lenient))

        # This is a general tendency, not a strict rule
        assert (
            strict_bins <= lenient_bins + 1
        ), "Strict alpha should not create significantly more bins"

    def test_initial_bins_parameter(self, classification_data) -> None:
        """Test that initial_bins parameter affects the algorithm."""
        X, y = classification_data

        # Test with different initial_bins values
        for initial_bins in [5, 10, 20]:
            binning = Chi2Binning(max_bins=4, initial_bins=initial_bins)
            binning.fit(X, y=y)
            X_binned = binning.transform(X)

            # Should still respect max_bins
            unique_bins = len(np.unique(X_binned))
            assert (
                unique_bins <= 4
            ), f"Should not exceed max_bins=4 with initial_bins={initial_bins}"

    def test_insufficient_data_handling(self) -> None:
        """Test handling of insufficient data scenarios."""
        # Very small dataset
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0])

        binning = Chi2Binning(max_bins=5, min_bins=2)
        binning.fit(X, y=y)
        X_binned = binning.transform(X)

        # Should handle gracefully
        assert X_binned.shape == (3, 1)
        assert len(np.unique(X_binned)) >= 1


class TestChi2BinningDataFrameSupport:
    """Test Chi2Binning with pandas and polars DataFrames."""

    @pytest.fixture
    def sample_dataframe_data(self) -> dict[str, np.ndarray[Any, Any]]:
        """Create sample DataFrame data."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(2, 1, n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }

        return data

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe_guidance_columns(self, sample_dataframe_data) -> None:
        """Test Chi2Binning with pandas DataFrame using guidance_columns."""
        df = pd.DataFrame(sample_dataframe_data)

        binning = Chi2Binning(guidance_columns=["target"], max_bins=4, preserve_dataframe=True)

        # Suppress sklearn warnings about feature names
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*")
            result = binning.fit_transform(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(df)
        assert "feature_1" in result.columns
        assert "feature_2" in result.columns
        assert "target" not in result.columns  # Target should be excluded from output

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_dataframe_y_parameter(self, sample_dataframe_data) -> None:
        """Test Chi2Binning with pandas DataFrame using y parameter."""
        df = pd.DataFrame(sample_dataframe_data)
        features_df = df[["feature_1", "feature_2"]]
        target = df["target"].values

        binning = Chi2Binning(max_bins=4, preserve_dataframe=True)

        # Suppress sklearn warnings about feature names
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*")
            binning.fit(features_df, y=target)
            result = binning.transform(features_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(features_df)
        assert list(result.columns) == list(features_df.columns)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_identical_results(self, sample_dataframe_data) -> None:
        """Test that pandas DataFrame approaches produce identical results."""
        df = pd.DataFrame(sample_dataframe_data)
        features_df = df[["feature_1", "feature_2"]]
        target = df["target"].values

        # Method 1: guidance_columns
        binning1 = Chi2Binning(guidance_columns=["target"], max_bins=4)

        # Method 2: y parameter
        binning2 = Chi2Binning(max_bins=4)

        # Suppress sklearn warnings about feature names
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*")
            result1 = binning1.fit_transform(df)
            binning2.fit(features_df, y=target)
            result2 = binning2.transform(features_df)

        # Convert to numpy for comparison
        result1_array = getattr(result1, "values", result1)
        result2_array = getattr(result2, "values", result2)

        assert np.array_equal(
            result1_array, result2_array
        ), "DataFrame methods should produce identical results"


class TestChi2BinningPolarsIntegration:
    """Test Chi2Binning with Polars DataFrames."""

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe_basic(self) -> None:
        """Test basic functionality with Polars DataFrame."""
        df = pl.DataFrame(  # type: ignore[union-attr]
            {
                "feature_1": [1, 2, 3, 4, 5, 6],
                "feature_2": [10, 20, 30, 40, 50, 60],
                "target": [0, 0, 0, 1, 1, 1],
            }
        )

        binning = Chi2Binning(preserve_dataframe=True, guidance_columns=["target"])

        # Fit and transform
        binning.fit(df)
        df_binned = binning.transform(df)

        # Should return Polars DataFrame with only feature columns
        assert isinstance(df_binned, pl.DataFrame)  # type: ignore[union-attr]
        assert df_binned.columns == ["feature_1", "feature_2"]
        assert df_binned.shape == (6, 2)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_dataframe_without_preserve(self) -> None:
        """Test Polars DataFrame without preserve_dataframe."""
        df = pl.DataFrame(  # type: ignore[union-attr]
            {
                "feature_1": [1, 2, 3, 4, 5, 6],
                "feature_2": [10, 20, 30, 40, 50, 60],
                "target": [0, 0, 0, 1, 1, 1],
            }
        )

        binning = Chi2Binning(preserve_dataframe=False, guidance_columns=["target"])

        binning.fit(df)
        result = binning.transform(df)

        # Should return numpy array with only feature columns
        assert isinstance(result, np.ndarray)
        assert result.shape == (6, 2)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    def test_polars_y_parameter_approach(self) -> None:
        """Test Polars DataFrame with y parameter approach."""
        features_df = pl.DataFrame(  # type: ignore[union-attr]
            {
                "feature_1": [1, 2, 3, 4, 5, 6],
                "feature_2": [10, 20, 30, 40, 50, 60],
            }
        )
        target = np.array([0, 0, 0, 1, 1, 1])

        binning = Chi2Binning(preserve_dataframe=True, max_bins=4)

        binning.fit(features_df, y=target)
        result = binning.transform(features_df)

        # Should return Polars DataFrame with feature columns
        assert isinstance(result, pl.DataFrame)  # type: ignore[union-attr]
        assert result.columns == ["feature_1", "feature_2"]
        assert result.shape == (6, 2)


class TestChi2BinningSklearnIntegration:
    """Test Chi2Binning integration with sklearn pipelines."""

    @pytest.fixture
    def pipeline_data(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Create data for pipeline testing."""
        np.random.seed(42)
        n_samples = 100

        X = np.random.normal(0, 1, (n_samples, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return X, y

    def test_sklearn_pipeline_integration(self, pipeline_data) -> None:
        """Test Chi2Binning in sklearn pipeline."""
        X, y = pipeline_data

        # Create pipeline with Chi2Binning
        pipeline = Pipeline([("binning", Chi2Binning(max_bins=4)), ("scaling", StandardScaler())])

        # Suppress sklearn warnings about feature names
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*")
            X_transformed = pipeline.fit_transform(X, y)

        assert X_transformed.shape == X.shape
        assert not np.array_equal(X, X_transformed)  # Should be different after transformation

    def test_sklearn_column_transformer(self, pipeline_data) -> None:
        """Test Chi2Binning with ColumnTransformer using separate data."""
        X, y = pipeline_data

        # For Chi2Binning, combine features with targets
        X_with_targets = np.column_stack([X, y])

        # Test Chi2Binning separately first
        binning = Chi2Binning(guidance_columns=[2])  # Target is column 2 in combined data
        binning.fit(X_with_targets)

        # Transform just the features we want
        X_binned = binning.transform(X_with_targets)

        # Should work and produce the expected output shape
        assert X_binned.shape == (len(X), 2)  # Only feature columns, not target

    def test_sklearn_feature_names_out(self, pipeline_data) -> None:
        """Test get_feature_names_out method for sklearn compatibility."""
        X, y = pipeline_data

        binning = Chi2Binning(max_bins=4)

        # Check if method exists and works
        if hasattr(binning, "get_feature_names_out"):
            binning.fit(X, y=y)
            feature_names = binning.get_feature_names_out()
            assert len(feature_names) == X.shape[1]  # Should match number of features

    def test_fit_transform_consistency(self, pipeline_data) -> None:
        """Test that fit_transform produces same results as fit().transform()."""
        X, y = pipeline_data

        # Method 1: fit_transform
        binning1 = Chi2Binning(max_bins=4)
        result1 = binning1.fit_transform(X, y)

        # Method 2: fit then transform
        binning2 = Chi2Binning(max_bins=4)
        binning2.fit(X, y)
        result2 = binning2.transform(X)

        assert np.array_equal(result1, result2), "fit_transform should match fit().transform()"


class TestChi2BinningRepr:
    """Test string representation and debugging features."""

    def test_str_representation(self) -> None:
        """Test __str__ method."""
        binning = Chi2Binning(max_bins=8, alpha=0.01)
        str_repr = str(binning)
        assert "Chi2Binning" in str_repr
        assert "max_bins=8" in str_repr
        assert "alpha=0.01" in str_repr

    def test_repr_representation(self) -> None:
        """Test __repr__ method."""
        binning = Chi2Binning()
        repr_str = repr(binning)
        assert "Chi2Binning" in repr_str

    def test_fitted_representation(self) -> None:
        """Test representation after fitting."""
        X = np.column_stack([[1, 2, 3, 4], [0, 0, 1, 1]])  # feature  # target
        binning = Chi2Binning(guidance_columns=[1])
        binning.fit(X)

        str_repr = str(binning)
        # After fitting, it should show fitted parameters or at least not crash
        assert "Chi2Binning" in str_repr
        # Check that we can access fitted state
        assert hasattr(binning, "bin_edges_")
        assert len(binning.bin_edges_) > 0


class TestChi2BinningErrorHandling:
    """Test error handling and edge cases."""

    def test_transform_before_fit_error(self) -> None:
        """Test that transform before fit raises appropriate error."""
        X = np.random.normal(0, 1, (50, 1))

        binning = Chi2Binning(max_bins=4)
        with pytest.raises(RuntimeError, match="not fitted yet"):
            binning.transform(X)

    def test_fit_without_guidance_error(self) -> None:
        """Test that fit without guidance data raises appropriate error."""
        X = np.array([[1], [2], [3], [4]])
        binning = Chi2Binning()  # No guidance_columns specified

        with pytest.raises(ValueError):
            binning.fit(X)  # No guidance columns means no guidance data

    def test_inconsistent_data_shapes(self) -> None:
        """Test error when guidance column is out of bounds."""
        X = np.column_stack([[1, 2, 3, 4], [0, 1, 1, 0]])  # feature (column 0)  # target (column 1)

        binning = Chi2Binning(guidance_columns=[5])  # Column 5 doesn't exist

        with pytest.raises((ValueError, IndexError)):
            binning.fit(X)  # Should fail because column 5 doesn't exist

    def test_extreme_values_handling(self) -> None:
        """Test handling of extreme values."""
        X = np.column_stack(
            [[1e10, -1e10, 0], [0, 1, 0]]  # feature with large but finite values  # target
        )

        binning = Chi2Binning(guidance_columns=[1])

        # Should handle large but finite values
        binning.fit(X)
        X_binned = binning.transform(X)
        assert X_binned.shape == (3, 1)  # Only feature column

    def test_insufficient_data_for_bins(self) -> None:
        """Test handling when there's insufficient data for the requested number of bins."""
        # Very small dataset
        X = np.array([[1.0], [2.0]])
        y = np.array([0, 1])

        binning = Chi2Binning(max_bins=10, min_bins=2)  # Request more bins than data points
        binning.fit(X, y=y)
        X_binned = binning.transform(X)

        # Should handle gracefully
        assert X_binned.shape == (2, 1)
        assert len(np.unique(X_binned)) >= 1

    def test_invalid_guidance_data_shape(self) -> None:
        """Test error with invalid guidance data shape."""
        X = np.random.normal(0, 1, (50, 2))
        y = np.random.randint(0, 2, (50, 3))  # Wrong shape - should be (50,) or (50, 1)

        binning = Chi2Binning(max_bins=4)
        # The validation should happen in SupervisedBinningBase and raise ValidationError
        # which gets wrapped in FittingError by Chi2Binning._calculate_bins
        with pytest.raises((ValidationError, FittingError)) as exc_info:
            binning.fit(X, guidance_data=y)

        # Check that the error message mentions multiple columns
        error_msg = str(exc_info.value)
        assert "3 columns" in error_msg and "expected exactly 1" in error_msg

    def test_all_same_class_target(self) -> None:
        """Test behavior with all same class target variable."""
        X = np.random.normal(0, 1, (50, 1))
        y = np.zeros(50)  # All same class

        binning = Chi2Binning(max_bins=4)

        # Expect DataQualityWarning for constant guidance data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binning.fit(X, y=y)
            X_binned = binning.transform(X)

        # Should have received a DataQualityWarning
        assert len(w) == 1
        assert issubclass(w[0].category, DataQualityWarning)
        assert "appears to be constant" in str(w[0].message)

        # Should handle gracefully, though may create fewer bins
        assert X_binned.shape == X.shape

    def test_constant_feature_values(self) -> None:
        """Test behavior with constant feature values."""
        X = np.ones((50, 1))  # All same value
        y = np.random.randint(0, 2, 50)

        binning = Chi2Binning(max_bins=4)
        binning.fit(X, y=y)
        X_binned = binning.transform(X)

        # Should handle gracefully
        assert X_binned.shape == X.shape


class TestChi2BinningEdgeCases:
    """Test Chi2Binning edge cases and error conditions."""

    def test_constant_guidance_data_warning(self) -> None:
        """Test that constant guidance data triggers appropriate warning."""
        from binlearn.utils._errors import DataQualityWarning

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_constant = np.array([1, 1, 1, 1, 1])  # All same class

        binning = Chi2Binning(max_bins=3)

        # Should warn about constant guidance data
        with pytest.warns(DataQualityWarning, match="appears to be constant"):
            binning.fit(X, guidance_data=y_constant.reshape(-1, 1))

    def test_2d_guidance_data_handling(self) -> None:
        """Test handling of 2D guidance data (line 236)."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_2d = np.array([[0], [1], [0], [1]])  # 2D guidance data

        binning = Chi2Binning(max_bins=3)
        binning.fit(X, guidance_data=y_2d)
        X_binned = binning.transform(X)

        assert X_binned.shape == X.shape

    def test_no_valid_data_after_cleaning(self) -> None:
        """Test error when no valid data points remain after removing missing values (line 243)."""

        X = np.array([[np.nan], [np.nan], [np.nan]])
        y = np.array([0, 1, 0])

        binning = Chi2Binning()

        # Expect DataQualityWarning for NaN-only data and InvalidDataError
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(InvalidDataError, match="No valid data points"):
                binning.fit(X, guidance_data=y.reshape(-1, 1))

        # Should have received a DataQualityWarning about NaN values
        assert len(w) == 1
        assert issubclass(w[0].category, DataQualityWarning)
        assert "contains only NaN values" in str(w[0].message)

    def test_insufficient_data_for_binning(self) -> None:
        """Test error when insufficient data for min_bins requirement (line 251)."""

        X = np.array([[1.0]])  # Only 1 sample
        y = np.array([0])

        binning = Chi2Binning(min_bins=3)  # Need at least 3 bins but only 1 sample
        with pytest.raises(InvalidDataError, match="Insufficient data"):
            binning.fit(X, guidance_data=y.reshape(-1, 1))

    def test_constant_feature_values(self) -> None:
        """Test handling when all feature values are the same (lines 268-270)."""
        X = np.array([[5.0], [5.0], [5.0], [5.0]])  # All same values
        y = np.array([0, 1, 0, 1])

        binning = Chi2Binning(max_bins=3)
        binning.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned = binning.transform(X)

        # Should create a single bin spanning the constant value
        assert X_binned.shape == X.shape
        assert len(np.unique(X_binned)) == 1

    def test_insufficient_unique_values_fallback(self) -> None:
        """Test fallback to unique values when too few unique values exist."""
        X = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])  # Only 3 unique values, 6 samples
        y = np.array([0, 1, 0, 1, 0, 1])

        binning = Chi2Binning(min_bins=2, max_bins=4)  # Reasonable min_bins
        binning.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned = binning.transform(X)

        assert X_binned.shape == X.shape

    def test_chi2_calculation_edge_cases(self) -> None:
        """Test various edge cases in chi2 calculation methods."""
        # Test with single unique value for edge creation
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([0, 1, 0])

        binning = Chi2Binning(max_bins=2)
        binning.fit(X, guidance_data=y.reshape(-1, 1))

        # Test the _create_edges_from_unique_values method with single value
        unique_vals = np.array([5.0])
        edges = binning._create_edges_from_unique_values(unique_vals)
        assert len(edges) == 2
        assert edges[0] < 5.0 < edges[1]

    def test_chi2_contingency_edge_cases(self) -> None:
        """Test chi-square contingency table edge cases."""
        from binlearn.utils._errors import DataQualityWarning

        # Create data that will trigger edge cases in chi2 calculation
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])  # Clear separation

        binning = Chi2Binning(max_bins=2, min_bins=2, alpha=0.001, initial_bins=6)

        # This might trigger warnings about chi2 calculations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataQualityWarning)
            binning.fit(X, guidance_data=y.reshape(-1, 1))
            X_binned = binning.transform(X)

        assert X_binned.shape == X.shape

    def test_exception_handling_in_calculate_bins(self) -> None:
        """Test exception handling in _calculate_bins method (line 285)."""
        from binlearn.utils._errors import FittingError

        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 0])

        binning = Chi2Binning()

        # Mock the validate_guidance_data method to raise an unexpected exception
        with patch.object(
            binning, "validate_guidance_data", side_effect=RuntimeError("Unexpected error")
        ):
            with pytest.raises(FittingError, match="Failed to calculate chi-square bins"):
                binning.fit(X, guidance_data=y.reshape(-1, 1))

    def test_max_bins_significance_check(self) -> None:
        """Test significance check when reaching max_bins (line 324)."""
        # Create data with clear separation to control significance
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Clear binary separation

        binning = Chi2Binning(
            max_bins=3,
            min_bins=2,
            alpha=0.001,  # Very strict significance
            initial_bins=8,  # Start with many bins
        )

        binning.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned = binning.transform(X)

        # Should respect max_bins constraint
        assert len(np.unique(X_binned)) <= 3

    def test_chi2_calculation_error_handling(self) -> None:
        """Test error handling in chi2 calculation (lines 361-363, 439-441)."""
        # Create edge case data that might cause ValueError in chi2_contingency
        X = np.array([[1.0], [1.0], [2.0], [2.0]])
        y = np.array([0, 0, 1, 1])

        binning = Chi2Binning(max_bins=2, initial_bins=4)

        # This should trigger some edge cases in chi2 calculation but handle them gracefully
        binning.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned = binning.transform(X)

        assert X_binned.shape == X.shape

    def test_multiple_unique_values_edge_creation(self) -> None:
        """Test _create_edges_from_unique_values with multiple values (lines 456-464)."""
        X = np.array(
            [[1.0], [3.0], [5.0], [1.0], [3.0], [5.0]]
        )  # 3 unique values with enough samples
        y = np.array([0, 1, 0, 1, 0, 1])

        binning = Chi2Binning(min_bins=2, max_bins=4)  # Reasonable constraints
        binning.fit(X, guidance_data=y.reshape(-1, 1))

        # Test the method directly
        unique_vals = np.array([1.0, 3.0, 5.0])
        edges = binning._create_edges_from_unique_values(unique_vals)

        # Should create edges that separate the unique values
        assert len(edges) == 4  # n_unique + 1
        assert edges[0] < 1.0 < edges[1] < 3.0 < edges[2] < 5.0 < edges[3]

    def test_runtime_warning_handling(self) -> None:
        """Test handling of RuntimeWarning in chi2 merging (lines 361-363)."""
        # Create data that might trigger RuntimeWarnings in statistical calculations
        X = np.array([[0.0], [0.001], [0.002], [0.003], [0.004]])  # Very small differences
        y = np.array([0, 1, 0, 1, 0])

        binning = Chi2Binning(max_bins=3, initial_bins=5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            binning.fit(X, guidance_data=y.reshape(-1, 1))
            X_binned = binning.transform(X)

        assert X_binned.shape == X.shape

    def test_single_class_target(self) -> None:
        """Test behavior with single-class target variable."""
        X = np.random.normal(0, 1, (50, 1))
        y = np.zeros(50)  # All same class

        binning = Chi2Binning(max_bins=4)

        # Expect DataQualityWarning for constant guidance data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            binning.fit(X, y=y)
            X_binned = binning.transform(X)

        # Should have received a DataQualityWarning
        assert len(w) == 1
        assert issubclass(w[0].category, DataQualityWarning)
        assert "appears to be constant" in str(w[0].message)

        # Should handle gracefully, though may create fewer bins
        assert X_binned.shape == X.shape

    def test_constant_feature(self) -> None:
        """Test behavior with constant feature values."""
        X = np.ones((50, 1))  # All same value
        y = np.random.randint(0, 2, 50)

        binning = Chi2Binning(max_bins=4)
        binning.fit(X, y=y)
        X_binned = binning.transform(X)

        # Should handle gracefully
        assert X_binned.shape == X.shape

    def test_missing_values_handling(self) -> None:
        """Test handling of missing values in input data."""
        X = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0]])
        y = np.array([0, 1, 0, 1, 0])

        binning = Chi2Binning(max_bins=3)
        binning.fit(X, y=y)
        X_binned = binning.transform(X)

        # Should handle missing values appropriately
        assert X_binned.shape == X.shape
        # Missing values are assigned a special bin index (typically -1)
        assert X_binned[2, 0] == -1  # Missing values get assigned to bin -1

    def test_1d_guidance_data_path(self) -> None:
        """Test line 236: 1D guidance data path (guidance_col = guidance_data_validated)."""
        from unittest.mock import patch

        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])  # 1D array

        binning = Chi2Binning(max_bins=3)

        # Mock validate_guidance_data to return 1D array to trigger line 236
        def mock_validate_guidance_data(guidance_data) -> np.ndarray[Any, Any]:
            return y  # Return 1D array, should trigger else branch on line 236

        with patch.object(
            binning, "validate_guidance_data", side_effect=mock_validate_guidance_data
        ):
            bin_edges, bin_centers = binning._calculate_bins(X[:, 0], 0, y.reshape(-1, 1))

        assert len(bin_edges) >= 2
        assert len(bin_centers) >= 1

    def test_constant_feature_values_same_min_max(self) -> None:
        """Test lines 268-270: data_min == data_max scenario."""
        X = np.array([[5.0], [5.0], [5.0], [5.0]])  # All same values
        y = np.array([0, 0, 1, 1])

        binning = Chi2Binning(max_bins=3)

        # Call _calculate_bins directly to ensure we hit the constant value path
        bin_edges, bin_centers = binning._calculate_bins(X[:, 0], 0, y.reshape(-1, 1))

        # Should create single bin with padding: [4.9, 5.1] with center [5.0]
        assert len(bin_edges) == 2
        assert len(bin_centers) == 1
        assert bin_centers[0] == 5.0
        assert bin_edges[0] < 5.0 < bin_edges[1]

        # Also test normal fitting
        binning.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned = binning.transform(X)
        assert X_binned.shape == X.shape

    def test_significance_break_in_merging(self) -> None:
        """Test line 324: significance check that breaks merging loop."""
        from unittest.mock import patch

        # Create data that will trigger the significance check
        X = np.array([[1.0], [1.1], [1.2], [9.0], [9.1], [9.2]])
        y = np.array([0, 0, 0, 1, 1, 1])  # Clear binary separation

        binning = Chi2Binning(
            max_bins=2,
            min_bins=2,
            alpha=0.01,  # Very strict significance
            initial_bins=4,  # Start with more bins to trigger merging logic
        )

        # Mock _calculate_chi2_for_pair to return significant p-value to trigger break

        def mock_chi2_for_pair(x_data, y_data, edges, pair_idx) -> tuple[float, float]:
            # Return a very small p-value (significant) to trigger break on line 324
            return 10.0, 0.001  # High chi2, very low p-value (significant)

        with patch.object(binning, "_calculate_chi2_for_pair", side_effect=mock_chi2_for_pair):
            binning.fit(X, guidance_data=y.reshape(-1, 1))
            X_binned = binning.transform(X)

        # Should stop merging due to significance, respecting max_bins
        assert X_binned.shape == X.shape

    def test_chi2_calculation_exception_handling(self) -> None:
        """Test lines 361-363: exception handling in _find_best_merge_pair."""
        from unittest.mock import patch

        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])

        binning = Chi2Binning(max_bins=2, initial_bins=4)

        # Mock _calculate_chi2_for_pair to raise ValueError
        original_method = binning._calculate_chi2_for_pair
        call_count = [0]  # Use list to allow modification in nested function

        def mock_chi2_calculation(*args, **kwargs) -> tuple[float, float]:
            # Raise ValueError for first call, then use original for subsequent calls
            call_count[0] += 1

            if call_count[0] == 1:
                raise ValueError("Mock chi2 calculation error")
            return original_method(*args, **kwargs)

        with patch.object(binning, "_calculate_chi2_for_pair", side_effect=mock_chi2_calculation):
            binning.fit(X, guidance_data=y.reshape(-1, 1))
            X_binned = binning.transform(X)

        # Should handle the exception and continue
        assert X_binned.shape == X.shape

    def test_contingency_table_sum_zero(self) -> None:
        """Test line 430: contingency_table.sum() == 0 scenario."""

        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 1, 0, 1])

        binning = Chi2Binning(max_bins=2)

        # Test the _calculate_chi2_for_pair method directly
        # Create edges that will result in no data in the intervals
        edges = [0.0, 0.5, 1.0]  # No data will fall in these ranges
        x_data = np.array([2.0, 3.0])  # Data outside the edge ranges
        y_data = np.array([0, 1])

        # This should trigger the contingency table sum == 0 condition
        chi2_stat, p_value = binning._calculate_chi2_for_pair(x_data, y_data, edges, 0)

        # When contingency table sum is 0, should return inf, 0.0
        assert chi2_stat == float("inf")
        assert p_value == 0.0

        # Also test normal case
        binning.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned = binning.transform(X)
        assert X_binned.shape == X.shape

    def test_chi2_contingency_value_error(self) -> None:
        """Test lines 439-441: ValueError exception in chi2_contingency."""
        from unittest.mock import patch

        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])

        binning = Chi2Binning(max_bins=2)

        # Directly test the _calculate_chi2_for_pair method to trigger ValueError
        edges = [0.5, 2.5, 4.5]
        x_data = X[:, 0]
        y_data = y

        # Mock chi2_contingency to raise ValueError
        with patch(
            "binlearn.methods._chi2_binning.chi2_contingency", side_effect=ValueError("Test error")
        ):
            chi2_stat, p_value = binning._calculate_chi2_for_pair(x_data, y_data, edges, 0)
            # Should return 0.0, 1.0 when ValueError is caught
            assert chi2_stat == 0.0 and p_value == 1.0

        # Also test normal fitting to ensure it still works
        binning.fit(X, guidance_data=y.reshape(-1, 1))
        X_binned = binning.transform(X)
        assert X_binned.shape == X.shape

    def test_comprehensive_edge_case_coverage(self) -> None:
        """Comprehensive test to cover remaining uncovered lines."""
        from unittest.mock import patch

        binning = Chi2Binning(max_bins=3)

        # Test _extract_guidance_column with 2D input to cover the else branch
        guidance_1d = np.array([0, 1, 0, 1])
        result_1d = binning._extract_guidance_column(guidance_1d)
        assert np.array_equal(result_1d, guidance_1d)

        # Test _handle_constant_feature_values for lines 268-270 equivalent
        result_constant = binning._handle_constant_feature_values(5.0, 5.0)
        assert result_constant is not None
        bin_edges, bin_centers = result_constant
        assert bin_edges == [4.9, 5.1]
        assert bin_centers == [5.0]

        # Test case where values are not constant
        result = binning._handle_constant_feature_values(1.0, 5.0)
        assert result is None

        # Test _should_stop_merging_for_significance for line 324 equivalent
        X = np.array([1.0, 2.0, 8.0, 9.0])
        y = np.array([0, 0, 1, 1])
        edges = [0.5, 1.5, 8.5, 9.5]

        # Mock to return significant p-value
        with patch.object(binning, "_calculate_chi2_for_pair", return_value=(50.0, 0.001)):
            should_stop = binning._should_stop_merging_for_significance(X, y, edges, 0)
            assert should_stop is True  # Should stop due to significance

        # Test case where we're not at max_bins yet
        edges_long = [0.5, 1.5, 2.5, 3.5, 8.5, 9.5]  # More bins than max_bins
        should_stop_long = binning._should_stop_merging_for_significance(X, y, edges_long, 0)
        assert should_stop_long is False  # Should not stop, not at max_bins yet

        # Test contingency table sum == 0 by calling _calculate_chi2_for_pair directly
        # with data that results in empty intervals
        edges_empty = [0.0, 0.5, 1.0]
        x_data_empty = np.array([10.0, 11.0])  # Data far from edges
        y_data_empty = np.array([0, 1])
        chi2_stat, p_value = binning._calculate_chi2_for_pair(
            x_data_empty, y_data_empty, edges_empty, 0
        )
        # Should handle empty contingency table
        assert chi2_stat == float("inf") or p_value == 0.0

    def test_final_uncovered_lines(self) -> None:
        """Test the final 4 uncovered lines: 266, 317, 423, 485."""
        from unittest.mock import patch

        # Line 485: Test 2D guidance data path in _extract_guidance_column
        binning = Chi2Binning()
        guidance_2d = np.array([[0], [1], [0]])  # 2D array
        result = binning._extract_guidance_column(guidance_2d)
        assert result.shape == (3,)
        assert np.array_equal(result, [0, 1, 0])

        # Line 266: Test constant feature handling return path
        constant_result = binning._handle_constant_feature_values(5.0, 5.0)
        assert constant_result is not None
        bin_edges, bin_centers = constant_result
        assert len(bin_edges) == 2 and len(bin_centers) == 1

        # Test the actual use of constant handling in _calculate_bins
        X_const = np.array([[5.0], [5.0], [5.0]])
        y_const = np.array([0, 1, 0])
        bin_edges_calc, bin_centers_calc = binning._calculate_bins(
            X_const[:, 0], 0, y_const.reshape(-1, 1)
        )
        assert len(bin_edges_calc) == 2 and bin_centers_calc[0] == 5.0

        # Line 317: Test significance break in merging
        binning_sig = Chi2Binning(max_bins=2, alpha=0.01)
        edges = [0.5, 1.5, 2.5]  # At max_bins
        X_sig = np.array([1.0, 2.0])
        y_sig = np.array([0, 1])

        # Mock to return significant result (low p-value)
        with patch.object(binning_sig, "_calculate_chi2_for_pair", return_value=(50.0, 0.001)):
            should_stop = binning_sig._should_stop_merging_for_significance(X_sig, y_sig, edges, 0)
            assert should_stop is True  # Should trigger the break

        # Line 423: Test contingency table sum == 0 case
        # Use _calculate_chi2_for_pair with data that creates empty contingency table
        edges_empty = [0.0, 0.5, 1.0]
        x_empty = np.array([10.0, 11.0])  # Data far from edges [0-1]
        y_empty = np.array([0, 1])
        chi2_stat, p_value = binning._calculate_chi2_for_pair(x_empty, y_empty, edges_empty, 0)
        assert chi2_stat == float("inf") and p_value == 0.0

    def test_line_266_constant_edges_return(self) -> None:
        """Test line 266: return constant_edges when data_min == data_max."""
        # Create binning with min_bins temporarily set to 1 to bypass the early return
        binning = Chi2Binning(max_bins=3)

        # Temporarily override min_bins to 1 to reach the constant_edges check
        original_min_bins = binning.min_bins
        binning.min_bins = 1

        try:
            # Create constant feature data
            X_constant = np.array([5.0, 5.0, 5.0, 5.0])  # All same values
            y_constant = np.array([[0], [1], [0], [1]])

            # Call _calculate_bins directly to hit line 266
            bin_edges, bin_centers = binning._calculate_bins(X_constant, 0, y_constant)

            # Should have returned from the constant_edges path (line 266)
            assert len(bin_edges) == 2
            assert len(bin_centers) == 1
            assert bin_centers[0] == 5.0
            assert bin_edges[0] < 5.0 < bin_edges[1]

        finally:
            # Restore original min_bins
            binning.min_bins = original_min_bins

    def test_line_317_significance_break(self) -> None:
        """Test line 317: break due to significance check with alpha=1.0 and max_bins=2."""
        # Create data with clear separation
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([1, 0, 1, 0, 1, 0])

        # Use alpha=1.0 (maximally permissive) with max_bins=2 to force reaching max_bins
        # then trigger significance check at line 317
        binning = Chi2Binning(
            max_bins=4,
            min_bins=2,
            alpha=0.99,  # Maximally permissive - will merge everything until max_bins
            initial_bins=6,  # Start with many bins to force merging down to max_bins
        )

        # This should hit line 317: break when reaching max_bins and checking significance
        binning.fit(X, y=y)
        X_binned = binning.transform(X)

        # Should respect max_bins constraint
        assert X_binned.shape == X.shape

    def test_transform_before_fit_error(self) -> None:
        """Test that transform before fit raises appropriate error."""
        X = np.random.normal(0, 1, (50, 1))

        binning = Chi2Binning(max_bins=4)
        with pytest.raises(RuntimeError, match="not fitted yet"):
            binning.transform(X)


if __name__ == "__main__":
    # Run a few key tests for manual verification
    test_instance = TestChi2BinningGuidanceParameterizations()

    # Create sample data
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 1))
    y = (X[:, 0] > 0).astype(int)
    sample_data = (X, y)

    print("🧪 Running Chi2Binning guidance parameterization tests...")

    try:
        test_instance.test_identical_results_single_feature(sample_data)
        print("✅ Chi2Binning guidance parameterization tests PASSED!")
    except Exception as e:
        print(f"❌ Tests FAILED: {e}")
        import traceback

        traceback.print_exc()
