"""
Comprehensive tests for TreeBinning implementation.

Tests cover all required scenarios:
- Various input/output formats (numpy, pandas, polars)
- preserve_dataframe True and False
- Fitted state reconstruction via set_params(params) and constructor(**params)
- Test repeated fitting on reconstructed state
- Test sklearn pipeline integration
- Test edge cases with nans and infs and constant columns
- Test supervised-specific functionality:
  - Target column provided through guidance_columns parameter
  - Target column provided through y argument of fit function
  - Both classification and regression task types
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from binlearn import PANDAS_AVAILABLE, POLARS_AVAILABLE, pd, pl
from binlearn.methods import TreeBinning
from binlearn.utils import ConfigurationError, FittingError, ValidationError


class TestTreeBinning:
    """Comprehensive test suite for TreeBinning."""

    @pytest.fixture
    def classification_data(self):
        """Basic classification data for testing."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        # Create binary target based on simple rule
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Basic regression data for testing."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        # Create continuous target
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, 100)
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Multiclass classification data."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (150, 2))
        # Create 3-class target
        y = np.zeros(150, dtype=int)
        y[X[:, 0] > 0.5] = 1
        y[X[:, 0] < -0.5] = 2
        return X, y

    # ======================
    # Constructor Tests
    # ======================

    def test_constructor_defaults(self):
        """Test constructor with default parameters."""
        binner = TreeBinning()
        assert binner.task_type == "classification"  # Default
        assert binner.tree_params is None
        assert binner.clip is True  # Config default
        assert binner.preserve_dataframe is False  # Config default
        assert binner.guidance_columns is None

    def test_constructor_classification_task(self):
        """Test constructor with classification task type."""
        binner = TreeBinning(task_type="classification")
        assert binner.task_type == "classification"

    def test_constructor_regression_task(self):
        """Test constructor with regression task type."""
        binner = TreeBinning(task_type="regression")
        assert binner.task_type == "regression"

    def test_constructor_invalid_task_type(self):
        """Test constructor with invalid task type."""
        with pytest.raises(ConfigurationError, match="task_type must be"):
            TreeBinning(task_type="invalid")

    def test_constructor_with_tree_params(self):
        """Test constructor with custom tree parameters."""
        tree_params = {"max_depth": 5, "min_samples_leaf": 10, "random_state": 42}
        binner = TreeBinning(tree_params=tree_params)
        assert binner.tree_params == tree_params

    def test_constructor_invalid_tree_params(self):
        """Test constructor with invalid tree parameters."""
        with pytest.raises(ConfigurationError, match="tree_params must be a dictionary"):
            TreeBinning(tree_params="invalid")  # type: ignore

    def test_constructor_with_guidance_columns(self):
        """Test constructor with guidance columns."""
        binner = TreeBinning(guidance_columns=["target"])
        assert binner.guidance_columns == ["target"]

    def test_constructor_with_preserve_dataframe(self):
        """Test constructor with preserve_dataframe option."""
        binner = TreeBinning(preserve_dataframe=True)
        assert binner.preserve_dataframe is True

    def test_constructor_with_clip(self):
        """Test constructor with clip option."""
        binner = TreeBinning(clip=True)
        assert binner.clip is True

    def test_constructor_state_reconstruction_params(self):
        """Test constructor with state reconstruction parameters."""
        bin_edges = {0: [0.0, 1.0, 2.0], 1: [0.0, 0.5, 1.0]}
        bin_representatives = {0: [0.5, 1.5], 1: [0.25, 0.75]}
        binner = TreeBinning(
            bin_edges=bin_edges,
            bin_representatives=bin_representatives,
            class_="TreeBinning",
            module_="binlearn.methods",
        )
        assert binner.bin_edges == bin_edges
        assert binner.bin_representatives == bin_representatives

    # ======================
    # Target Column Provision Tests (Core supervised functionality)
    # ======================

    def test_target_via_guidance_columns_classification(self, classification_data):
        """Test target column provided through guidance_columns parameter (classification)."""
        X, y = classification_data

        # Create data with target as guidance column
        X_with_target = np.column_stack([X, y])

        binner = TreeBinning(
            task_type="classification",
            guidance_columns=[2],  # Target is in column 2
            tree_params={"max_depth": 3, "random_state": 42},
        )
        binner.fit(X_with_target)

        # Should be fitted
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")
        assert len(binner.bin_edges_) == 2  # Two feature columns

        # Test transform - should only transform the feature columns (not target)
        result = binner.transform(X_with_target)
        assert result.shape == (100, 2)  # Only feature columns, not target
        assert np.issubdtype(result.dtype, np.integer)

    def test_target_via_guidance_columns_regression(self, regression_data):
        """Test target column provided through guidance_columns parameter (regression)."""
        X, y = regression_data

        # Create data with target as guidance column
        X_with_target = np.column_stack([X, y])

        binner = TreeBinning(
            task_type="regression",
            guidance_columns=[2],  # Target is in column 2
            tree_params={"max_depth": 3, "random_state": 42},
        )
        binner.fit(X_with_target)

        # Should be fitted
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")
        assert len(binner.bin_edges_) == 2  # Two feature columns

        # Test transform - should only transform the feature columns (not target)
        result = binner.transform(X_with_target)
        assert result.shape == (100, 2)  # Only feature columns, not target

    def test_target_via_y_parameter_classification(self, classification_data):
        """Test target column provided through y parameter (classification)."""
        X, y = classification_data

        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )
        binner.fit(X, y=y)

        # Should be fitted
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")
        assert len(binner.bin_edges_) == 2  # Two feature columns

        # Test transform
        result = binner.transform(X)
        assert result.shape == (100, 2)
        assert np.issubdtype(result.dtype, np.integer)

    def test_target_via_y_parameter_regression(self, regression_data):
        """Test target column provided through y parameter (regression)."""
        X, y = regression_data

        binner = TreeBinning(
            task_type="regression", tree_params={"max_depth": 3, "random_state": 42}
        )
        binner.fit(X, y=y)

        # Should be fitted
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")
        assert len(binner.bin_edges_) == 2  # Two feature columns

    def test_target_via_guidance_data_parameter_classification(self, classification_data):
        """Test target column provided through guidance_data parameter (classification)."""
        X, y = classification_data

        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )
        binner.fit(X, guidance_data=y.reshape(-1, 1))

        # Should be fitted
        assert hasattr(binner, "bin_edges_")
        assert hasattr(binner, "bin_representatives_")
        assert len(binner.bin_edges_) == 2  # Two feature columns

    def test_multiclass_classification(self, multiclass_data):
        """Test with multiclass classification data."""
        X, y = multiclass_data

        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 4, "random_state": 42}
        )
        binner.fit(X, y=y)

        # Should handle multiclass
        assert hasattr(binner, "bin_edges_")
        result = binner.transform(X)
        assert result.shape == (150, 2)

    def test_priority_order_guidance_columns_over_y(self, classification_data):
        """Test that guidance_columns takes priority over y parameter."""
        X, y = classification_data

        # Create different target for guidance_columns
        y_guidance = 1 - y  # Flip the labels
        X_with_target = np.column_stack([X, y_guidance])

        binner = TreeBinning(
            task_type="classification",
            guidance_columns=[2],
            tree_params={"max_depth": 2, "random_state": 42},
        )

        # Fit with both guidance_columns and y - guidance_columns should win
        binner.fit(X_with_target, y=y)

        # Create second binner using only y_guidance via y parameter
        binner2 = TreeBinning(
            task_type="classification", tree_params={"max_depth": 2, "random_state": 42}
        )
        binner2.fit(X, y=y_guidance)

        # Results should be the same (guidance_columns was used, not y)
        result1 = binner.transform(X_with_target)
        result2 = binner2.transform(X)

        np.testing.assert_array_equal(result1, result2)

    def test_no_target_provided_error(self, classification_data):
        """Test error when no target is provided for supervised binning."""
        X, _ = classification_data

        binner = TreeBinning()

        with pytest.raises(ValidationError, match="Supervised binning requires guidance data"):
            binner.fit(X)

    # ======================
    # Input Format Tests
    # ======================

    @pytest.mark.parametrize("preserve_dataframe", [True, False])
    def test_numpy_input_classification(self, classification_data, preserve_dataframe):
        """Test TreeBinning with numpy input (classification)."""
        X, y = classification_data

        binner = TreeBinning(
            task_type="classification",
            preserve_dataframe=preserve_dataframe,
            tree_params={"max_depth": 3, "random_state": 42},
        )
        binner.fit(X, y=y)
        result = binner.transform(X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 2)
        assert np.issubdtype(result.dtype, np.integer)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    @pytest.mark.parametrize("preserve_dataframe", [True, False])
    def test_pandas_input_classification(self, classification_data, preserve_dataframe):
        """Test TreeBinning with pandas input (classification)."""
        X, y = classification_data

        X_df = pd.DataFrame(X, columns=["feat1", "feat2"])
        y_series = pd.Series(y, name="target")

        binner = TreeBinning(
            task_type="classification",
            preserve_dataframe=preserve_dataframe,
            tree_params={"max_depth": 3, "random_state": 42},
        )
        binner.fit(X_df, y=y_series)
        result = binner.transform(X_df)

        if preserve_dataframe:
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["feat1", "feat2"]
        else:
            assert isinstance(result, np.ndarray)

        assert result.shape == (100, 2)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_with_guidance_columns(self, classification_data):
        """Test pandas input with guidance columns."""
        X, y = classification_data

        # Create DataFrame with target column
        df = pd.DataFrame(X, columns=["feat1", "feat2"])
        df["target"] = y

        binner = TreeBinning(
            task_type="classification",
            guidance_columns=["target"],
            preserve_dataframe=True,
            tree_params={"max_depth": 3, "random_state": 42},
        )
        binner.fit(df)
        result = binner.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feat1", "feat2"]  # No target in output
        assert result.shape == (100, 2)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars not available")
    @pytest.mark.parametrize("preserve_dataframe", [True, False])
    def test_polars_input_classification(self, classification_data, preserve_dataframe):
        """Test TreeBinning with polars input (classification)."""
        X, y = classification_data

        assert pl is not None
        X_df = pl.DataFrame({"feat1": X[:, 0], "feat2": X[:, 1]})

        binner = TreeBinning(
            task_type="classification",
            preserve_dataframe=preserve_dataframe,
            tree_params={"max_depth": 3, "random_state": 42},
        )
        binner.fit(X_df, y=y)
        result = binner.transform(X_df)

        if preserve_dataframe:
            assert isinstance(result, pl.DataFrame)
            assert result.columns == ["feat1", "feat2"]
        else:
            assert isinstance(result, np.ndarray)

        assert result.shape == (100, 2)

    # ======================
    # State Reconstruction Tests
    # ======================

    def test_get_params(self, classification_data):
        """Test parameter extraction for state reconstruction."""
        X, y = classification_data

        tree_params = {"max_depth": 4, "min_samples_leaf": 5, "random_state": 42}
        binner = TreeBinning(
            task_type="regression", tree_params=tree_params, clip=True, preserve_dataframe=True
        )
        binner.fit(X, y=y)

        params = binner.get_params()

        # Check all constructor parameters are included
        assert params["task_type"] == "regression"
        assert params["tree_params"] == tree_params
        assert params["clip"] is True
        assert params["preserve_dataframe"] is True
        assert "bin_edges" in params
        assert "bin_representatives" in params

    def test_set_params_reconstruction(self, classification_data):
        """Test state reconstruction using set_params."""
        X, y = classification_data

        # Fit original binner
        original = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )
        original.fit(X, y=y)
        original_result = original.transform(X)

        # Get parameters and create new binner
        params = original.get_params()
        reconstructed = TreeBinning()
        reconstructed.set_params(**params)

        # Test without refitting
        reconstructed_result = reconstructed.transform(X)
        np.testing.assert_array_equal(original_result, reconstructed_result)

    def test_constructor_reconstruction(self, regression_data):
        """Test state reconstruction using constructor."""
        X, y = regression_data

        # Fit original binner
        original = TreeBinning(
            task_type="regression", tree_params={"max_depth": 2, "random_state": 42}
        )
        original.fit(X, y=y)
        original_result = original.transform(X)

        # Get parameters and create new binner via constructor
        params = original.get_params()
        reconstructed = TreeBinning(**params)

        # Test without refitting
        reconstructed_result = reconstructed.transform(X)
        np.testing.assert_array_equal(original_result, reconstructed_result)

    def test_repeated_fitting_on_reconstructed_state(self, classification_data):
        """Test repeated fitting on reconstructed state."""
        X, y = classification_data

        # Create and fit original
        original = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )
        original.fit(X, y=y)

        # Reconstruct and fit again on same data
        params = original.get_params()
        reconstructed = TreeBinning(**params)
        reconstructed.fit(X, y=y)  # Refit

        # Results should be identical (same random state)
        original_result = original.transform(X)
        reconstructed_result = reconstructed.transform(X)
        np.testing.assert_array_equal(original_result, reconstructed_result)

    def test_repeated_fitting_different_data(self, classification_data, regression_data):
        """Test repeated fitting on different data."""
        X1, y1 = classification_data
        X2, y2 = regression_data

        # Fit on first dataset
        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )
        binner.fit(X1, y=y1)
        result1 = binner.transform(X1)

        # Refit on second dataset (convert to classification)
        y2_class = (y2 > np.median(y2)).astype(int)
        binner.fit(X2, y=y2_class)
        result2 = binner.transform(X2)

        # Results should be different
        assert result1.shape[1] == result2.shape[1] == 2
        # Can't compare arrays directly as they're from different datasets

    # ======================
    # Pipeline Integration Tests
    # ======================

    def test_sklearn_pipeline_integration(self, classification_data):
        """Test TreeBinning in sklearn Pipeline."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        X, y = classification_data

        # Create pipeline
        pipeline = Pipeline(
            [
                (
                    "binning",
                    TreeBinning(
                        task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
                    ),
                ),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )

        # Fit pipeline
        pipeline.fit(X, y)

        # Make predictions
        y_pred = pipeline.predict(X)
        accuracy = accuracy_score(y, y_pred)

        # Should achieve reasonable accuracy
        assert accuracy > 0.6  # Tree binning should preserve some signal

    def test_pipeline_get_set_params(self, classification_data):
        """Test parameter handling in pipeline context."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = classification_data

        pipeline = Pipeline(
            [
                ("binning", TreeBinning(task_type="classification")),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        )

        # Test parameter access
        params = pipeline.get_params()
        assert "binning__task_type" in params
        assert "binning__tree_params" in params

        # Test parameter setting
        pipeline.set_params(binning__tree_params={"max_depth": 5}, classifier__n_estimators=50)

        # Fit and test
        pipeline.fit(X, y)
        result = pipeline.predict(X)
        assert len(result) == len(y)

    def test_pipeline_clone(self, classification_data):
        """Test cloning TreeBinning in pipeline context."""
        X, y = classification_data

        original_binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )

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
        X = np.random.normal(0, 1, (100, 1))
        y = (X[:, 0] > 0).astype(int)

        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )
        binner.fit(X, y=y)
        result = binner.transform(X)

        assert result.shape == (100, 1)
        assert len(np.unique(result)) > 1  # Should create multiple bins

    def test_constant_feature_column(self):
        """Test with constant feature column."""
        np.random.seed(42)
        X = np.ones((100, 2))  # All values are 1.0
        y = np.random.randint(0, 2, 100)  # Random target

        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )
        binner.fit(X, y=y)
        result = binner.transform(X)

        assert result.shape == (100, 2)
        # Constant columns may result in default bins (could be -1 or 0 depending on implementation)
        # The key is that all values should be the same for each column
        assert len(np.unique(result[:, 0])) == 1  # All values in column 0 should be the same
        assert len(np.unique(result[:, 1])) == 1  # All values in column 1 should be the same

    def test_target_with_missing_values(self):
        """Test handling of missing values in target."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 2, 100).astype(float)

        # Add NaN values to target
        y[::10] = np.nan

        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )

        # Should handle NaN in target by excluding those rows
        from binlearn.utils import DataQualityWarning

        with pytest.warns(DataQualityWarning):  # Should warn about missing data
            binner.fit(X, y=y)

        result = binner.transform(X)
        assert result.shape == (100, 2)

    def test_feature_with_inf_values(self):
        """Test handling of inf values in features."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Add inf values to features
        X[5, 0] = np.inf
        X[15, 1] = -np.inf

        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )

        # Should handle inf values - either by clipping, error, or other processing
        # Based on the error, sklearn can't handle inf values
        with pytest.raises(FittingError, match="Input.*infinity"):
            binner.fit(X, y=y)

    def test_insufficient_data_points(self):
        """Test with insufficient data points for tree fitting."""
        X = np.array([[1.0], [2.0]])  # Only 2 points
        y = np.array([0, 1])

        binner = TreeBinning(
            task_type="classification", tree_params={"min_samples_split": 5}  # Requires 5 points
        )

        with pytest.raises(FittingError, match="Insufficient data points"):
            binner.fit(X, y=y)

    def test_invalid_task_type_for_target(self):
        """Test classification task with continuous target values."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.normal(0, 1, 100)  # Continuous values

        # This should fail for classification with continuous targets
        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )

        # sklearn will raise error for continuous targets in classification
        with pytest.raises(FittingError, match="Unknown label type"):
            binner.fit(X, y=y)

    def test_empty_input(self):
        """Test with empty input data."""
        X = np.empty((0, 2))
        y = np.empty((0,))

        binner = TreeBinning(task_type="classification")

        with pytest.raises(
            (ValueError, FittingError, ValidationError)
        ):  # Should fail during input validation
            binner.fit(X, y=y)

    def test_mismatched_feature_target_length(self):
        """Test with mismatched feature and target lengths."""
        X = np.random.normal(0, 1, (100, 2))
        y = np.random.randint(0, 2, 90)  # Different length

        binner = TreeBinning(task_type="classification")

        with pytest.raises((ValueError, ValidationError)):  # Should fail during validation
            binner.fit(X, y=y)

    def test_transform_without_fitting(self):
        """Test transform before fitting."""
        X = np.random.normal(0, 1, (10, 2))

        binner = TreeBinning()

        with pytest.raises(
            (AttributeError, ValidationError, RuntimeError)
        ):  # Should fail - not fitted
            binner.transform(X)

    def test_very_deep_tree_params(self):
        """Test with very deep tree parameters."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (1000, 2))
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)

        binner = TreeBinning(
            task_type="classification",
            tree_params={
                "max_depth": 20,  # Very deep
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
        )

        binner.fit(X, y=y)
        result = binner.transform(X)

        assert result.shape == (1000, 2)
        # Deep tree might create many bins
        assert len(np.unique(result[:, 0])) >= 2
        assert len(np.unique(result[:, 1])) >= 2

    # ======================
    # Configuration and Internal Tests
    # ======================

    def test_tree_template_creation(self):
        """Test internal tree template creation."""
        binner = TreeBinning(
            task_type="regression", tree_params={"max_depth": 5, "random_state": 123}
        )

        # Tree template should be created
        assert binner._tree_template is not None
        tree_params = binner._tree_template.get_params()
        assert tree_params["max_depth"] == 5
        assert tree_params["random_state"] == 123

    def test_invalid_tree_params_during_fitting(self):
        """Test invalid tree parameters that fail during fitting."""
        # Invalid params should fail during tree template creation, not fitting
        with pytest.raises(ConfigurationError, match="Invalid tree_params"):
            TreeBinning(task_type="classification", tree_params={"invalid_param": "invalid_value"})

    def test_fitted_tree_storage(self, classification_data):
        """Test that fitted trees are stored internally."""
        X, y = classification_data

        binner = TreeBinning(
            task_type="classification", tree_params={"max_depth": 3, "random_state": 42}
        )
        binner.fit(X, y=y)

        # Should have stored trees for each feature
        assert len(binner._fitted_trees) == 2
        assert len(binner._tree_importance) == 2

        # All importance values should be 1.0 (single feature trees)
        assert all(imp == 1.0 for imp in binner._tree_importance.values())

    def test_regression_vs_classification_tree_types(self):
        """Test that correct tree types are created for different task types."""
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        # Classification
        binner_clf = TreeBinning(task_type="classification")
        assert isinstance(binner_clf._tree_template, DecisionTreeClassifier)

        # Regression
        binner_reg = TreeBinning(task_type="regression")
        assert isinstance(binner_reg._tree_template, DecisionTreeRegressor)

    def test_tree_parameter_merging(self):
        """Test that tree parameters are properly merged with defaults."""
        custom_params = {"max_depth": 7, "random_state": 999}
        binner = TreeBinning(task_type="classification", tree_params=custom_params)

        tree = binner._tree_template
        assert tree is not None
        tree_params = tree.get_params()
        assert tree_params["max_depth"] == 7  # Custom value
        assert tree_params["random_state"] == 999  # Custom value
        assert tree_params["min_samples_leaf"] == 1  # Default value
        assert tree_params["min_samples_split"] == 2  # Default value

    # Additional tests for missing coverage

    def test_invalid_task_type_error(self):
        """Test ConfigurationError for invalid task_type (lines 63-65)."""
        from binlearn.utils import ConfigurationError

        # Test during initialization (lines 63-65)
        with pytest.raises(
            ConfigurationError, match="task_type must be 'classification' or 'regression'"
        ):
            TreeBinning(task_type="invalid_type")

    def test_tree_template_caching(self):
        """Test that tree template is cached and not recreated (line 108)."""
        binner = TreeBinning(task_type="classification")

        # First call creates the template
        binner._create_tree_template()
        template1 = binner._tree_template
        assert template1 is not None

        # Second call should hit line 108 (early return)
        binner._create_tree_template()
        template2 = binner._tree_template
        assert template2 is template1  # Same object reference (cached)

    def test_missing_guidance_data_error(self):
        """Test FittingError when guidance_data is None in _calculate_bins."""
        from binlearn.utils import FittingError

        binner = TreeBinning(task_type="classification")
        x_col = np.array([1.0, 2.0, 3.0])

        with pytest.raises(FittingError, match="guidance_data is required for tree binning"):
            binner._calculate_bins(x_col, "test_col", guidance_data=None)

    def test_uninitialized_tree_template_error(self):
        """Test FittingError when tree template is not initialized."""
        from binlearn.utils import FittingError

        binner = TreeBinning(task_type="classification")

        # Force the tree template to be None
        binner._tree_template = None

        x_col = np.array([1.0, 2.0, 3.0])
        y_col = np.array([0, 1, 0])

        with pytest.raises(FittingError, match="Tree template not initialized"):
            binner._calculate_bins(x_col, "test_col", guidance_data=y_col)

    def test_duplicate_edges_filtering(self):
        """Test that duplicate edges are properly filtered in bin edge creation (lines 217->216)."""
        # Create data that would generate duplicate or very close edges
        # to trigger both branches of the condition:
        # if not bin_edges or abs(edge - bin_edges[-1]) > config.float_tolerance:

        # Create edge case data where split points might be very close
        X = np.array([[1.0], [1.0 + 1e-12], [2.0]])  # Very close values (less than float_tolerance)
        y = np.array([0, 0, 1])

        binner = TreeBinning(
            task_type="classification",
            tree_params={
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
        )

        # Fit the binner - this should trigger the duplicate edge filtering logic
        # where some edges are close enough to be considered duplicates
        binner.fit(X, y=y)

        # Transform should work without issues
        result = binner.transform(X)
        assert result is not None
        assert result.shape == X.shape

        # Test with more extreme case to ensure the filtering logic is fully exercised
        X2 = np.array([[1.0], [1.0], [1.0000000001], [2.0]])  # Even closer values
        y2 = np.array([0, 0, 1, 1])

        binner2 = TreeBinning(task_type="classification")
        binner2.fit(X2, y=y2)
        result2 = binner2.transform(X2)
        assert result2 is not None

    def test_edge_filtering_branch_coverage(self):
        """Test the _filter_duplicate_edges method to cover both branches."""
        from binlearn import get_config, set_config

        binner = TreeBinning(task_type="classification")

        # Test case 1: Normal filtering where edges are kept (condition is True)
        edges_normal = [1.0, 1.5, 2.0, 2.5, 3.0]
        result_normal = binner._filter_duplicate_edges(edges_normal)
        assert len(result_normal) == 5  # All edges should be kept
        assert result_normal == edges_normal

        # Save original config for restoration
        original_tolerance = get_config().float_tolerance

        try:
            # Test case 2: Force edge filtering by setting large tolerance (condition becomes False)
            set_config(float_tolerance=1.0)  # Large tolerance to filter out edges

            # Create edges that are closer than the tolerance
            edges_close = [1.0, 1.2, 1.4, 2.0, 2.1]  # Some are within tolerance of 1.0
            result_filtered = binner._filter_duplicate_edges(edges_close)

            # Should have fewer edges due to filtering
            assert len(result_filtered) < len(edges_close)
            assert result_filtered[0] == 1.0  # First edge always kept

            # Test edge case: empty list
            result_empty = binner._filter_duplicate_edges([])
            assert result_empty == []

            # Test single edge
            result_single = binner._filter_duplicate_edges([5.0])
            assert result_single == [5.0]

        finally:
            # Restore original tolerance
            set_config(float_tolerance=original_tolerance)
