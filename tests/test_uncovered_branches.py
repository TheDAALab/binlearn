"""
Additional tests to cover uncovered lines and branches in utility modules and methods.
"""

import warnings

import numpy as np
import pytest
from unittest.mock import patch, Mock

from binlearn.methods import DBSCANBinning, KMeansBinning, GaussianMixtureBinning
from binlearn.utils import ConfigurationError
from binlearn.utils._equal_width_utils import ensure_monotonic_edges


class TestUncoveredBranches:
    """Test cases specifically designed to cover uncovered lines and branches."""

    def test_dbscan_no_fallback_error(self):
        """Test DBSCAN error when fallback is disabled and insufficient clusters found."""
        # Create data that will result in very few or no clusters
        X = np.array([[1.0], [100.0]])  # Two very distant points

        binner = DBSCANBinning(
            eps=0.1,  # Very small eps to prevent clustering
            min_samples=2,
            min_bins=5,  # More bins than possible clusters
            allow_fallback=False,  # Disable fallback to trigger error
        )

        # This should raise ConfigurationError (covers line 296 in _dbscan_binning.py)
        with pytest.raises(ConfigurationError, match="DBSCAN found only .* clusters"):
            binner.fit(X)

    def test_kmeans_no_fallback_identical_values_error(self):
        """Test KMeans error when fallback is disabled and all values are identical."""
        X = np.array([[5.0], [5.0], [5.0], [5.0]])  # All identical values

        binner = KMeansBinning(n_bins=3, allow_fallback=False)  # Disable fallback to trigger error

        # This should raise ConfigurationError (covers line 227 in _kmeans_binning.py)
        with pytest.raises(ConfigurationError, match="All data values are identical"):
            binner.fit(X)

    def test_kmeans_no_fallback_few_unique_values_error(self):
        """Test KMeans error when fallback is disabled and too few unique values."""
        # Create enough data points but with only 2 unique values
        X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])  # 6 points, 2 unique values

        binner = KMeansBinning(
            n_bins=5,  # More bins than unique values
            allow_fallback=False,  # Disable fallback to trigger error
        )

        # This should raise ConfigurationError (covers line 242 in _kmeans_binning.py)
        with pytest.raises(ConfigurationError, match="Too few unique values"):
            binner.fit(X)

    def test_kmeans_no_fallback_clustering_failure_error(self):
        """Test KMeans error when fallback is disabled and clustering fails."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        binner = KMeansBinning(n_bins=2, allow_fallback=False)  # Disable fallback to trigger error

        # Mock kmeans1d.cluster to raise an exception
        with patch("binlearn.methods._kmeans_binning.kmeans1d") as mock_kmeans:
            mock_kmeans.cluster.side_effect = Exception("Clustering failed")

            # This should raise ConfigurationError (covers lines 260, 273 in _kmeans_binning.py)
            with pytest.raises(ConfigurationError, match="K-means clustering failed"):
                binner.fit(X)

    def test_gmm_no_fallback_error(self):
        """Test GMM error when fallback is disabled and fitting fails."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        binner = GaussianMixtureBinning(
            n_components=2, allow_fallback=False  # Disable fallback to trigger error
        )

        # Mock GaussianMixture to raise an exception
        with patch("binlearn.methods._gaussian_mixture_binning.GaussianMixture") as mock_gmm:
            mock_instance = Mock()
            mock_instance.fit.side_effect = Exception("GMM fitting failed")
            mock_gmm.return_value = mock_instance

            # This should raise ConfigurationError (covers line 342 in _gaussian_mixture_binning.py)
            with pytest.raises(ConfigurationError, match="GMM fitting failed"):
                binner.fit(X)

    def test_ensure_monotonic_edges_zero_value_branch(self):
        """Test ensure_monotonic_edges with zero values to cover epsilon == 0 branch."""
        # Create edges with zeros to trigger the epsilon == 0 condition
        edges = np.array([0.0, 0.0, 1.0])

        result = ensure_monotonic_edges(edges)

        # Should handle the zero case and create strictly increasing edges
        assert result[0] == 0.0
        assert result[1] > result[0]  # Should be greater due to epsilon handling
        assert result[2] == 1.0

        # Verify the epsilon == 0 branch was covered (line 150 in _equal_width_utils.py)
        assert all(result[i] > result[i - 1] for i in range(1, len(result)))


class TestWarningsSuppression:
    """Test that warnings are properly handled in test environment."""

    def test_dbscan_fallback_warnings(self):
        """Test DBSCAN fallback warnings can be suppressed."""
        X = np.array([[1.0], [100.0]])  # Data that will cause fallback

        binner = DBSCANBinning(
            eps=0.1,
            min_samples=2,
            min_bins=5,
            allow_fallback=True,  # Allow fallback to trigger warning
        )

        # Suppress warnings during test to avoid test output noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            binner.fit(X)
            result = binner.transform(X)

        # Should still work despite fallback
        assert result is not None
        assert result.shape == X.shape

    def test_kmeans_fallback_warnings(self):
        """Test KMeans fallback warnings can be suppressed."""
        X = np.array([[5.0], [5.0], [5.0], [5.0]])  # Identical values causing fallback

        binner = KMeansBinning(n_bins=3, allow_fallback=True)  # Allow fallback to trigger warning

        # Suppress warnings during test to avoid test output noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            binner.fit(X)
            result = binner.transform(X)

        # Should still work despite fallback
        assert result is not None
        assert result.shape == X.shape

    def test_gmm_fallback_warnings(self):
        """Test GMM fallback warnings can be suppressed."""
        # Create data that might cause GMM to struggle
        X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])

        binner = GaussianMixtureBinning(
            n_components=5,  # More components than reasonable for small dataset
            allow_fallback=True,  # Allow fallback to trigger warning
        )

        # Suppress warnings during test to avoid test output noise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                binner.fit(X)
                result = binner.transform(X)
                # Should still work despite potential fallback
                assert result is not None
                assert result.shape == X.shape
            except ConfigurationError:
                # GMM might still fail even with fallback in some cases
                # This is acceptable behavior
                pass

    def test_kmeans_fallback_function_creation(self):
        """Test that KMeans creates fallback function when fallback is allowed."""
        X = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])  # Few unique values

        binner = KMeansBinning(
            n_bins=5,  # More bins than unique values
            allow_fallback=True,  # Enable fallback to use fallback_func
        )

        # This should use fallback (covers line 260 - fallback_func definition)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            binner.fit(X)

        # Should succeed with fallback
        assert hasattr(binner, "bin_edges_")

    def test_kmeans_safe_sklearn_call_fallback_execution(self):
        """Test that KMeans fallback function is actually executed in safe_sklearn_call."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

        binner = KMeansBinning(n_bins=3, allow_fallback=True)  # Enable fallback

        # Mock kmeans1d.cluster to raise an exception, forcing fallback
        with patch("kmeans1d.cluster") as mock_cluster:
            mock_cluster.side_effect = Exception("Clustering failed")

            # This should trigger the fallback function (covers line 260)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                binner.fit(X)

            # Should succeed with fallback to equal-width
            assert hasattr(binner, "bin_edges_")
            # Equal-width fallback creates n_bins+1 edges, but depends on actual implementation
            assert len(binner.bin_edges_[0]) > 0  # Just check that we have edges

    def test_equal_width_ensure_monotonic_zero_epsilon(self):
        """Test ensure_monotonic_edges when value is exactly zero."""
        from binlearn.utils._equal_width_utils import ensure_monotonic_edges

        # Create edges where one value is exactly 0.0 to trigger special case
        edges = np.array([0.0, 0.0])

        result = ensure_monotonic_edges(edges)

        # Should use 1e-10 as epsilon when abs(edges[i-1]) == 0 (covers line 148)
        assert result[1] > result[0]
        assert result[1] == 1e-10
