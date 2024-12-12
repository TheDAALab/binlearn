"""
This module tests the binning techniques
"""

import pytest

import numpy as np

from binning import (BinningBase,
                            PredefinedDiscreteBinning,
                            PredefinedBinCentersBinning,
                            PredefinedBinRangesBinning,
                            InferredBinsBinning,
                            EqualWidthBinning,
                            EqualFrequencyBinning,
                            KMeansClusteringBinning,
                            AdaptiveBinning)

def test_inferred_bins_binning():
    """
    Testing the binning based on inferred bins
    """

    values = [2, 1, 3, 6, 2, 6, 3]

    binning = InferredBinsBinning()

    binning.fit(values)

    bins = binning.transform(values)

    assert bins.shape[0] == len(values)

    inverse = binning.inverse_transform(bins)

    np.testing.assert_array_equal(np.array(values), inverse)

    assert len(binning.get_params()) == 1

    assert len(binning.bin_range()) == len(np.unique(values))

    np.testing.assert_array_equal(binning.bin_representatives(),
                                  np.array([1, 2, 3, 6]))

    np.testing.assert_array_equal(binning.lookup_bin_widths(np.array([1, 1, 2])),
                                  np.array([1, 1, 3]))

    np.testing.assert_array_equal(binning.lookup_bin_boundaries(np.array([1, 1, 2])),
                                  np.array([[2, 3],
                                            [2, 3],
                                            [3, 6]]))

    np.testing.assert_array_equal(InferredBinsBinning(**binning.get_params())._bins,
                                         binning._bins)

    assert len(binning.to_tuple()) == 3

def test_predefined_discrete_binning():
    """
    Testing the predefined discrete binning
    """

    bins = [[3], [1, 2], [4, 5, 6]]
    values = np.array([2, 1, 5, 3, 5])

    binning = PredefinedDiscreteBinning(bins=bins)

    binning.fit(values)

    bin_indices = binning.transform(values)

    np.testing.assert_array_equal(bin_indices,
                                  np.array([0, 0, 2, 1, 2]))

    inverse = binning.inverse_transform(binning.transform(values))

    np.testing.assert_array_equal(inverse,
                                  np.array([1, 1, 5, 3, 5]))

    np.testing.assert_array_equal(binning.lookup_bin_widths(bin_indices),
                                  np.array([2, 2, 3, 1, 3]))

    np.testing.assert_array_equal(binning.lookup_bin_boundaries(bin_indices),
                                  np.array([[1, 3],
                                            [1, 3],
                                            [4, 7],
                                            [3, 4],
                                            [4, 7]]))

    np.testing.assert_array_equal(binning.bin_range(),
                                  np.array([0, 1, 2]))

    np.testing.assert_array_equal(binning.bin_representatives(),
                                  np.array([1, 3, 5]))

    assert PredefinedDiscreteBinning(**binning.get_params())._bins == binning._bins

    assert len(binning.to_tuple()) == 3

def test_predefined_bin_centers_binning():
    """
    Testing the bin center binning
    """

    bin_centers = [4, 2, 6, 7]

    x = np.array([1.1, 3.4, 1.0, 2.8, 9.9])

    binning = PredefinedBinCentersBinning(bin_centers=bin_centers)

    binning.fit(x)

    tmp = binning.transform(x)

    np.testing.assert_array_equal(tmp, np.array([0, 1, 0, 0, 3]))

    inverse = binning.inverse_transform(tmp)

    np.testing.assert_array_equal(inverse, np.array([2, 4, 2, 2, 7]))

    np.testing.assert_array_equal(binning.lookup_bin_widths(tmp),
                                  np.array([2., 2., 2., 2., 1.]))

    np.testing.assert_array_equal(binning.lookup_bin_boundaries(tmp),
                                  np.array([[1., 3.],
                                            [3., 5.],
                                            [1., 3.],
                                            [1., 3.],
                                            [6.5, 7.5]]))

    np.testing.assert_array_equal(binning.bin_representatives(),
                                  np.array([2, 4, 6, 7]))

    np.testing.assert_array_equal(binning.bin_range(),
                                  np.array([0, 1, 2, 3]))

    tmp_boundaries = PredefinedBinCentersBinning(**binning.get_params())\
                                                .lookup_bin_boundaries(tmp)

    np.testing.assert_array_equal(binning.lookup_bin_boundaries(tmp),
                                  tmp_boundaries)

    assert len(binning.to_tuple()) == 3

def test_predefined_bin_ranges_binning():
    """
    Testing the bin ranges binning
    """

    bin_ranges = np.array([[1, 3], [3, 4], [4, 6], [6, 10]])

    x = np.array([1.1, 3.4, 1.0, 2.8, 9.9, 11.2, 0.1])

    binning = PredefinedBinRangesBinning(bin_ranges=bin_ranges)

    binning.fit(x)

    bin_indices = binning.transform(x)

    np.testing.assert_array_equal(bin_indices, np.array([0, 1, 0, 0, 3, 3, 0]))

    inverse = binning.inverse_transform(bin_indices)

    np.testing.assert_array_equal(inverse, np.array([2., 3.5, 2., 2., 8., 8., 2.]))

    assert len(binning.get_params()) == 1

    assert len(binning.bin_range()) == len(bin_ranges)

    np.testing.assert_array_equal(binning.lookup_bin_widths(bin_indices),
                                  np.array([2, 1, 2, 2, 4, 4, 2]))

    np.testing.assert_array_equal(binning.lookup_bin_boundaries(bin_indices),
                                  np.array([[1, 3],
                                            [3, 4],
                                            [1, 3],
                                            [1, 3],
                                            [6, 10],
                                            [6, 10],
                                            [1, 3]]))

    np.testing.assert_array_equal(binning.bin_representatives(),
                                  np.array([2., 3.5, 5., 8.]))

    np.testing.assert_array_equal(binning.bin_range(),
                                  np.array([0, 1, 2, 3]))

    tmp_boundaries = PredefinedBinRangesBinning(**binning.get_params())\
                                                .lookup_bin_boundaries(bin_indices)

    np.testing.assert_array_equal(binning.lookup_bin_boundaries(bin_indices),
                                  tmp_boundaries)

    assert len(binning.to_tuple()) == 3

def test_equal_width_binning():
    """
    Testing the equal width binning
    """

    values = [1, 3, 4, 10]
    eqwb = EqualWidthBinning(n_bins=3)

    eqwb.fit(values)

    assert eqwb._lower_bounds[0] == np.min(values)
    assert eqwb._upper_bounds[-2] == np.max(values)

    np.testing.assert_array_equal(eqwb.transform(values),
                                  np.array([0, 0, 1, 3]))

    np.testing.assert_array_equal(eqwb.inverse_transform(eqwb.transform(values)),
                                  np.array([2.5, 2.5, 5.5, 11.5]))

    np.testing.assert_array_equal(eqwb.bin_range(),
                                  np.array([0, 1, 2, 3]))

    np.testing.assert_array_equal(eqwb.bin_representatives(),
                                  np.array([2.5, 5.5, 8.5, 11.5]))

    np.testing.assert_array_equal(eqwb.lookup_bin_widths(eqwb.transform(values)),
                                  np.array([3., 3., 3., 3.]))

    np.testing.assert_array_equal(eqwb.lookup_bin_boundaries(eqwb.transform(values)),
                                  np.array([[1., 4.],
                                            [1., 4.],
                                            [4., 7.],
                                            [10., 13.]]))

    assert len(eqwb.get_params()) == 2

    tmp_boundaries = EqualWidthBinning(**eqwb.get_params())\
                                    .lookup_bin_boundaries(eqwb.transform(values))

    np.testing.assert_array_equal(eqwb.lookup_bin_boundaries(eqwb.transform(values)),
                                  tmp_boundaries)

    assert len(eqwb.to_tuple()) == 3

def test_equal_frequency_binning():
    """
    Testing the equal frequency binning
    """

    values = np.array([1, 3, 4, 10])
    eqfb = EqualFrequencyBinning(n_bins=3)

    eqfb.fit(values)

    np.testing.assert_array_equal(eqfb.transform(values),
                                  np.array([0, 0, 1, 2]))

    np.testing.assert_array_equal(eqfb.inverse_transform(eqfb.transform(values)),
                                  np.array([2.25, 2.25, 6.75, 13.25]))

    np.testing.assert_array_equal(eqfb.bin_range(),
                                  np.array([0, 1, 2]))

    np.testing.assert_array_equal(eqfb.bin_representatives(),
                                  np.array([2.25, 6.75, 13.25]))

    np.testing.assert_array_equal(eqfb.lookup_bin_widths(eqfb.transform(values)),
                                  np.array([2.5, 2.5, 6.5, 6.5]))

    np.testing.assert_array_equal(eqfb.lookup_bin_boundaries(eqfb.transform(values)),
                                  np.array([[1., 3.5],
                                            [1., 3.5],
                                            [3.5, 10.],
                                            [10., 16.5]]))

    assert len(eqfb.get_params()) == 2

    tmp_boundaries = EqualFrequencyBinning(**eqfb.get_params())\
                                    .lookup_bin_boundaries(eqfb.transform(values))

    np.testing.assert_array_equal(eqfb.lookup_bin_boundaries(eqfb.transform(values)),
                                  tmp_boundaries)

    assert len(eqfb.to_tuple()) == 3

def test_kmeans_clustering_binning():
    """
    Testing the k-means clustering binning
    """

    values = np.array([1, 4, 4, 10, 11, 15, 31])
    kmcb = KMeansClusteringBinning(n_bins=3)

    kmcb.fit(values)

    np.testing.assert_array_equal(kmcb.transform(values),
                                  np.array([0, 0, 0, 1, 1, 1, 2]))

    np.testing.assert_array_equal(kmcb.inverse_transform(kmcb.transform(values)),
                                  np.array([3., 3., 3., 12., 12., 12., 31.]))

    np.testing.assert_array_equal(kmcb.bin_range(),
                                  np.array([0, 1, 2]))

    np.testing.assert_array_equal(kmcb.bin_representatives(),
                                  np.array([3., 12., 31.]))

    np.testing.assert_array_equal(kmcb.lookup_bin_widths(kmcb.transform(values)),
                                  np.array([9., 9., 9., 14., 14., 14., 19.]))

    np.testing.assert_array_equal(kmcb.lookup_bin_boundaries(kmcb.transform(values)),
                                  np.array([[-1.5, 7.5],
                                            [-1.5, 7.5],
                                            [-1.5, 7.5],
                                            [7.5, 21.5],
                                            [7.5, 21.5],
                                            [7.5, 21.5],
                                            [21.5, 40.5]]))

    assert len(kmcb.get_params()) == 2

    tmp_boundaries = KMeansClusteringBinning(**kmcb.get_params())\
                                    .lookup_bin_boundaries(kmcb.transform(values))

    np.testing.assert_array_equal(kmcb.lookup_bin_boundaries(kmcb.transform(values)),
                                  tmp_boundaries)

    assert len(kmcb.to_tuple()) == 3

def test_adaptive_binning():
    values = np.array([0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5])
    weights  = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1])

    # binning that needs adjustment- removing bins that are below the threshold
    bins = np.array([(0, 1), (1, 2), (2, 3), (3, 5), (5, 9)])

    ab = AdaptiveBinning(binning=('binning', 'PredefinedBinRangesBinning', {'bin_ranges': bins}),
                           min_weight=2,
                           value_distance_tolerance=3
    )
    ab.fit(values=values, observed_values=values, weights=weights, bins_below_threshold='remove')
    
    np.testing.assert_array_equal(ab.lookup_bin_boundaries(np.array([0, 1])),
                                  np.array([[0, 3],
                                            [3, 5]]))
    
    np.testing.assert_array_equal(ab.transform(values),
                                  np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    
    np.testing.assert_array_equal(ab.inverse_transform(ab.transform(values)),
                                  np.array([3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5]))
    
    np.testing.assert_array_equal(ab.lookup_bin_representatives(values),
                                  np.array([3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5]))
    
    # binning that needs adjustment - keeping bins that are below the threshold

    ab = AdaptiveBinning(binning=('binning', 'PredefinedBinRangesBinning', {'bin_ranges': bins}),
                           min_weight=2,
                           value_distance_tolerance=3)
    
    ab.fit(values=values, observed_values=values, weights=weights)
    
    np.testing.assert_array_equal(ab.lookup_bin_boundaries(np.array([0, 1, 2])),
                                  np.array([[0, 3],
                                            [3, 5],
                                            [5, 9]]))
    
    np.testing.assert_array_equal(ab.transform(values),
                                  np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]))
    
    np.testing.assert_array_equal(ab.inverse_transform(ab.transform(values)),
                                  np.array([3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 9]))
    
    np.testing.assert_array_equal(ab.lookup_bin_representatives(values),
                                  np.array([3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 9]))
    
    # binning that needs adjustment- merging bins below threshold with the previous bin
    bins = np.array([(0, 1), (1, 2), (2, 3), (3, 5), (5, 9)])

    ab = AdaptiveBinning(binning=('binning', 'PredefinedBinRangesBinning', {'bin_ranges': bins}),
                           min_weight=2,
                           value_distance_tolerance=3
    )
    ab.fit(values=values, observed_values=values, weights=weights, bins_below_threshold='merge')
    
    np.testing.assert_array_equal(ab.lookup_bin_boundaries(np.array([0, 1])),
                                  np.array([[0, 3],
                                            [3, 9]]))
    
    # binning that needs adjustment- merging bins below threshold with the previous bin
    bins = np.array([(0, 1), (1, 2), (2, 3), (3, 5), (5, 9)])

    ab = AdaptiveBinning(binning=('binning', 'PredefinedBinRangesBinning', {'bin_ranges': bins}),
                           min_weight=2,
                           value_distance_tolerance=1
    )
    ab.fit(values=values, observed_values=values, weights=weights, bins_below_threshold='merge')
    
    np.testing.assert_array_equal(ab.lookup_bin_boundaries(np.array([0, 1])),
                                  np.array([[0, 3],
                                            [3, 9]]))

    
    # binning that needs adjustment- merging bins where none of them has enough purchases
    bins = np.array([(0, 1), (1, 2), (2, 3), (3, 5), (5, 9)])
    zeros  = np.zeros(len(values))

    ab = AdaptiveBinning(binning=('binning', 'PredefinedBinRangesBinning', {'bin_ranges': bins}),
                           min_weight=2,
                           value_distance_tolerance=3
    )
    ab.fit(values=values, observed_values=values, weights=zeros, bins_below_threshold='merge')
    
    np.testing.assert_array_equal(ab.lookup_bin_boundaries(np.array([0])),
                                  np.array([[0, 9]]))
    
    # no adjustment
    ab = AdaptiveBinning(binning=('binning', 'PredefinedBinRangesBinning', {'bin_ranges': bins}),
                           min_weight=0,
                           value_distance_tolerance=None
    ).fit(values=values, observed_values=values, weights=None)

    np.testing.assert_array_equal(ab.lookup_bin_boundaries([0, 1, 2, 3, 4]),
                                  bins)
    
    np.testing.assert_array_equal(ab.transform(values),
                                  np.array([0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4]))

    # binning that already satisfies the requirement
    values = np.array([0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5])
    weights  = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    bins = [(0, 3), (3, 5), (5, 9)]

    ab = AdaptiveBinning(binning=('binning', 'PredefinedBinRangesBinning', {'bin_ranges': bins}),
                           min_weight=2,
                           value_distance_tolerance=3
    ).fit(values=values, observed_values=values, weights=weights)

    np.testing.assert_array_equal(ab.lookup_bin_boundaries([0, 1, 2]),
                                  np.array([[0, 3],
                                            [3, 5],
                                            [5, 9]]))
    
    np.testing.assert_array_equal(ab.transform(values),
                                  np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]))
    
    np.testing.assert_array_equal(ab.lookup_bin_representatives(values),
                                  np.array([3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 9, 9]))
    
    
    
    
