# flake8: noqa
"""
This module implements some general purpose binning capabilities.
"""

# TODO: better interfaces and structure would be desired
# TODO: possibly the binnings are sklearn Transformers?
# TODO: own kmeans1d implementation would be great, the kmeans1d package is not a popular one
# TODO: there might be problems (empty bins) with equal frequency binning if there are many ties

from __future__ import annotations

import abc
import numpy as np
import kmeans1d
import importlib
#from ..base import instantiate_obj

__all__ = [
    'BinningBase',
    'PredefinedBinCentersBinning',
    'PredefinedBinRangesBinning',
    'PredefinedDiscreteBinning',
    'EqualWidthBinning',
    'EqualFrequencyBinning',
    'KMeansClusteringBinning',
    'InferredBinsBinning',
    'AdaptiveBinning',
]


class BinningBase(abc.ABC):
    """
    Base class for binning
    """

    def fit(self, values) -> BinningBase:
        """
        Fitting the binning to values

        Args:
            values (np.ndarray): the values to fit to

        Returns:
            self: the fitted object
        """
        return self

    @abc.abstractmethod
    def transform(self, values):
        """
        Assign bin indices to the values in `x`

        Args:
            values (np.ndarray): the values to assign bin indices to

        Returns:
            np.ndarray: the bin indices
        """

    @abc.abstractmethod
    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.ndarray: the representative values for the bins
        """

    def inverse_transform(self, bin_indices: np.ndarray):
        """
        Transforms bin indices to the original domain. The result contains
        representative values for the bins, for example bin centers

        Args:
            bin_indices (np.ndarray): the bin index array to invert

        Returns:
            np.ndarray: the representative values for the bins
        """
        return self.bin_representatives()[bin_indices]

    @abc.abstractmethod
    def lookup_bin_widths(self, bin_indices: np.ndarray):
        """
        Returns the bin widths to bin indices

        Args:
            bin_indices (np.ndarray): the bin index array to invert

        Returns:
            np.ndarray: the bin widths
        """

    @abc.abstractmethod
    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.ndarray): the bin index array to invert

        Returns:
            np.ndarray: the bin boundaries
        """

    def bin_range(self):
        """
        Returns the range of bin indices

        Returns:
            np.ndarray: the range of bin indices
        """
        return np.arange(len(self.bin_representatives()))

    @abc.abstractmethod
    def get_params(self):
        """
        Returns the parameters of the binning

        Returns:
            dict: the parameters of the binning
        """

    def to_tuple(self):
        """
        Transforms the binning into a tuple representation

        Returns:
            tuple: the tuple representation of the binning
        """
        return (self.__class__.__module__, self.__class__.__name__, self.get_params())


class InferredBinsBinning(BinningBase):
    """
    Binning based on bins inferred from the data
    """

    def __init__(self, *, bins: np.ndarray | None = None, strict: bool = False):
        """
        Constructor of the object

        Args:
            bins (None/np.ndarray): the bin values
            strict (bool): if True, out of range values result in a
                            ValueError during a transform call
        """
        self._bins = None
        self._lower_bounds = None
        self._upper_bounds = None
        self._boundaries = None
        self._widths = None
        self._strict = strict

        if bins is not None:
            self._init_internals(bins_sorted=np.unique(bins))

    def _init_internals(self, bins_sorted: np.ndarray):
        # lower bounds are the representatives
        self._bins = bins_sorted
        self._lower_bounds = self._bins.copy()  # andrasva: why copy?

        self._upper_bounds = np.hstack(
            [
                self._bins[1:],
                self._bins[-1]
                + ((self._bins[-1] - self._bins[-2]) if len(self._bins) > 1 else 1),
            ]
        )

        self._boundaries = np.vstack([self._lower_bounds, self._upper_bounds]).T

        # andrasva: could be lifted to BinningBase...
        self._widths = self._upper_bounds - self._lower_bounds

    def fit(self, values) -> InferredBinsBinning:
        """
        Fitting to data

        Args:
            values (np.ndarray): the data to fit to

        Returns:
            self: the fitted binning
        """
        # mutate and return self to allow chain of member function calls
        self._init_internals(bins_sorted=np.unique(values))
        return self

    def transform(self, values):
        """
        Assign bin indices to the values in `values`

        Args:
            values (np.ndarray): the values to assign bin indices to

        Returns:
            np.ndarray: the bin indices
        """
        values = np.array(values)
        lower_values = values[values < self._lower_bounds[0]].tolist()
        higher_values = values[values >= self._upper_bounds[-1]].tolist()
        out_of_range_values = lower_values + higher_values

        if self._strict and len(out_of_range_values) > 0:
            raise ValueError(f'The following value(s) {out_of_range_values} are out of range for the strict binning')
        
        mask_matrix = values[:, None] == self._bins
        indices = np.argmax(mask_matrix, axis=1)
        mask = np.any(mask_matrix, axis=1)
        res = np.where(mask, indices, np.searchsorted(self._bins, values) - 1) 
        
        return np.maximum(0, res)

    def lookup_bin_widths(self, bin_indices: np.ndarray):
        """
        Returns the bin widths to bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin widths
        """
        return self._widths[bin_indices]

    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin boundaries
        """
        return self._boundaries[bin_indices]

    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.ndarray: the representatives
        """

        # same as lower bounds...
        return self._bins

    def get_params(self):
        """
        Returns the parameters of the binning

        Returns:
            dict: the parameters of the binning
        """
        bins = None if self._bins is None else self._bins.copy()
        return {'bins': bins}


class PredefinedDiscreteBinning(BinningBase):
    """
    Binning with predefined discrete bins
    """

    def __init__(self, *, bins: list[list[int]]):
        """
        The expected bin specification is a list of lists encoding
        the elements of the bins.

        Args:
            bins (list[list[int]]): the bin specifications
        """
        bin_mins = [np.min(bin) for bin in bins]
        sorter = np.argsort(bin_mins)
        self._bins = [bins[idx] for idx in sorter]

        # number of points in the bins
        self._widths = np.array([len(bin) for bin in self._bins])

        lower_bounds = np.array([np.min(bin) for bin in self._bins])  # inclusive
        upper_bounds = np.array([np.max(bin) + 1 for bin in self._bins])  # exclusive
        self._boundaries = np.vstack([lower_bounds, upper_bounds]).T

        self._representatives = np.mean(self._boundaries - 1, axis=1)
        self._representatives = np.ceil(self._representatives).astype(int)

        self._lowest = np.min(self._boundaries)  # inclusive
        self._highest = np.max(self._boundaries)  # exclusive
        self._lookup = np.zeros(int(self._highest - self._lowest + 1))

        for idx, bin in enumerate(self._bins):
            self._lookup[(np.array(bin) - self._lowest).astype(int)] = idx

        self._lookup = self._lookup.astype(int)

    def transform(self, values):
        """
        Assign bin indices to the values in `values`

        Args:
            values (np.ndarray): the values to assign bin indices to

        Returns:
            np.ndarray: the bin indices
        """
        return self._lookup[(values - self._lowest).astype(int)]

    def lookup_bin_widths(self, bin_indices: np.ndarray):
        """
        Returns the bin widths to bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin widths
        """
        return self._widths[bin_indices]

    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin boundaries
        """
        return self._boundaries[bin_indices]

    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.ndarray: the representatives
        """
        return self._representatives

    def get_params(self):
        """
        Returns the parameters of the binning

        Returns:
            dict: the parameters of the binning
        """
        return {'bins': self._bins.copy()}


class PredefinedBinCentersBinning(BinningBase):
    """
    Predefined bin centers binning
    """

    def __init__(self, *, bin_centers: np.ndarray):
        """
        Constructor of the binning

        Args:
            bin_centers (np.ndarray): sorted and unique bin centers
        """

        # self._bin_centers = bin_centers
        self._bin_centers = np.sort(bin_centers)  # sorting should be done outside

        midpoints = (self._bin_centers[:-1] + self._bin_centers[1:]) / 2.0
        self._lower_bounds = np.hstack([np.array([-np.inf]), midpoints])
        self._upper_bounds = np.hstack([midpoints, np.array([np.inf])])
        self._widths = self._upper_bounds - self._lower_bounds

    def transform(self, values):
        """
        Assign bin indices to the values in `values`

        Args:
            values (np.ndarray): the values to assign bin indices to

        Returns:
            np.ndarray: the bin indices
        """
        distances = np.abs(values[:, None] - self._bin_centers)
        return np.argmin(distances, axis=1)

    def lookup_bin_widths(self, bin_indices: np.ndarray):
        """
        Returns the bin widths to bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin widths
        """
        return self._widths[bin_indices]

    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin boundaries
        """
        boundaries = np.vstack([self._lower_bounds, self._upper_bounds]).T
        return boundaries[bin_indices]

    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.ndarray: the representatives
        """
        return self._bin_centers

    def get_params(self):
        """
        Returns the parameters of the binning

        Returns:
            dict: the parameters of the binning
        """
        return {'bin_centers': self._bin_centers.copy()}


class PredefinedBinRangesBinning(BinningBase):
    """
    Binning with predefined bin ranges
    """

    def __init__(self, *, bin_ranges: np.ndarray, strict=False):
        """
        Constructor of binning with predefined bin ranges

        Args:
            bin_ranges (np.ndarray): an iterable defining the ranges of the bins,
                    each item should contain either one element (to indicate a discrete bin),
                    or two elements indicating the lower (inclusive) and upper (exclusive)
                    boundaries of the bin
            strict (bool): if True, out of range values result in a
                            ValueError during a transform call
        """
        self._bin_ranges = np.array(bin_ranges)
        self._lower_bounds = np.sort(self._bin_ranges[:, 0])
        self._upper_bounds = np.sort(self._bin_ranges[:, 1])
        self._representatives = np.mean(self._bin_ranges, axis=1)
        self._widths = self._bin_ranges[:, 1] - self._bin_ranges[:, 0]
        self._strict = strict

    def transform(self, values):
        """
        Assign bin indices to the values in `values`

        Args:
            values (np.ndarray): the values to assign bin indices to

        Returns:
            np.ndarray: the bin indices
        """
        values = np.array(values)
        lower_values = values[values < self._lower_bounds[0]].tolist()
        higher_values = values[values >= self._upper_bounds[-1]].tolist()
        out_of_range_values = lower_values + higher_values

        if self._strict and len(out_of_range_values) > 0:
            raise ValueError(f'The following value(s) {out_of_range_values} are out of range for the strict binning')
        
        mask = (values[:, None] >= self._lower_bounds) & (
            values[:, None] < self._upper_bounds
        )

        mask_upper = values[:, None] >= self._upper_bounds
        mu = np.all(mask_upper, axis=1)

        if np.any(mu):
            mask[mu, -1] = True

        return np.argmax(mask, axis=1)

    def lookup_bin_widths(self, bin_indices: np.ndarray):
        """
        Returns the bin widths to bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin widths
        """
        return self._widths[bin_indices]

    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin boundaries
        """
        return self._bin_ranges[bin_indices]

    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.ndarray: the representatives
        """
        return self._representatives

    def get_params(self):
        """
        Returns the parameters of the binning

        Returns:
            dict: the parameters of the binning
        """
        return {'bin_ranges': self._bin_ranges.copy()}


class EqualWidthBinning(BinningBase):
    """
    Implements the equal width binning technique inferred from data
    """

    def __init__(self, *, n_bins: int, binning_params=None, strict=False):
        """
        Constructor of the equal width binning

        Args:
            n_bins (int): the number of bins (the actual number will
                            be this `n_bins + 1` to cover the
                            latest data point with a bin)
            binning_params (None/dict): the parameters of the binning
                                        if it has been already fitted
            strict (bool): if True, out of range values result in a
                            ValueError during a transform call
        """

        self._n_bins = n_bins
        self._binning = None
        self._lower_bounds = None
        self._upper_bounds = None
        self._strict = strict

        if binning_params is not None:
            self._binning = PredefinedBinRangesBinning(**binning_params)
            self._lower_bounds = self._binning._lower_bounds
            self._upper_bounds = self._binning._upper_bounds

    def fit(self, values) -> EqualWidthBinning:
        """
        Fitting to data

        Args:
            values (np.ndarray): the data to fit to

        Returns:
            self: the fitted binning
        """
        x_min = np.min(values)
        diff = (np.max(values) - x_min) / self._n_bins
        self._lower_bounds = np.arange(self._n_bins + 1) * diff + x_min
        self._upper_bounds = np.arange(1, self._n_bins + 2) * diff + x_min
        bin_ranges = np.vstack([self._lower_bounds, self._upper_bounds]).T
        self._binning = PredefinedBinRangesBinning(bin_ranges=bin_ranges, strict=self._strict)

        return self

    def transform(self, values):
        """
        Assign bin indices to the values in `values`

        Args:
            values (np.ndarray): the values to assign bin indices to

        Returns:
            np.ndarray: the bin indices
        """        
        return self._binning.transform(values)

    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin boundaries
        """
        return self._binning.lookup_bin_boundaries(bin_indices)

    def lookup_bin_widths(self, bin_indices: np.ndarray):
        """
        Returns the bin widths to bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin widths
        """
        return self._binning.lookup_bin_widths(bin_indices)

    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.ndarray: the representatives
        """
        return self._binning.bin_representatives()

    def get_params(self):
        """
        Returns the parameters of the binning

        Returns:
            dict: the parameters of the binning
        """
        binning_params = None if self._binning is None else self._binning.get_params()
        return {'n_bins': self._n_bins, 'binning_params': binning_params}


class EqualFrequencyBinning(BinningBase):
    """
    Implements the equal frequency binning
    """

    def __init__(self, *, n_bins, binning_params=None):
        """
        Constructor of the equal frequency binning
        """
        self._n_bins = n_bins
        self._binning = None
        if binning_params is not None:
            self._binning = PredefinedBinRangesBinning(**binning_params)

    def fit(self, values) -> EqualFrequencyBinning:
        """
        Fitting to data

        Args:
            values (np.ndarray): the data to fit to

        Returns:
            self: the fitted binning
        """
        percentiles = np.linspace(0, 100, self._n_bins)

        bin_boundaries = np.percentile(values, percentiles)

        lower_bounds = bin_boundaries
        upper_bounds = np.hstack(
            [
                bin_boundaries[1:],
                bin_boundaries[-1] + bin_boundaries[-1] - bin_boundaries[-2],
            ]
        )
        bin_ranges = np.vstack([lower_bounds, upper_bounds]).T

        self._binning = PredefinedBinRangesBinning(bin_ranges=bin_ranges)

    def transform(self, values):
        """
        Assign bin indices to the values in `values`

        Args:
            values (np.ndarray): the values to assign bin indices to

        Returns:
            np.ndarray: the bin indices
        """
        return self._binning.transform(values)

    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin boundaries
        """
        return self._binning.lookup_bin_boundaries(bin_indices)

    def lookup_bin_widths(self, bin_indices: np.ndarray):
        """
        Returns the bin widths to bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin widths
        """
        return self._binning.lookup_bin_widths(bin_indices)

    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.ndarray: the representatives
        """
        return self._binning.bin_representatives()

    def get_params(self):
        """
        Returns the parameters of the binning

        Returns:
            dict: the parameters of the binning
        """
        binning_params = None if self._binning is None else self._binning.get_params()
        return {'n_bins': self._n_bins, 'binning_params': binning_params}


class KMeansClusteringBinning(BinningBase):
    """
    1D k-means clustering based binning, the bins are defined as the clusters
    """

    def __init__(self, *, n_bins: int, binning_params=None):
        """
        Constructor of the equal width binning

        Args:
            n_bins (int): the number of bins (the actual number will
                            be this `n_bins + 1` to cover the
                            latest data point with a bin)
            binning_params (None/dict): the parameters of the binning
                                        if it has been already fitted
        """
        self._n_bins = n_bins
        self._binning = None
        if binning_params is not None:
            self._binning = PredefinedBinCentersBinning(**binning_params)

    def fit(self, values):
        """
        Fitting to data

        Args:
            values (np.ndarray): the data to fit to

        Returns:
            self: the fitted binning
        """
        _, centroids = kmeans1d.cluster(values, self._n_bins)
        self._binning = PredefinedBinCentersBinning(bin_centers=centroids)
        return self

    def transform(self, values):
        """
        Assign bin indices to the values in `values`

        Args:
            values (np.ndarray): the values to assign bin indices to

        Returns:
            np.ndarray: the bin indices
        """
        return self._binning.transform(values)

    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin boundaries
        """
        return self._binning.lookup_bin_boundaries(bin_indices)

    def lookup_bin_widths(self, bin_indices: np.ndarray):
        """
        Returns the bin widths to bin indices

        Args:
            bin_indices (np.ndarray): the bin indices

        Returns:
            np.ndarray: the bin widths
        """
        return self._binning.lookup_bin_widths(bin_indices)

    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.ndarray: the representatives
        """
        return self._binning.bin_representatives()

    def get_params(self):
        """
        Returns the parameters of the binning

        Returns:
            dict: the parameters of the binning
        """
        binning_params = None if self._binning is None else self._binning.get_params()
        return {'n_bins': self._n_bins, 'binning_params': binning_params}

def instantiate_obj(description):
    """
    Instantiates an object from a description
    Args:
        description (tuple): (module_name, class_name, params_dict)
    Returns:
        obj: the instantiated object
    """

    module = importlib.import_module(description[0])
    class_ = getattr(module, description[1])
    return class_(**description[2])

class AdaptiveBinning:
    def __init__(self, *, binning, value_distance_tolerance, min_weight=0):
        """
        Constructor of adaptive binning with adjustable bin ranges according to a weight vector and a weight limit

        Args:
            binning: the 'base' binning to apply to the values
            value_distance_tolerance: the maximum applicable value distance by which the bins can be adjusted
            min_weight: the minimum weight a bin should have. If this is not met for a given bin,
                                the fit() method will adjust the bin's upper bound and the subsequent bin's lower bound.
        """

        """if isinstance(binning, tuple):
            self.binning = instantiate_obj(binning)
        elif isinstance(binning, BinningBase):
            self.binning = binning
        else:
            self.binning = InferredBinsBinning()
        """

        if isinstance(binning, tuple):
            self.binning = instantiate_obj(binning)
        elif isinstance(binning, BinningBase):
            self.binning = binning
        else:
            self.binning = InferredBinsBinning()

        self.value_distance_tolerance = value_distance_tolerance
        self.min_weight = min_weight if min_weight is not None else 0
        self._bin_boundaries = None

    def fit(self, values, observed_values, weights, bins_below_threshold='keep'):
        """
        Fitting to data. If a min_weight is set the bins will be adjusted to include enough weight, starting with the lowest bin.
        The resulting bins that are still below the min_weight threshold are either kept (default), removed or merged
        into the previous (lower) bin. Removal can leave "holes" in the binning coverage of the support, this can be mitigated by either
        merging instead or increasing the value_distance_tolerance in the constructor.

        Args:
            observed_values (np.array): the value data to fit the binning to
            weights (np.array): the weight data which are used to adjust the binning
            bins_below_threshold (str): how to handle bins that after the adjustment
                                        have weights below the threshold. Only effective if adjustment takes place.

        Returns:
            self: the fitted binning
        """
        self.binning.fit(values)
        bin_boundaries = self.binning.lookup_bin_boundaries(self.binning.bin_range())

        if self.min_weight > 0 and weights is not None:
            self._bin_boundaries = self.adjust_bin_boundaries(
                bin_boundaries, observed_values, weights, bins_below_threshold
            )
        else:
            self._bin_boundaries = bin_boundaries

        return self

    def transform(self, values):
        """
        Assign bin indices to the values in `values`.

        Args:
            values (np.array): the values to assign bin indices to

        Returns:
            np.array: the bin indices
        """

        mask = (values[:, None] >= self._bin_boundaries[:, 0]) & (
            values[:, None] < self._bin_boundaries[:, 1]
        )

        # TODO what if a value is outside of the bin range - currently they are assigned to the closest bin defined
        mask_upper = values[:, None] >= self._bin_boundaries[:, 1]
        mu = np.all(mask_upper, axis=1)

        if np.any(mu):
            mask[mu, -1] = True

        # the below would make the original behavior explicit
        # mask_lower = values[:, None] < self._bin_boundaries[:, 0]
        # ml = np.all(mask_lower, axis=1)
        # if np.any(ml):
        #     mask[ml, 0] = True

        return np.argmax(mask, axis=1)

    def inverse_transform(self, bin_indices: np.ndarray):
        """
        Transforms bin indices to the original domain. The result contains
        representative values for the bins, usually the bin centers.

        Args:
            bin_indices (np.array): the bin index array to invert

        Returns:
            np.array: the values representatives for the bins
        """
        return self._bin_boundaries[bin_indices, 1]

    def lookup_bin_boundaries(self, bin_indices: np.ndarray):
        """
        Returns the bin boundaries to the bin indices

        Args:
            bin_indices (np.array): the bin indices

        Returns:
            np.array: the bin boundaries
        """
        return self._bin_boundaries[bin_indices]

    def bin_representatives(self):
        """
        Return representative values from the original domain for each bin index

        Returns:
            np.array: the representatives
        """
        return self._bin_boundaries[:, 1]

    def lookup_bin_representatives(self, values):
        """
        For each value return the corresponding bin's representative value (upper limit).

        Args:
            values (np.array): the array of upper values

        Returns:
            np.array: the representatives
        """
        bin_indices = self.transform(values)
        return self.inverse_transform(bin_indices)

    @staticmethod
    def get_weights_by_values(observed_values, weights):
        '''
        Returns the weights by observed values sorted by the latter.

        Args:
            observed_values (np.array): the observed value vector
            weights (np.array): the observed weight vector

        Returns:
            np.array: the two dimensional array of weights by values
        '''
        weights_by_values = []
        for val in np.unique(observed_values):
            weight = np.sum(weights[observed_values == val])
            weights_by_values.append([val, weight])

        value_weight = np.array(weights_by_values)

        return value_weight[np.argsort(value_weight[:, 0])]

    def get_closest_available_upper_bound(self, value, values_weights, base_weight):
        """
        Return the smallest value with enough cumulative weights.

        Args:
            value (int): the value the closest to which is being searched for
            values_weights (np.array): the two dimensional array of weights by values

        Returns:
            tuple: the closest value to extend to in order to have enough weight
                    and the new weight
        """

        applicable_subset = values_weights[
            (values_weights[:, 0] >= value)
            & (values_weights[:, 0] - value <= self.value_distance_tolerance)
        ]

        for r in range(1, applicable_subset.shape[0] + 1):
            additional_weight = np.sum(applicable_subset[:r, 1])
            weight = base_weight + additional_weight
            if weight >= self.min_weight and r < applicable_subset.shape[0]:
                # we need the value from the next row to get a new open upper bound
                return applicable_subset[r, 0], weight

        return None, base_weight

    def adjust_bin_boundaries(
        self, bin_boundaries: np.array, observed_values, weights, bins_below_threshold
    ):
        """
        Adjust bin_boundaries to include the minimum required weight.

        Args:
            bin_boundaries (np.array): the bin boundaries to be adjusted
            observed_values (np.array): the array of observations to fit the bin boundaries
            weights (np.array): the array of weight observations
            bins_below_threshold (str): how to handle bins that after the adjustment
                                        have weights below the threshold. It should be
                                        one of ['keep', 'remove', 'merge'].

        Returns:
            np.array: the new bin boundaries
        """
        # the resulting zero width bins are removed
        # currently it only considers adjusting to values observed in the data, so it doesn't interpolate

        bins_below_threshold_options = ['keep', 'remove', 'merge']
        if bins_below_threshold not in bins_below_threshold_options:
            raise ValueError(
                f'bins_below_threshold=={bins_below_threshold}, it should be one of {bins_below_threshold_options}'
            )

        values_weights = self.get_weights_by_values(observed_values, weights)
        updated_bin_boundaries = self.update_bins(
            values_weights, bin_boundaries, bins_below_threshold
        )

        return updated_bin_boundaries

    def update_bins(self, values_weights, bin_boundaries, bins_below_threshold):
        """
        Calculates the new bin_boundaries.

        Args:
            bin_boundaries (np.array): the bin boundaries to be adjusted
            values_weights (np.array): the sum of weights for each value observed in the data
            bins_below_threshold (str): how to handle bins that after the adjustment
                                        have weights below the threshold. It should be
                                        one of ['keep', 'remove', 'merge'].

        Returns:
            np.array: the new bin boundaries
        """

        new_bin_boundaries = []
        bin_weights = []
        tmp_bin_boundaries = bin_boundaries.copy()

        while 0 < len(tmp_bin_boundaries):
            current_bin = tmp_bin_boundaries[0].copy()

            if len(tmp_bin_boundaries) > 1:
                tmp_bin_boundaries = tmp_bin_boundaries[1:]
            else:
                tmp_bin_boundaries = np.empty((0, 0))

            lb, ub = current_bin[0], current_bin[1]

            bin_subset = values_weights[
                (values_weights[:, 0] < ub) & (values_weights[:, 0] >= lb)
            ]
            bin_weight = np.sum(bin_subset[:, 1])

            if bin_weight < self.min_weight:
                closest_value, bin_weight = self.get_closest_available_upper_bound(
                    ub, values_weights, bin_weight
                )

                if closest_value is not None:
                    current_bin = [lb, closest_value]
                    if len(tmp_bin_boundaries) > 0:
                        tmp_bin_boundaries = tmp_bin_boundaries[
                            tmp_bin_boundaries[:, 1] > closest_value
                        ]

                    if len(tmp_bin_boundaries) > 0:
                        tmp_bin_boundaries[0, 0] = closest_value

                elif bins_below_threshold == 'remove':
                    current_bin = []

            if len(current_bin) > 0:
                new_bin_boundaries.append(current_bin)
                bin_weights.append(bin_weight)

        updated_bin_boundaries = np.array(new_bin_boundaries)
        bin_weights = np.array(bin_weights)

        bin_widths = updated_bin_boundaries[:, 1] - updated_bin_boundaries[:, 0]
        updated_bin_boundaries = updated_bin_boundaries[bin_widths > 0]
        bin_weights = bin_weights[bin_widths > 0]

        if bins_below_threshold == 'merge':
            updated_bin_boundaries = self.merge_bins(
                updated_bin_boundaries, bin_weights
            )

        return updated_bin_boundaries

    def merge_bins(self, updated_bin_boundaries, bin_weights):
        """
        Merges the bins that don't have enough weight with the bin below that do.
        The first bin is merged with the next one having enough weight.
        If none of the bins have enough weight, there will be only one large final bin.

        Args:
            bin_boundaries (np.array): the bin boundaries to be adjusted
            bin_weights (np.array): the sum of weights for each bin

        Returns:
            np.array: the new bin boundaries
        """

        merged_bin_boundaries = []
        first_bin_lb = None

        for i, b in enumerate(updated_bin_boundaries):
            lb, ub = b[0], b[1]
            bin_weight = bin_weights[i]

            if bin_weight < self.min_weight:
                if i == 0:
                    first_bin_lb = lb
                elif len(merged_bin_boundaries) > 0:
                    merged_bin_boundaries[-1][1] = ub
                # if there is only one bin and it has not enough weight
                if (i == len(updated_bin_boundaries) - 1) and (
                    len(merged_bin_boundaries) == 0
                ):
                    b[0] = first_bin_lb
                    merged_bin_boundaries.append(b)
            else:
                if first_bin_lb is not None:
                    b[0] = first_bin_lb
                    first_bin_lb = None
                merged_bin_boundaries.append(b)
        updated_bin_boundaries = np.array(merged_bin_boundaries)

        return updated_bin_boundaries
