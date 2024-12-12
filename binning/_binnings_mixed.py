"""
This module implements applying multiply types of binnings to a 2D dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _binning

from sklearn.base import TransformerMixin, BaseEstimator

__all__ = [
    "BinningsMixed",
    "BinningsMultiple",
]

class BinningsMixed(TransformerMixin, BaseEstimator):
    """
    Class to apply differently parametrized 1D binnings to different columns of a dataset
    """
    def __init__(self, *, binning_definition_tuple_list: list):
        all_columns = []
        for tup in binning_definition_tuple_list:
            all_columns.extend(tup[2])
        if len(set(all_columns)) != len(all_columns):
            raise ValueError("Duplicated columns passed")

        self.binning_map = {}
        for binning_definition_tuple in binning_definition_tuple_list.items():
            (binning_class_name, binning_params, column_list) = binning_definition_tuple
            binning_class = getattr(_binning, binning_class_name)
            for col in column_list:
                binning_instance = binning_class(**binning_params)
                self.binning_map[col] = binning_instance

    def fit(self, X: np.ndarray | pd.DataFrame, y=None, **fit_params):
        for col, binning_instance in self.binning_map.items():
            if isinstance(X, pd.DataFrame) and col in X.columns:
                binning_instance.fit(X[col], y, **fit_params)
        return self

    def _transform_internal(self, X: np.ndarray | pd.DataFrame, func_name: str):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        for col, binning_instance in self.binning_map.items():
            X[col] = getattr(binning_instance, func_name)(X[col])
        return X

    def transform(self, X: np.ndarray | pd.DataFrame):
        return self._transform_internal(X, "transform")

    def inverse_transform(self, X: np.ndarray | pd.DataFrame):
        return self._transform_internal(X, "inverse_transform")

class BinningsMultiple(BinningsMixed):
    """
    Class to apply the same 1D binning to multiple columns of a dataset
    """
    def __init__(self, *, binning_class_name, binning_params, column_list):
        binning_definition_tuple_list = [(binning_class_name, binning_params, column_list)]
        super().__init__(binning_definition_tuple_list=binning_definition_tuple_list)
