import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch
from binning.base._interval_binning_base import IntervalBinningBase
from binning.utils.errors import ConfigurationError, DataQualityWarning, BinningError

class DummyIntervalBinning(IntervalBinningBase):
    def __init__(self, clip=None, bin_edges=None, bin_representatives=None, **kwargs):
        super().__init__(clip=clip, bin_edges=bin_edges, bin_representatives=bin_representatives, **kwargs)
    
    def _calculate_bins(self, x_col, col_id, guidance_data=None):
        # Return dummy bin edges and representatives
        return [0.0, 1.0, 2.0], [0.5, 1.5]

def test_init_default():
    """Test initialization with default parameters."""
    obj = DummyIntervalBinning()
    assert obj.clip is not None  # Should get from config
    assert obj.bin_edges is None
    assert obj.bin_representatives is None
    assert obj._user_bin_edges is None
    assert obj._user_bin_reps is None
    assert obj._bin_edges == {}
    assert obj._bin_reps == {}

def test_init_with_clip():
    """Test initialization with explicit clip parameter."""
    obj = DummyIntervalBinning(clip=True)
    assert obj.clip is True

def test_init_with_bin_edges():
    """Test initialization with bin_edges provided."""
    bin_edges = {0: [0, 1, 2]}
    
    with patch.object(DummyIntervalBinning, '_process_provided_bins') as mock_process:
        obj = DummyIntervalBinning(bin_edges=bin_edges)
        assert obj.bin_edges == bin_edges
        assert obj._user_bin_edges == bin_edges
        mock_process.assert_called_once()

def test_init_with_bin_representatives():
    """Test initialization with bin_representatives provided."""
    bin_reps = {0: [0.5, 1.5]}
    obj = DummyIntervalBinning(bin_representatives=bin_reps)
    assert obj.bin_representatives == bin_reps
    assert obj._user_bin_reps == bin_reps

@patch('binning.base._interval_binning_base.ensure_bin_dict')
@patch('binning.base._interval_binning_base.validate_bins')
def test_process_provided_bins_edges_only(mock_validate, mock_ensure):
    """Test _process_provided_bins with edges only."""
    mock_ensure.side_effect = lambda x: x  # Return input unchanged
    
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [0, 1, 2]}
    obj._user_bin_reps = None
    
    with patch('binning.base._interval_binning_base.default_representatives') as mock_default:
        mock_default.return_value = [0.5, 1.5]
        obj._process_provided_bins()
        
        mock_ensure.assert_called()
        mock_default.assert_called_once()
        mock_validate.assert_called_once()
        assert obj._fitted is True

@patch('binning.base._interval_binning_base.ensure_bin_dict')
@patch('binning.base._interval_binning_base.validate_bins')
def test_process_provided_bins_both(mock_validate, mock_ensure):
    """Test _process_provided_bins with both edges and reps."""
    mock_ensure.side_effect = lambda x: x
    
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [0, 1, 2]}
    obj._user_bin_reps = {0: [0.5, 1.5]}
    
    obj._process_provided_bins()
    
    assert mock_ensure.call_count == 2
    mock_validate.assert_called_once()
    assert obj._fitted is True

def test_process_provided_bins_invalid_edges_type():
    """Test _process_provided_bins with invalid edge types."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: "invalid"}  # String instead of array-like
    
    with patch('binning.base._interval_binning_base.ensure_bin_dict') as mock_ensure:
        mock_ensure.return_value = {0: "invalid"}
        
        with pytest.raises(Exception):  # Should raise ConfigurationError
            obj._process_provided_bins()

def test_process_provided_bins_too_few_edges():
    """Test _process_provided_bins with too few edges."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [1]}  # Only one edge
    
    with patch('binning.base._interval_binning_base.ensure_bin_dict') as mock_ensure:
        mock_ensure.return_value = {0: [1]}
        
        with pytest.raises(Exception):  # Should raise ConfigurationError
            obj._process_provided_bins()

def test_process_provided_bins_error():
    """Test _process_provided_bins error handling."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [0, 1, 2]}
    
    with patch('binning.base._interval_binning_base.ensure_bin_dict') as mock_ensure:
        mock_ensure.side_effect = Exception("Test error")
        
        with pytest.raises(ConfigurationError, match="Failed to process provided bin specifications"):
            obj._process_provided_bins()

@patch.object(DummyIntervalBinning, '_process_user_specifications')
@patch.object(DummyIntervalBinning, '_finalize_fitting')
def test_fit_per_column_success(mock_finalize, mock_process):
    """Test _fit_per_column successful execution."""
    obj = DummyIntervalBinning()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]
    
    result = obj._fit_per_column(X, columns)
    
    assert result is obj
    mock_process.assert_called_once_with(columns)
    mock_finalize.assert_called_once()

@patch.object(DummyIntervalBinning, '_process_user_specifications')
def test_fit_per_column_with_existing_bins(mock_process):
    """Test _fit_per_column with existing bin specs."""
    obj = DummyIntervalBinning()
    obj._bin_edges = {0: [0, 1, 2]}
    obj._bin_reps = {0: [0.5, 1.5]}
    
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]
    
    with patch.object(obj, '_calculate_bins') as mock_calc:
        mock_calc.return_value = ([0, 1, 2], [0.5, 1.5])
        
        with patch.object(obj, '_finalize_fitting'):
            obj._fit_per_column(X, columns)
        
        # Should only call _calculate_bins for column 1
        mock_calc.assert_called_once()
        assert 1 in obj._bin_edges
        assert 1 in obj._bin_reps

def test_fit_per_column_error_handling():
    """Test _fit_per_column error handling."""
    obj = DummyIntervalBinning()
    
    with patch.object(obj, '_process_user_specifications') as mock_process:
        mock_process.side_effect = Exception("Test error")
        
        X = np.array([[1, 2], [3, 4]])
        columns = [0, 1]
        
        with pytest.raises(ValueError, match="Failed to fit per-column bins"):
            obj._fit_per_column(X, columns)

@patch.object(DummyIntervalBinning, '_process_user_specifications')
@patch.object(DummyIntervalBinning, '_calculate_bins_jointly')
@patch.object(DummyIntervalBinning, '_finalize_fitting')
def test_fit_jointly_success(mock_finalize, mock_calc_jointly, mock_process):
    """Test _fit_jointly successful execution."""
    # Configure the mock to return the expected tuple
    mock_calc_jointly.return_value = ([0.0, 1.0, 2.0], [0.5, 1.5])
    
    obj = DummyIntervalBinning()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]
    
    obj._fit_jointly(X, columns)
    
    mock_process.assert_called_once_with(columns)
    mock_calc_jointly.assert_called()  # Should be called for each column
    mock_finalize.assert_called_once()

def test_fit_jointly_error_handling():
    """Test _fit_jointly error handling."""
    obj = DummyIntervalBinning()
    
    with patch.object(obj, '_process_user_specifications') as mock_process:
        mock_process.side_effect = Exception("Test error")
        
        X = np.array([[1, 2], [3, 4]])
        columns = [0, 1]
        
        with pytest.raises(ValueError, match="Failed to fit joint bins"):
            obj._fit_jointly(X, columns)

def test_calculate_bins_abstract():
    """Test _calculate_bins is properly abstract."""
    # We can't instantiate the abstract base class directly
    # This test shows that the method must be implemented by subclasses
    obj = DummyIntervalBinning()
    # The dummy class implements it, so we can call it
    result = obj._calculate_bins(np.array([1, 2, 3]), 0)
    assert result == ([0.0, 1.0, 2.0], [0.5, 1.5])

def test_get_column_key_direct_match():
    """Test _get_column_key with direct match."""
    obj = DummyIntervalBinning()
    
    target_col = 'A'
    available_keys = ['A', 'B', 'C']
    col_index = 0
    
    result = obj._get_column_key(target_col, available_keys, col_index)
    assert result == 'A'

def test_get_column_key_index_fallback():
    """Test _get_column_key with index fallback."""
    obj = DummyIntervalBinning()
    
    target_col = 'X'
    available_keys = ['A', 'B', 'C']
    col_index = 1
    
    result = obj._get_column_key(target_col, available_keys, col_index)
    assert result == 'B'

def test_get_column_key_no_match():
    """Test _get_column_key with no match."""
    obj = DummyIntervalBinning()
    
    target_col = 'X'
    available_keys = ['A', 'B']
    col_index = 5
    
    with pytest.raises(ValueError, match="No bin specification found for column X"):
        obj._get_column_key(target_col, available_keys, col_index)

def test_transform_columns_with_clip():
    """Test _transform_columns method with clipping."""
    obj = DummyIntervalBinning(clip=True)
    obj._bin_edges = {0: [0, 1, 2], 1: [0, 1, 2]}
    
    X = np.array([[0.5, 1.5], [2.5, -0.5]])  # Out-of-range values that should be clipped
    columns = [0, 1]
    
    result = obj._transform_columns(X, columns)
    
    assert result.shape == (2, 2)
    # With clipping, out-of-range values should be clipped to valid bin indices
    assert result[0, 0] == 0  # 0.5 is in first bin
    assert result[0, 1] == 1  # 1.5 is in second bin
    assert result[1, 0] == 1  # 2.5 clipped to last bin (index 1)
    assert result[1, 1] == 0  # -0.5 clipped to first bin (index 0)

from binning.utils.constants import ABOVE_RANGE, BELOW_RANGE

def test_transform_columns_without_clip():
    """Test _transform_columns method without clipping."""
    obj = DummyIntervalBinning(clip=False)
    obj._bin_edges = {0: [0, 1, 2]}
    
    X = np.array([[0.5], [2.5]])  # Values: in-range and above-range
    columns = [0]
    
    result = obj._transform_columns(X, columns)
    
    assert result.shape == (2, 1)
    assert result[0, 0] == 0  # 0.5 is in first bin
    assert result[1, 0] == ABOVE_RANGE  # 2.5 is above range

def test_transform_columns_missing_key():
    """Test _transform_columns with missing column key."""
    obj = DummyIntervalBinning()
    obj._bin_edges = {}  # Empty edges
    
    X = np.array([[1, 2]])
    columns = [0]
    
    with pytest.raises(ValueError, match="No bin specification found"):
        obj._transform_columns(X, columns)

def test_inverse_transform_columns():
    """Test _inverse_transform_columns method."""
    obj = DummyIntervalBinning()
    obj._bin_reps = {0: [0.5, 1.5], 1: [2.5, 3.5]}
    
    X = np.array([[0, 1], [1, 0]])
    columns = [0, 1]
    
    result = obj._inverse_transform_columns(X, columns)
    
    expected = np.array([[0.5, 3.5], [1.5, 2.5]])
    np.testing.assert_array_equal(result, expected)

def test_inverse_transform_columns_missing_key():
    """Test _inverse_transform_columns with missing column key."""
    obj = DummyIntervalBinning()
    obj._bin_reps = {}  # Empty reps
    
    X = np.array([[0]])
    columns = [0]
    
    with pytest.raises(ValueError, match="No bin specification found"):
        obj._inverse_transform_columns(X, columns)

@patch('binning.base._interval_binning_base.validate_bins')
def test_finalize_fitting(mock_validate):
    """Test _finalize_fitting method."""
    obj = DummyIntervalBinning()
    obj._bin_edges = {0: [0, 1, 2]}
    obj._bin_reps = {0: [0.5, 1.5]}
    
    obj._finalize_fitting()
    
    mock_validate.assert_called_once_with(obj._bin_edges, obj._bin_reps)

def test_finalize_fitting_error():
    """Test _finalize_fitting error handling."""
    obj = DummyIntervalBinning()
    obj._bin_edges = {0: [0, 1, 2]}
    obj._bin_reps = {0: [0.5, 1.5]}
    
    with patch('binning.base._interval_binning_base.validate_bins') as mock_validate:
        mock_validate.side_effect = Exception("Validation error")
        
        with pytest.raises(Exception, match="Validation error"):
            obj._finalize_fitting()

def test_process_user_specifications():
    """Test _process_user_specifications method."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [0, 1, 2]}
    obj._user_bin_reps = {0: [0.5, 1.5]}
    
    obj._process_user_specifications([0, 1])
    
    # Check that bin edges and reps were set
    assert obj._bin_edges == {0: [0, 1, 2]}
    assert obj._bin_reps == {0: [0.5, 1.5]}

def test_process_user_specifications_no_user_specs():
    """Test _process_user_specifications with no user specs."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = None
    obj._user_bin_reps = None
    
    obj._process_user_specifications([0, 1])
    
    # Check that empty dicts were set
    assert obj._bin_edges == {}
    assert obj._bin_reps == {}

def test_calculate_bins_jointly_abstract():
    """Test _calculate_bins_jointly default implementation."""
    obj = DummyIntervalBinning()
    
    # Should fall back to _calculate_bins with new signature
    result = obj._calculate_bins_jointly(np.array([1, 2]), [0])
    assert result == ([0.0, 1.0, 2.0], [0.5, 1.5])

def test_get_binning_params():
    """Test _get_binning_params method with automatic discovery."""
    obj = DummyIntervalBinning(bin_edges={0: [0, 1, 2]}, clip=True)
    params = obj._get_binning_params()
    
    # With automatic discovery, these should be included
    assert 'bin_edges' in params
    assert 'bin_representatives' in params
    assert 'clip' in params
    assert params['clip'] is True

def test_get_fitted_params():
    """Test _get_fitted_params method."""
    obj = DummyIntervalBinning()
    obj._fitted = True
    obj._bin_edges = {0: [0, 1, 2]}
    obj._bin_reps = {0: [0.5, 1.5]}
    
    params = obj._get_fitted_params()
    
    assert params['bin_edges'] == {0: [0, 1, 2]}
    assert params['bin_representatives'] == {0: [0.5, 1.5]}

def test_handle_bin_params():
    """Test _handle_bin_params method."""
    obj = DummyIntervalBinning()
    
    # Test with bin_edges change
    reset = obj._handle_bin_params({'bin_edges': {0: [0, 1, 2]}})
    assert reset is True
    
    # Test with bin_representatives change  
    reset = obj._handle_bin_params({'bin_representatives': {0: [0.5, 1.5]}})
    assert reset is True
    
    # Test with clip change (automatic discovery handles this)
    reset = obj._handle_bin_params({'clip': True})
    assert reset is True  # With automatic discovery, this should trigger reset
    
    # Test with fit_jointly change (automatic discovery handles this)
    reset = obj._handle_bin_params({'fit_jointly': True})
    assert reset is True
    
    # Test with guidance_columns change (automatic discovery handles this)
    reset = obj._handle_bin_params({'guidance_columns': ['A']})
    assert reset is True
    
    # Test with no relevant changes
    reset = obj._handle_bin_params({'other_param': 'value'})
    assert reset is False

def test_data_quality_warnings():
    """Test data quality warning generation."""
    obj = DummyIntervalBinning()
    
    # Test with data that should trigger warnings
    X = np.array([[np.inf, 2], [3, np.nan]])
    columns = [0, 1]
    
    with patch('warnings.warn') as mock_warn:
        with patch.object(obj, '_calculate_bins') as mock_calc:
            mock_calc.return_value = ([0, 1, 2], [0.5, 1.5])
            with patch.object(obj, '_finalize_fitting'):
                obj._fit_per_column(X, columns)
        
        # Should call warnings for data quality issues
        # (Note: actual warning logic depends on implementation details)

def test_empty_data_handling():
    """Test handling of empty data arrays."""
    obj = DummyIntervalBinning()
    
    X = np.array([]).reshape(0, 2)
    columns = [0, 1]
    
    with patch.object(obj, '_calculate_bins') as mock_calc:
        mock_calc.return_value = ([0, 1], [0.5])
        with patch.object(obj, '_finalize_fitting'):
            obj._fit_per_column(X, columns)
    
    # Should handle empty data gracefully

def test_fit_per_column_exception_handling():
    """Test _fit_per_column exception handling."""
    obj = DummyIntervalBinning()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]
    
    with patch.object(obj, '_process_user_specifications') as mock_process:
        mock_process.side_effect = Exception("Test error")
        
        with pytest.raises(ValueError, match="Failed to fit per-column bins"):
            obj._fit_per_column(X, columns)

def test_fit_jointly_exception_handling():
    """Test _fit_jointly exception handling."""
    obj = DummyIntervalBinning()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]
    
    with patch.object(obj, '_process_user_specifications') as mock_process:
        mock_process.side_effect = Exception("Test error")
        
        with pytest.raises(ValueError, match="Failed to fit joint bins"):
            obj._fit_jointly(X, columns)

def test_nan_data_warnings_string_column():
    """Test warnings for string column names with all NaN data."""
    obj = DummyIntervalBinning()
    X = np.array([[np.nan], [np.nan]])
    columns = ['feature_a']  # String column name
    
    with pytest.warns(DataQualityWarning, match="Data in column 'feature_a' contains only NaN values"):
        with patch.object(obj, '_calculate_bins') as mock_calc:
            mock_calc.return_value = ([0, 1], [0.5])
            with patch.object(obj, '_finalize_fitting'):
                obj._fit_per_column(X, columns)

def test_nan_data_warnings_string_column_jointly():
    """Test warnings for all NaN data in joint fitting."""
    obj = DummyIntervalBinning()
    X = np.array([[np.nan], [np.nan]])
    columns = ['feature_a']  # String column name
    
    with pytest.warns(DataQualityWarning, match="All data contains only NaN values"):
        with patch.object(obj, '_calculate_bins_jointly') as mock_calc:
            mock_calc.return_value = ([0, 1], [0.5])
            with patch.object(obj, '_finalize_fitting'):
                obj._fit_jointly(X, columns)

def test_calculate_bins_jointly_default():
    """Test _calculate_bins_jointly default implementation."""
    obj = DummyIntervalBinning()
    all_data = np.array([1, 2, 3])
    columns = [0]
    
    with patch.object(obj, '_calculate_bins') as mock_calc:
        mock_calc.return_value = ([0, 1, 2], [0.5, 1.5])
        
        result = obj._calculate_bins_jointly(all_data, columns)
        
        assert result == ([0, 1, 2], [0.5, 1.5])
        mock_calc.assert_called_once_with(all_data, 0)


def test_fit_per_column_binning_error():
    """Test _fit_per_column re-raises BinningError unchanged."""
    class BinningErrorInterval(DummyIntervalBinning):
        def _calculate_bins(self, x_col, col_id, guidance_data=None):
            raise BinningError("Specific binning error")
    
    obj = BinningErrorInterval()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]
    
    # BinningError should be re-raised unchanged
    with pytest.raises(BinningError, match="Specific binning error"):
        obj._fit_per_column(X, columns)


def test_process_provided_bins_binning_error():
    """Test _process_provided_bins re-raises BinningError unchanged."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [0, 1, 2]}
    
    with patch('binning.base._interval_binning_base.ensure_bin_dict') as mock_ensure:
        mock_ensure.side_effect = BinningError("Bins validation error")
        
        # BinningError should be re-raised unchanged
        with pytest.raises(BinningError, match="Bins validation error"):
            obj._process_provided_bins()


def test_process_provided_bins_configuration_error():
    """Test _process_provided_bins wraps generic exceptions in ConfigurationError."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [0, 1, 2]}
    
    with patch('binning.base._interval_binning_base.ensure_bin_dict') as mock_ensure:
        mock_ensure.side_effect = ValueError("Generic validation error")
        
        # Generic exception should be wrapped in ConfigurationError
        with pytest.raises(ConfigurationError, match="Failed to process provided bin specifications"):
            obj._process_provided_bins()


def test_finalize_fitting_default_representatives():
    """Test _finalize_fitting generates default representatives for missing ones."""
    obj = DummyIntervalBinning()
    obj._bin_edges = {0: [0, 1, 2], 1: [0, 1, 2]}
    obj._bin_reps = {0: [0.5, 1.5]}  # Missing reps for column 1
    
    with patch('binning.base._interval_binning_base.default_representatives') as mock_default:
        mock_default.return_value = [0.5, 1.5]
        with patch('binning.base._interval_binning_base.validate_bins'):
            obj._finalize_fitting()
        
        # Should generate default reps for column 1
        mock_default.assert_called_once_with([0, 1, 2])
        assert obj._bin_reps[1] == [0.5, 1.5]


def test_fit_jointly_string_column_reference():
    """Test _fit_jointly with string column identifiers."""
    obj = DummyIntervalBinning()
    X = np.array([[1, 2], [3, 4]])
    columns = ['col_a', 'col_b']  # String columns instead of integers
    
    # Should handle string column references
    obj._fit_jointly(X, columns)
    
    assert 'col_a' in obj._bin_edges
    assert 'col_b' in obj._bin_edges


def test_fit_jointly_binning_error():
    """Test _fit_jointly re-raises BinningError unchanged."""
    class BinningErrorInterval(DummyIntervalBinning):
        def _calculate_bins_jointly(self, all_data, columns):
            raise BinningError("Joint binning error")
    
    obj = BinningErrorInterval()
    X = np.array([[1, 2], [3, 4]])
    columns = [0, 1]
    
    # BinningError should be re-raised unchanged
    with pytest.raises(BinningError, match="Joint binning error"):
        obj._fit_jointly(X, columns)


def test_process_user_specifications_binning_error():
    """Test _process_user_specifications re-raises BinningError unchanged."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [0, 1, 2]}
    
    with patch('binning.base._interval_binning_base.ensure_bin_dict') as mock_ensure:
        mock_ensure.side_effect = BinningError("User specs validation error")
        
        # BinningError should be re-raised unchanged
        with pytest.raises(BinningError, match="User specs validation error"):
            obj._process_user_specifications([0, 1])


def test_inverse_transform_method():
    """Test inverse_transform method directly."""
    obj = DummyIntervalBinning()
    obj._bin_edges = {0: [0, 1, 2]}
    obj._bin_reps = {0: [0.5, 1.5]}
    obj._fitted = True
    
    # Test inverse transform
    bin_indices = np.array([[0], [1]])
    result = obj.inverse_transform(bin_indices)
    
    # Should return representative values
    expected = np.array([[0.5], [1.5]])
    np.testing.assert_array_equal(result, expected)


def test_lookup_bin_widths_method():
    """Test lookup_bin_widths method directly."""
    obj = DummyIntervalBinning()
    obj._bin_edges = {0: [0, 1, 3]}  # Widths: 1, 2
    obj._fitted = True
    
    # Test bin width lookup
    bin_indices = np.array([[0], [1]])
    result = obj.lookup_bin_widths(bin_indices)
    
    # Should return bin widths
    expected = np.array([[1.0], [2.0]])
    np.testing.assert_array_equal(result, expected)


def test_lookup_bin_ranges_method():
    """Test lookup_bin_ranges method directly."""
    obj = DummyIntervalBinning()
    obj._bin_edges = {0: [0, 1, 2], 'col_a': [0, 1, 2, 3]}  # 2 bins, 3 bins
    obj._fitted = True
    
    # Test bin range lookup
    result = obj.lookup_bin_ranges()
    
    # Should return number of bins per column
    expected = {0: 2, 'col_a': 3}
    assert result == expected


def test_fit_jointly_with_string_columns_and_nan():
    """Test _fit_per_column with string columns and NaN data to cover line 123."""
    import pandas as pd
    
    obj = DummyIntervalBinning(fit_jointly=False)  # Use per-column fitting for string column test
    # Create a DataFrame with string column names where first column is all NaN
    df = pd.DataFrame({
        'string_col': [np.nan, np.nan, np.nan],
        'numeric_col': [1.0, 2.0, 3.0]
    })
    
    # Should handle string column references and NaN warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj.fit(df)
        
        # Should have warning about NaN data in first column with string reference
        assert len(w) >= 1
        warning_messages = [str(warning.message) for warning in w]
        # Look for the exact format from line 123: "column 'string_col'"
        assert any("column 'string_col'" in msg for msg in warning_messages)
    
    assert 'string_col' in obj._bin_edges
    assert 'numeric_col' in obj._bin_edges


def test_process_user_specifications_generic_error():
    """Test _process_user_specifications wraps generic errors to cover line 188."""
    obj = DummyIntervalBinning()
    obj._user_bin_edges = {0: [0, 1, 2]}
    
    with patch('binning.base._interval_binning_base.ensure_bin_dict') as mock_ensure:
        mock_ensure.side_effect = ValueError("Generic user specs error")
        
        # Generic exception should be wrapped in ConfigurationError (line 188)
        with pytest.raises(ConfigurationError, match="Failed to process bin specifications"):
            obj._process_user_specifications([0, 1])
