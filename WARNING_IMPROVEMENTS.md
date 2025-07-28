# Warning Message Improvements Summary

## Issue Identified
The user reported seeing a confusing warning when fitting EqualWidthBinning with numpy arrays, where the warning mentioned "column names" but the data was fitted with a numpy array (which uses integer column indices).

## Root Cause
When fitting with numpy arrays, column identifiers are integers (0, 1, 2, etc.), but the warning messages were formatted as:
```
"Column 1 contains only NaN values"
```

This could be confusing because:
- It's unclear whether "1" refers to column index 1 or a column named "1"
- The phrasing "column names" in the context doesn't match the actual integer identifiers

## Solutions Implemented

### 1. Improved Warning Messages
Updated warning messages in both `_interval_binning_base.py` and `_supervised_binning_base.py` to be more descriptive:

**Before:**
```python
warnings.warn(f"Column {col} contains only NaN values", DataQualityWarning)
```

**After:**
```python
# Create a more descriptive column reference
if isinstance(col, (int, np.integer)):
    col_ref = f"column {col} (index {i})"
else:
    col_ref = f"column '{col}'"
warnings.warn(f"Data in {col_ref} contains only NaN values", DataQualityWarning)
```

**Result:**
- For numpy arrays: "Data in column 1 (index 1) contains only NaN values"
- For DataFrames: "Data in column 'feature_name' contains only NaN values"

### 2. Fixed EqualWidthBinning NaN Handling
The original issue also revealed that EqualWidthBinning couldn't handle columns with all NaN values properly. Fixed this by:

```python
def _get_data_range(self, x_col: np.ndarray, col_id: Any) -> Tuple[float, float]:
    """Get the data range for a column."""
    # Check if all values are NaN
    if np.all(np.isnan(x_col)):
        # Create a default range for all-NaN columns
        return 0.0, 1.0
    
    # ... rest of the method
```

## Benefits

1. **Clearer Warnings**: Users can now easily distinguish between:
   - Column indices (for numpy arrays): "column 1 (index 1)"  
   - Column names (for DataFrames): "column 'feature_name'"

2. **Better Robustness**: EqualWidthBinning no longer crashes when encountering all-NaN columns

3. **Consistent Behavior**: The warning system now works seamlessly with both numpy arrays and pandas DataFrames

## Test Results
```python
# Before fix: Would crash with ValueError
# After fix: Works correctly with clear warning
X = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, np.nan]])
ewb = EqualWidthBinning(n_bins=2)
ewb.fit(X)  # Now succeeds with clear warning message
```

The warning now shows: "Data in column 1 (index 1) contains only NaN values" making it crystal clear that this refers to the second column (index 1) of the numpy array.

## Files Modified
- `binning/base/_interval_binning_base.py` - Improved warning messages
- `binning/base/_supervised_binning_base.py` - Improved warning messages  
- `binning/methods/_equal_width_binning.py` - Added all-NaN column handling

The warning system is now much more user-friendly and informative!
