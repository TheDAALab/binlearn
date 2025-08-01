"""
Simple representation mixin for binning classes.
"""

from typing import Any, Dict

from ..utils.inspection import safe_get_constructor_info


# pylint: disable=too-few-public-methods
class ReprMixin:
    """
    Simple mixin providing a clean __repr__ method.

    Shows only parameters that are relevant to the specific class,
    determined by inspecting the class's constructor signature.
    """

    def _get_constructor_info(self) -> Dict[str, Any]:
        """Get constructor parameter names and their default values.

        Extracts parameter information from the class constructor signature,
        focusing only on concrete class parameters to avoid inherited attributes
        that may not be relevant for string representation.

        Returns:
            Dict[str, Any]: A dictionary mapping parameter names to their default
                values as defined in the constructor signature. Parameters without
                default values are included with inspect.Parameter.empty as the value.

        Note:
            Uses safe inspection utilities to handle edge cases in parameter
            extraction across different Python versions and class hierarchies.
        """
        return safe_get_constructor_info(self.__class__, concrete_only=True)

    def __repr__(self) -> str:
        """Clean string representation showing only relevant parameters.

        Generates a concise string representation of the object that includes
        only constructor parameters that differ from their default values.
        This provides a clean, readable representation focused on the meaningful
        configuration of the instance.

        The method intelligently handles various parameter types:
        - Skips parameters that match their default values
        - Abbreviates large complex objects (bin_edges, bin_representatives, etc.)
        - Properly quotes string values
        - Excludes None values when they are the default
        - Handles empty containers appropriately

        Returns:
            str: A string representation in the format "ClassName(param1=value1, param2=value2)"
                or "ClassName()" if no parameters differ from defaults.

        Example:
            >>> binning = EqualWidthBinning(n_bins=5, clip=True)
            >>> repr(binning)
            'EqualWidthBinning(n_bins=5, clip=True)'

            >>> default_binning = EqualWidthBinning()
            >>> repr(default_binning)
            'EqualWidthBinning()'
        """
        class_name = self.__class__.__name__

        # Get constructor parameters and their defaults
        constructor_info = self._get_constructor_info()

        # Extract current values for ONLY parameters in the concrete constructor
        parts = []
        for param_name, default_value in constructor_info.items():
            # Only show parameters that are actually in this class's constructor
            # This prevents showing inherited attributes that aren't in the concrete constructor
            if not hasattr(self, param_name):
                continue

            current_value = getattr(self, param_name)

            # Skip if value matches default
            if current_value == default_value:
                continue

            # Skip None values that are defaults
            if current_value is None and default_value is None:
                continue

            # Skip empty containers unless they differ from default
            if current_value in ({}, []) and default_value in (None, {}, []):
                continue

            # Abbreviate large objects
            if param_name in {"bin_edges", "bin_representatives", "bin_spec"}:
                parts.append(f"{param_name}=...")
            elif isinstance(current_value, str):
                parts.append(f"{param_name}='{current_value}'")
            else:
                parts.append(f"{param_name}={current_value}")

        if parts:
            return f"{class_name}({', '.join(parts)})"
        return f"{class_name}()"
