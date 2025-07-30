"""
Simple representation mixin for binning classes.
"""

from typing import Dict, Any
from ..utils.inspection import safe_get_constructor_info


# pylint: disable=too-few-public-methods
class ReprMixin:
    """
    Simple mixin providing a clean __repr__ method.

    Shows only parameters that are relevant to the specific class,
    determined by inspecting the class's constructor signature.
    """

    def _get_constructor_info(self) -> Dict[str, Any]:
        """Get constructor parameter names and their default values."""
        return safe_get_constructor_info(self.__class__, concrete_only=True)

    def __repr__(self) -> str:
        """Clean string representation showing only relevant parameters."""
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
