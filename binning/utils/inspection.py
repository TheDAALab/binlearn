"""
Inspection utilities for introspecting class signatures and parameters.

These utilities help with automatic parameter discovery and handling
across different binning classes.
"""

import inspect
from typing import Any, Dict, List, Optional, Set, Type


def get_class_parameters(
    class_obj: Type,
    exclude_params: Optional[Set[str]] = None,
    exclude_base_class: Optional[str] = None,
) -> List[str]:
    """
    Get parameter names from a class constructor, optionally excluding base class parameters.

    Parameters
    ----------
    class_obj : Type
        The class to inspect
    exclude_params : Set[str], optional
        Parameter names to exclude (default: {'self', 'kwargs'})
    exclude_base_class : str, optional
        Name of base class whose parameters should be excluded

    Returns
    -------
    List[str]
        List of parameter names specific to the class

    Raises
    ------
    ValueError
        If inspect.signature fails due to invalid class definition
    TypeError
        If inspect.signature fails due to type issues
    """
    if exclude_params is None:
        exclude_params = {"self", "kwargs"}

    try:
        current_sig = inspect.signature(class_obj.__init__)
        current_params = set(current_sig.parameters.keys()) - exclude_params
    except (ValueError, TypeError) as exc:
        # Re-raise with context for easier debugging
        raise type(exc)(f"Failed to inspect {class_obj.__name__}.__init__: {str(exc)}") from exc

    if exclude_base_class is None:
        return list(current_params)

    # Find and exclude base class parameters
    for base_class in class_obj.__mro__:
        if base_class.__name__ == exclude_base_class:
            try:
                base_sig = inspect.signature(base_class.__init__)  # type: ignore[misc]
                base_params = set(base_sig.parameters.keys()) - exclude_params
                class_specific_params = list(current_params - base_params)
                return class_specific_params
            except (ValueError, TypeError) as exc:
                # If base class inspection fails, return all current params
                raise type(exc)(
                    f"Failed to inspect {exclude_base_class}.__init__: {str(exc)}"
                ) from exc

    # Base class not found in MRO, return all parameters
    return list(current_params)


def get_constructor_info(class_obj: Type, concrete_only: bool = True) -> Dict[str, Any]:
    """
    Get constructor parameter names and their default values.

    Parameters
    ----------
    class_obj : Type
        The class to inspect
    concrete_only : bool, default=True
        If True, only inspect the concrete class's __init__ method.
        If False, use normal method resolution order.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping parameter names to their default values.
        Uses inspect.Parameter.empty for required parameters.

    Raises
    ------
    ValueError
        If inspect.signature fails due to invalid class definition
    TypeError
        If inspect.signature fails due to type issues
    """
    try:
        if concrete_only and "__init__" in class_obj.__dict__:
            # Use the concrete class's own __init__ method
            sig = inspect.signature(class_obj.__dict__["__init__"])
        else:
            # Use normal method resolution
            sig = inspect.signature(class_obj.__init__)

        params = {}
        for name, param in sig.parameters.items():
            if name in {"self", "kwargs"}:
                continue
            # Get default value if it exists
            if param.default is not inspect.Parameter.empty:
                params[name] = param.default
            else:
                params[name] = inspect.Parameter.empty  # Mark as required parameter
        return params
    except (ValueError, TypeError) as exc:
        # Re-raise with context for easier debugging
        raise type(exc)(
            f"Failed to get constructor info for {class_obj.__name__}: {str(exc)}"
        ) from exc


def safe_get_class_parameters(
    class_obj: Type,
    exclude_params: Optional[Set[str]] = None,
    exclude_base_class: Optional[str] = None,
    fallback: Optional[List[str]] = None,
) -> List[str]:
    """
    Safely get class parameters with fallback on inspection failure.

    This is a safe wrapper around get_class_parameters that catches
    inspection exceptions and returns a fallback value.

    Parameters
    ----------
    class_obj : Type
        The class to inspect
    exclude_params : Set[str], optional
        Parameter names to exclude (default: {'self', 'kwargs'})
    exclude_base_class : str, optional
        Name of base class whose parameters should be excluded
    fallback : List[str], optional
        Value to return if inspection fails (default: empty list)

    Returns
    -------
    List[str]
        List of parameter names, or fallback value if inspection fails
    """
    if fallback is None:
        fallback = []

    try:
        return get_class_parameters(class_obj, exclude_params, exclude_base_class)
    except (ValueError, TypeError):
        return fallback


def safe_get_constructor_info(
    class_obj: Type, concrete_only: bool = True, fallback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safely get constructor info with fallback on inspection failure.

    This is a safe wrapper around get_constructor_info that catches
    inspection exceptions and returns a fallback value.

    Parameters
    ----------
    class_obj : Type
        The class to inspect
    concrete_only : bool, default=True
        If True, only inspect the concrete class's __init__ method
    fallback : Dict[str, Any], optional
        Value to return if inspection fails (default: empty dict)

    Returns
    -------
    Dict[str, Any]
        Constructor parameter info, or fallback value if inspection fails
    """
    if fallback is None:
        fallback = {}

    try:
        return get_constructor_info(class_obj, concrete_only)
    except (ValueError, TypeError):
        return fallback
