"""
Simple representation mixin for binning classes.
"""


class ReprMixin:
    """
    Simple mixin providing a clean __repr__ method.

    Just shows class name and non-default parameters.
    Large objects (bin_edges, bin_spec, etc.) are abbreviated as "...".
    """

    def __repr__(self) -> str:
        """Clean string representation."""
        class_name = self.__class__.__name__

        # Get parameters if available
        try:
            params = self.get_params(deep=False)  # type: ignore
        except Exception:
            # Fallback: extract common attributes
            params = {}
            for attr in [
                "n_bins",
                "max_unique_values",
                "task_type",
                "tree_params",
                "preserve_dataframe",
                "fit_jointly",
                "guidance_columns",
                "bin_edges",
                "bin_representatives",
                "bin_spec",
                "clip",
            ]:
                if hasattr(self, attr):
                    params[attr] = getattr(self, attr)

        # Simple defaults (most common values)
        defaults = {
            "preserve_dataframe": False,
            "fit_jointly": False,
            "guidance_columns": None,
            "bin_edges": None,
            "bin_representatives": None,
            "bin_spec": None,
            "n_bins": 10,
            "max_unique_values": 100,
            "task_type": "classification",
            "tree_params": {},
            "clip": True,
        }

        # Show only non-default parameters
        parts = []
        for key, value in params.items():
            if key in defaults and value == defaults[key]:
                continue
            if value is None or value == {} or value == []:
                continue

            # Abbreviate large objects
            if key in {"bin_edges", "bin_representatives", "bin_spec"}:
                parts.append(f"{key}=...")
            elif isinstance(value, str):
                parts.append(f"{key}='{value}'")
            else:
                parts.append(f"{key}={value}")

        if parts:
            return f"{class_name}({', '.join(parts)})"
        else:
            return f"{class_name}()"
