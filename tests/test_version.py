"""Test version module directly."""

import sys
import importlib.util


def test_version_import():
    """Test version module can be imported and has version."""
    spec = importlib.util.spec_from_file_location(
        "binning._version",
        "/home/gykovacs/workspaces/binning/binning/_version.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        assert hasattr(module, '__version__')
        assert module.__version__ == "0.0.1"
