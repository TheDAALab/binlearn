import subprocess
import sys


def test_no_pandas_isolated():
    code = """
import sys
import importlib
sys.modules.pop("pandas", None)
import types
sys.modules["pandas"] = None
import binning._binning_base as bb
importlib.reload(bb)
assert bb.PANDAS_AVAILABLE is False
assert bb.pd is None
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
