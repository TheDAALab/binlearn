try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pl = None
    POLARS_AVAILABLE = False
