"""Lexical availability analysis: type statistics, co-occurrence graphs and inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("glaurlex")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
