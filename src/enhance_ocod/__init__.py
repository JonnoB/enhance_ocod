"""
A package for enhancing and processing OCOD (Overseas companies that own property in England and Wales ) dataset
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("enhance-ocod")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"
