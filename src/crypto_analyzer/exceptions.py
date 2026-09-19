"""Project-wide exception root.

Every expected failure raised by the package derives from ``AnalyzerError``, so a
caller can separate a handled domain failure from a programming error with one
except clause. The per-component subclasses keep the categories differentiated:
network, collection, validation, storage, configuration, feature, and regime
failures remain distinguishable.
"""


class AnalyzerError(Exception):
    """Base class for every expected failure in the package."""
