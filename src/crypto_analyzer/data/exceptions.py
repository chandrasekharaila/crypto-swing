"""Explicit exception categories for the data foundation."""


class DataFoundationError(Exception):
    """Base class for expected data-foundation failures."""


class ConfigurationError(DataFoundationError):
    """Raised when runtime configuration is unusable."""


class DataCollectionError(DataFoundationError):
    """Raised when market data cannot be collected."""


class DataValidationError(DataFoundationError):
    """Raised when market data violates the expected schema or invariants."""


class DataStorageError(DataFoundationError):
    """Raised when market data cannot be read or persisted safely."""
