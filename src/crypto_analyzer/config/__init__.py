"""Application configuration."""

from crypto_analyzer.config.loader import load_settings
from crypto_analyzer.config.settings import AppSettings
from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.regimes.config import RegimeSettings
from crypto_analyzer.setups.config import SetupSettings

__all__ = [
    "AppSettings",
    "FeatureSettings",
    "RegimeSettings",
    "SetupSettings",
    "load_settings",
]
