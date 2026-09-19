"""Application configuration."""

from crypto_analyzer.config.loader import load_settings
from crypto_analyzer.config.settings import AppSettings

__all__ = ["AppSettings", "load_settings"]
