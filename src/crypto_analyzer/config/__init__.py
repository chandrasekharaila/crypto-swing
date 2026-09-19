"""Application configuration."""

from crypto_analyzer.config.settings import AppSettings
from crypto_analyzer.config.loader import load_settings

__all__ = ["AppSettings", "load_settings"]
