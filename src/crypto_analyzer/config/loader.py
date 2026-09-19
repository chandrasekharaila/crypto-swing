"""Load typed application settings from optional JSON configuration."""

from pathlib import Path

from pydantic import ValidationError

from crypto_analyzer.config.settings import AppSettings
from crypto_analyzer.data.exceptions import ConfigurationError


def load_settings(path: Path | None = None) -> AppSettings:
    """Load settings from JSON, or return validated defaults when omitted."""
    if path is None:
        return AppSettings()
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"could not read configuration file: {path}") from error
    try:
        return AppSettings.model_validate_json(content)
    except ValidationError as error:
        raise ConfigurationError(f"invalid configuration file {path}: {error}") from error
