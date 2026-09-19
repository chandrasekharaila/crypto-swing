"""Tests for feature configuration and its integration with AppSettings."""

import json

import pytest
from pydantic import ValidationError

from crypto_analyzer.config import AppSettings, load_settings
from crypto_analyzer.data.exceptions import ConfigurationError
from crypto_analyzer.features import FeatureGroup, FeatureSettings


def test_defaults_enable_every_group() -> None:
    settings = FeatureSettings()

    assert set(settings.enabled_groups) == set(FeatureGroup)
    assert settings.rsi_period == 14


def test_settings_are_immutable() -> None:
    settings = FeatureSettings()

    with pytest.raises(ValidationError):
        settings.rsi_period = 7


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enabled_groups": ()}, "at least one feature group"),
        ({"enabled_groups": (FeatureGroup.PRICE, FeatureGroup.PRICE)}, "unique"),
        ({"sma_windows": ()}, "must not be empty"),
        ({"sma_windows": (5, 5)}, "must not contain duplicates"),
        ({"sma_windows": (0,)}, "must be positive"),
        ({"macd_fast": 30, "macd_slow": 26}, "macd_fast must be shorter"),
        ({"stoch_d_period": 20, "stoch_k_period": 14}, "must not exceed"),
        ({"bollinger_num_std": 0.0}, "greater than 0"),
    ],
)
def test_rejects_inconsistent_configuration(overrides, message) -> None:
    with pytest.raises(ValidationError, match=message):
        FeatureSettings(**overrides)


def test_app_settings_embeds_feature_settings() -> None:
    settings = AppSettings()

    assert isinstance(settings.features, FeatureSettings)


def test_json_configuration_can_override_feature_windows(tmp_path) -> None:
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps({"features": {"sma_windows": [7, 21]}}),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.features.sma_windows == (7, 21)


def test_json_configuration_rejects_invalid_feature_windows(tmp_path) -> None:
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps({"features": {"macd_fast": 30, "macd_slow": 26}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="invalid configuration file"):
        load_settings(path)
