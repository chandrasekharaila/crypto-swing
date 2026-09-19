"""Tests for regime configuration and its integration with AppSettings."""

import json

import pytest
from pydantic import ValidationError

from crypto_analyzer.config import AppSettings, load_settings
from crypto_analyzer.data.exceptions import ConfigurationError
from crypto_analyzer.regimes import TREND_RULE_COUNT, RegimeSettings


def test_defaults_are_valid() -> None:
    settings = RegimeSettings()

    assert settings.trend_fast_window < settings.trend_slow_window
    assert settings.volatility_low_ratio < settings.volatility_high_ratio


def test_settings_are_immutable() -> None:
    settings = RegimeSettings()

    with pytest.raises(ValidationError):
        settings.trend_fast_window = 10


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trend_fast_window": 200, "trend_slow_window": 50}, "shorter"),
        ({"trend_fast_window": 50, "trend_slow_window": 50}, "shorter"),
        ({"trend_min_agreeing_rules": TREND_RULE_COUNT + 1}, "cannot exceed"),
        ({"trend_min_agreeing_rules": 0}, "greater than or equal to 1"),
        ({"trend_ma_spread_threshold": 0.0}, "greater than 0"),
        ({"volatility_high_ratio": 1.0}, "greater than 1"),
        ({"volatility_low_ratio": 1.0}, "less than 1"),
    ],
)
def test_rejects_inconsistent_configuration(overrides, message) -> None:
    with pytest.raises(ValidationError, match=message):
        RegimeSettings(**overrides)


def test_required_features_follow_the_configured_windows() -> None:
    settings = RegimeSettings(
        trend_fast_window=10,
        trend_slow_window=30,
        trend_ema_slope_span=7,
        volatility_atr_period=9,
    )

    assert settings.required_features == (
        "trend_sma_10",
        "trend_sma_30",
        "trend_price_vs_sma_10",
        "trend_ema_slope_7",
        "trend_ema_spread_pct",
        "volatility_atr_pct_9",
    )


def test_warmup_is_driven_by_the_slower_of_the_two_axes() -> None:
    settings = RegimeSettings(
        trend_slow_window=200,
        volatility_atr_period=14,
        volatility_lookback=200,
    )

    # Volatility needs its ATR period plus the whole trailing norm window.
    assert settings.warmup == 213


def test_app_settings_embeds_regime_settings() -> None:
    settings = AppSettings()

    assert isinstance(settings.regimes, RegimeSettings)


def test_json_configuration_can_override_regime_thresholds(tmp_path) -> None:
    path = tmp_path / "regimes.json"
    path.write_text(
        json.dumps({"regimes": {"trend_min_agreeing_rules": 2}}),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.regimes.trend_min_agreeing_rules == 2


def test_json_configuration_rejects_invalid_regime_thresholds(tmp_path) -> None:
    path = tmp_path / "regimes.json"
    path.write_text(
        json.dumps({"regimes": {"volatility_high_ratio": 0.5}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="invalid configuration file"):
        load_settings(path)
