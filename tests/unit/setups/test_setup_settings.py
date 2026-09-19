"""Tests for setup configuration and its integration with AppSettings."""

import json

import pytest
from pydantic import ValidationError

from crypto_analyzer.config import AppSettings, SetupSettings, load_settings
from crypto_analyzer.setups import SetupType


def test_defaults_enable_every_setup() -> None:
    settings = SetupSettings()

    assert set(settings.enabled_setups) == set(SetupType)


def test_settings_are_immutable() -> None:
    settings = SetupSettings()

    with pytest.raises(ValidationError):
        settings.trend_fast_window = 10


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enabled_setups": ()}, "at least one setup"),
        (
            {"enabled_setups": (SetupType.BREAKOUT, SetupType.BREAKOUT)},
            "unique",
        ),
        (
            {"continuation_rsi_floor": 70.0, "continuation_rsi_ceiling": 40.0},
            "must be below",
        ),
        (
            {"reversion_rsi_oversold": 80.0, "reversion_rsi_overbought": 20.0},
            "must be below",
        ),
        ({"reversion_long_band_max": 1.5}, "must be below"),
        ({"breakout_squeeze_ratio": 0.0}, "greater than 0"),
        ({"breakout_squeeze_lookback": 1}, "greater than or equal to 2"),
    ],
)
def test_rejects_inconsistent_configuration(overrides, message) -> None:
    with pytest.raises(ValidationError, match=message):
        SetupSettings(**overrides)


def test_features_follow_the_configured_windows() -> None:
    settings = SetupSettings(
        trend_fast_window=10,
        ema_slope_span=7,
        rsi_period=9,
        volume_window=30,
        structure_lookback=15,
        bollinger_period=25,
    )

    assert settings.features_for(SetupType.TREND_CONTINUATION) == (
        "trend_price_vs_sma_10",
        "trend_ema_slope_7",
        "momentum_rsi_9",
        "volume_relative_30",
        "structure_recent_low_15",
    )
    assert "structure_breakout_up_15" in settings.features_for(SetupType.BREAKOUT)
    assert "volatility_bollinger_position_25" in settings.features_for(
        SetupType.MEAN_REVERSION
    )


def test_required_features_only_cover_the_enabled_setups() -> None:
    settings = SetupSettings(enabled_setups=(SetupType.MEAN_REVERSION,))

    required = settings.required_features

    assert "volatility_bollinger_position_20" in required
    assert "structure_breakout_up_20" not in required


def test_app_settings_embeds_setup_settings() -> None:
    settings = AppSettings()

    assert isinstance(settings.setups, SetupSettings)


def test_json_configuration_can_disable_a_setup(tmp_path) -> None:
    path = tmp_path / "setups.json"
    path.write_text(
        json.dumps({"setups": {"enabled_setups": ["breakout"]}}),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.setups.enabled_setups == (SetupType.BREAKOUT,)
