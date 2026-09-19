"""Tests for backtest configuration."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from crypto_analyzer.backtesting import BacktestSettings
from crypto_analyzer.backtesting.types import AmbiguousExitPolicy
from crypto_analyzer.config import AppSettings


def test_defaults_charge_costs_and_keep_a_one_bar_delay() -> None:
    settings = BacktestSettings()

    assert settings.execution_delay_bars == 1
    assert settings.fee_bps > 0.0
    assert settings.slippage_bps > 0.0
    assert settings.ambiguous_exit_policy is AmbiguousExitPolicy.STOP_FIRST


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"execution_delay_bars": 0}, "greater than or equal to 1"),
        ({"fee_bps": -1.0}, "greater than or equal to 0"),
        ({"slippage_bps": -1.0}, "greater than or equal to 0"),
        ({"take_profit_r_multiple": 0.0}, "greater than 0"),
        ({"risk_fraction": 0.0}, "greater than 0"),
        ({"max_notional_multiple": 0.0}, "greater than 0"),
        ({"max_concurrent_positions": 0}, "greater than or equal to 1"),
        ({"max_holding_bars": 0}, "greater than or equal to 1"),
    ],
)
def test_rejects_unusable_values(overrides, message) -> None:
    with pytest.raises(ValidationError, match=message):
        BacktestSettings(**overrides)


def test_same_bar_execution_is_unrepresentable() -> None:
    """A zero delay would let the decision bar supply its own fill."""
    with pytest.raises(ValidationError):
        BacktestSettings(execution_delay_bars=0)


def test_window_must_be_ordered_and_aware() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 6, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="earlier than"):
        BacktestSettings(window_start=end, window_end=start)
    with pytest.raises(ValidationError, match="timezone-aware"):
        BacktestSettings(window_start=datetime(2024, 1, 1))


def test_costs_are_exposed_as_rates() -> None:
    settings = BacktestSettings(fee_bps=10.0, slippage_bps=5.0)

    assert settings.fee_rate == pytest.approx(0.001)
    assert settings.slippage_rate == pytest.approx(0.0005)


def test_app_settings_embeds_backtest_settings() -> None:
    assert isinstance(AppSettings().backtest, BacktestSettings)


def test_settings_are_immutable() -> None:
    settings = BacktestSettings()

    with pytest.raises(ValidationError):
        settings.fee_bps = 50.0
