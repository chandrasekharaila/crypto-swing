"""Tests that a regime always comes with the reasoning behind it."""

import pytest

from crypto_analyzer.regimes import UNKNOWN_REGIME, TrendRegime, VolatilityRegime

TREND_RULES = (
    "trend_ma_spread",
    "trend_price_distance",
    "trend_ema_slope",
    "trend_ema_spread",
)


def _by_rule(evidence) -> dict:
    return {item.rule: item for item in evidence}


def test_every_rule_reports_its_reasoning(run_regimes, uptrend_frame) -> None:
    result = run_regimes(uptrend_frame)

    rules = _by_rule(result.explain())

    assert set(rules) == {*TREND_RULES, "trend_aggregate", "volatility_ratio"}


def test_directional_rules_carry_their_measurement_and_threshold(
    run_regimes, uptrend_frame
) -> None:
    result = run_regimes(uptrend_frame)
    rules = _by_rule(result.explain())

    for name in TREND_RULES:
        assert rules[name].value is not None
        assert rules[name].threshold is not None
        assert rules[name].detail


def test_evidence_supports_the_reported_trend(run_regimes, uptrend_frame) -> None:
    result = run_regimes(uptrend_frame)
    rules = _by_rule(result.explain())

    assert rules["trend_aggregate"].supports == TrendRegime.BULLISH.value
    assert all(
        rules[name].supports == TrendRegime.BULLISH.value for name in TREND_RULES
    )


def test_aggregate_evidence_reports_the_vote_counts(run_regimes, uptrend_frame) -> None:
    result = run_regimes(uptrend_frame)

    aggregate = _by_rule(result.explain())["trend_aggregate"]

    assert "4 rule(s) support bullish" in aggregate.detail
    assert "0 support bearish" in aggregate.detail


def test_volatility_evidence_names_the_bands(
    run_regimes, volatility_spike_frame
) -> None:
    result = run_regimes(volatility_spike_frame)

    volatility = _by_rule(result.explain())["volatility_ratio"]

    assert volatility.supports == VolatilityRegime.HIGH.value
    assert volatility.value is not None
    assert "trailing" in volatility.detail


def test_warmup_rows_explain_themselves_as_unknown(run_regimes, uptrend_frame) -> None:
    result = run_regimes(uptrend_frame)

    rules = _by_rule(result.explain(0))

    assert rules["trend_aggregate"].supports == UNKNOWN_REGIME
    assert all(rules[name].supports == UNKNOWN_REGIME for name in TREND_RULES)
    assert "not available yet" in rules["trend_ma_spread"].detail
    assert rules["trend_ma_spread"].value is None


def test_a_specific_row_can_be_explained(
    run_regimes, uptrend_frame, regime_settings
) -> None:
    result = run_regimes(uptrend_frame)
    position = regime_settings.warmup

    rules = _by_rule(result.explain(position))

    assert (
        rules["trend_aggregate"].supports == result.frame["trend_regime"].iloc[position]
    )


def test_out_of_range_positions_are_rejected(run_regimes, uptrend_frame) -> None:
    result = run_regimes(uptrend_frame)

    with pytest.raises(IndexError, match="out of range"):
        result.explain(len(result.frame))
