"""End-to-end wiring: candles to features to regimes to setups."""

REQUIRED_FIELDS = (
    "symbol",
    "timeframe",
    "timestamp",
    "available_at",
    "direction",
    "setup",
    "evidence",
    "entry_context",
    "invalidation_context",
    "features",
)


def test_pipeline_produces_a_scan(run_pipeline, volatile_candles) -> None:
    scan = run_pipeline(volatile_candles)

    assert scan.symbol == "BTC/USDT"
    assert scan.timeframe == "1h"
    assert isinstance(scan.signals, tuple)


def test_every_signal_lands_on_a_candle(run_pipeline, volatile_candles) -> None:
    scan = run_pipeline(volatile_candles)

    open_times = set(volatile_candles["open_time"])
    for signal in scan.signals:
        assert signal.timestamp in open_times
        assert signal.available_at > signal.timestamp


def test_every_signal_carries_the_full_record(run_pipeline, volatile_candles) -> None:
    scan = run_pipeline(volatile_candles)

    for signal in scan.signals:
        for field in REQUIRED_FIELDS:
            assert getattr(signal, field) is not None
        assert signal.evidence
        assert signal.features
        assert signal.entry_context.detail
        assert signal.invalidation_context.detail


def test_no_signal_precedes_the_warmup(run_pipeline, trending_candles) -> None:
    """Setups depend on averages and regimes, so early rows cannot qualify."""
    scan = run_pipeline(trending_candles)

    earliest = trending_candles["open_time"].iloc[5]
    assert all(signal.timestamp >= earliest for signal in scan.signals)


def test_pipeline_is_deterministic(run_pipeline, volatile_candles) -> None:
    first = run_pipeline(volatile_candles)
    second = run_pipeline(volatile_candles)

    assert first.signals == second.signals
