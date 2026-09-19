"""Tests for feature registration, metadata, and declared warmup."""

import json

import pytest

from crypto_analyzer.features import (
    FeatureGroup,
    FeatureSettings,
    build_default_registry,
)
from crypto_analyzer.features.exceptions import FeatureConfigurationError


def test_default_registry_covers_every_group() -> None:
    registry = build_default_registry(FeatureSettings())

    assert {definition.group for definition in registry.definitions} == set(
        FeatureGroup
    )


def test_registry_rejects_a_duplicate_name() -> None:
    registry = build_default_registry(
        FeatureSettings(enabled_groups=(FeatureGroup.PRICE,))
    )

    with pytest.raises(FeatureConfigurationError, match="already registered"):
        registry.register(registry.definitions[0])


def test_disabling_a_group_removes_only_its_features() -> None:
    full = build_default_registry(FeatureSettings())
    price_only = build_default_registry(
        FeatureSettings(enabled_groups=(FeatureGroup.PRICE,))
    )

    assert len(price_only) < len(full)
    assert all(
        definition.group is FeatureGroup.PRICE for definition in price_only.definitions
    )


def test_configured_windows_drive_registered_names() -> None:
    registry = build_default_registry(FeatureSettings(sma_windows=(7, 11)))

    assert "trend_sma_7" in registry.names
    assert "trend_sma_11" in registry.names
    assert "trend_sma_20" not in registry.names


def test_ema_distance_tracks_the_configured_ema_windows() -> None:
    registry = build_default_registry(FeatureSettings(ema_windows=(5, 9)))

    assert "trend_price_vs_ema_5" in registry.names
    assert "trend_price_vs_ema_9" in registry.names
    assert "trend_price_vs_ema_12" not in registry.names


def test_trend_distance_covers_both_average_types() -> None:
    registry = build_default_registry(FeatureSettings())

    assert "trend_price_vs_sma_20" in registry.names
    assert "trend_price_vs_ema_12" in registry.names


def test_feature_names_are_unique() -> None:
    registry = build_default_registry(FeatureSettings())

    assert len(set(registry.names)) == len(registry.names)


def test_every_definition_has_complete_metadata() -> None:
    registry = build_default_registry(FeatureSettings())

    assert len(registry) > 0
    for definition in registry.definitions:
        metadata = definition.metadata
        assert metadata.name == definition.name
        assert metadata.group is definition.group
        assert metadata.description
        assert metadata.lookback >= 1


def test_metadata_serializes_to_plain_json() -> None:
    registry = build_default_registry(FeatureSettings())

    for metadata in registry.metadata:
        payload = metadata.to_dict()
        assert set(payload) == {
            "name",
            "group",
            "description",
            "lookback",
            "parameters",
        }
        json.dumps(payload)


def test_declared_lookback_matches_the_observed_warmup(
    wave_frame, leading_missing
) -> None:
    registry = build_default_registry(FeatureSettings())

    for definition in registry.definitions:
        series = definition.compute(wave_frame)
        assert leading_missing(series) == definition.lookback - 1, definition.name


def test_max_lookback_tracks_the_longest_configured_window() -> None:
    short = build_default_registry(FeatureSettings(sma_windows=(5,)))
    long = build_default_registry(FeatureSettings(sma_windows=(300,)))

    assert short.max_lookback < long.max_lookback
    assert long.max_lookback == 300


def test_for_group_returns_only_that_group() -> None:
    registry = build_default_registry(FeatureSettings())

    for group in FeatureGroup:
        selected = registry.for_group(group)
        assert selected
        assert all(definition.group is group for definition in selected)
