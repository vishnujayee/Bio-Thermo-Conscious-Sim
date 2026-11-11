"""Config loading, strictness, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from btcs.config import Config, ConfigError

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_are_valid():
    Config().validate()


def test_shipped_default_yaml_loads_and_validates():
    config = Config.load(CONFIG_DIR / "default.yaml")
    config.validate()
    assert config.run.seed == 42
    assert config.physics.landauer_enabled is True
    assert config.cognition.brain == "mock"


def test_shipped_yaml_matches_dataclass_defaults():
    """The YAML is documentation as much as configuration.

    If it drifts from the dataclass defaults, one of the two is lying to the
    reader. This catches a default changed in code but not in the file.
    """
    shipped = Config.load(CONFIG_DIR / "default.yaml")
    assert shipped.to_dict() == Config().to_dict()


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"wrold": {}})


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key.*physics"):
        Config.from_dict({"physics": {"landuaer_enabled": True}})


def test_partial_config_fills_defaults():
    config = Config.from_dict({"run": {"seed": 7}})
    assert config.run.seed == 7
    assert config.run.max_ticks == Config().run.max_ticks


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"physics": {"erase_efficiency": 0.0}}, "erase_efficiency"),
        ({"physics": {"erase_efficiency": 1.5}}, "erase_efficiency"),
        ({"physics": {"ingest_efficiency": 2.0}}, "ingest_efficiency"),
        ({"world": {"ambient_temperature": 0.0}}, "ambient_temperature"),
        ({"agent": {"s_warn": 99.0}}, "strictly increasing"),
        ({"selection": {"mutation_rate": 1.5}}, "mutation_rate"),
        ({"cognition": {"brain": "telepathy"}}, "must be .mock., .local. or .llm."),
        ({"selection": {"executable_genomes": True}}, "Phase 5b"),
        ({"world": {"width": 0}}, "dimensions"),
    ],
)
def test_validation_catches_bad_values(patch, expected):
    with pytest.raises(ConfigError, match=expected):
        Config.from_dict(patch).validate()


def test_node_temperature_must_exceed_ambient():
    """A node at ambient carries no extractable free energy (Carnot bound)."""
    with pytest.raises(ConfigError, match="must exceed ambient"):
        Config.from_dict(
            {"world": {"ambient_temperature": 5.0, "node_temperature_mean": 4.0}}
        ).validate()


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        Config.load(CONFIG_DIR / "does-not-exist.yaml")


class TestPhiTiers:
    def test_default_tiers_are_ordered_and_start_at_zero(self):
        tiers = Config().cognition.phi_tiers
        assert tiers[0].min_phi == 0.0
        assert [t.min_phi for t in tiers] == sorted(t.min_phi for t in tiers)

    @pytest.mark.parametrize(
        ("phi", "model"),
        [
            (0.0, "claude-haiku-4-5"),
            (0.69, "claude-haiku-4-5"),
            (0.7, "claude-sonnet-5"),
            (1.39, "claude-sonnet-5"),
            (1.4, "claude-opus-5"),
            (2.5, "claude-opus-5"),
            (99.0, "claude-opus-5"),
        ],
    )
    def test_routing_selects_expected_model(self, phi, model):
        assert Config().cognition.tier_for(phi).model == model

    def test_every_tier_is_reachable_in_practice(self):
        """A tier no organism reaches is a routing rule that never fires, which
        is worse than no rule at all because it looks like one.

        The shipped thresholds were set from a measured Phi distribution --
        median 0.38, p90 1.36, p95 1.93, max 3.61 on the default world. If the
        world's constants change enough to move that distribution, these
        boundaries need re-measuring, and this test is where that shows up.
        """
        observed_maximum = 3.61
        tiers = Config().cognition.phi_tiers
        assert tiers[-1].min_phi < observed_maximum, (
            "the top tier sits above anything ever measured; re-measure the "
            "distribution or lower the boundary"
        )

    def test_founders_start_in_the_cheapest_tier(self):
        """An organism with no experience has no integration to speak of, and
        should not begin life one rung up the model ladder."""
        config = Config()
        assert config.cognition.tier_for(config.agent.initial_phi).model == (
            config.cognition.phi_tiers[0].model
        )

    def test_entropy_multiplier_rises_with_phi(self):
        """Higher consciousness must cost strictly more entropy per token."""
        cognition = Config().cognition
        multipliers = [cognition.tier_for(phi).entropy_multiplier for phi in (0.0, 2.0, 5.0, 8.0)]
        assert multipliers == sorted(multipliers)
        assert multipliers[0] < multipliers[-1]

    def test_top_tier_uses_highest_effort(self):
        assert Config().cognition.tier_for(9.0).effort == "high"

    def test_custom_tiers_are_sorted_on_load(self):
        config = Config.from_dict(
            {
                "cognition": {
                    "phi_tiers": [
                        {"min_phi": 5.0, "model": "claude-opus-5"},
                        {"min_phi": 0.0, "model": "claude-haiku-4-5"},
                    ]
                }
            }
        )
        assert [t.min_phi for t in config.cognition.phi_tiers] == [0.0, 5.0]

    def test_tiers_must_cover_phi_zero(self):
        config = Config.from_dict(
            {"cognition": {"phi_tiers": [{"min_phi": 3.0, "model": "claude-opus-5"}]}}
        )
        with pytest.raises(ConfigError, match="min_phi 0.0"):
            config.validate()


def test_round_trips_through_yaml(tmp_path):
    original = Config.from_dict({"run": {"seed": 1234}, "world": {"width": 12}})
    path = tmp_path / "rt.yaml"
    path.write_text(yaml.safe_dump(original.to_dict()), encoding="utf-8")
    assert Config.load(path).to_dict() == original.to_dict()
