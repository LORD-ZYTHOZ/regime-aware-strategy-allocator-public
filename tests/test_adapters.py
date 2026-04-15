"""Tests for strategy adapters."""

import pytest
from datetime import datetime

from allocator.adapters import (
    DivergenceAdapter,
    DivergenceNative,
    HorizonAdapter,
    HorizonNative,
    SingularityAdapter,
    SingularityNative,
)
from allocator.schemas import StrategyOutput

TS = datetime(2024, 1, 15, 9, 30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _singularity(kelly: float = 0.25, win_rate: float = 0.65, spread: float = 0.30) -> SingularityNative:
    return SingularityNative(kelly_fraction=kelly, win_rate=win_rate, spread=spread)


def _divergence(
    gamma: float = 0.7,
    regime: str = "RISK_ON",
    div_score: float = 0.15,
    spread: float = 0.30,
) -> DivergenceNative:
    return DivergenceNative(
        gamma=gamma,
        regime=regime,
        persona_divergence_score=div_score,
        spread=spread,
    )


def _horizon(confidence: float = 0.6, spread: float = 0.30) -> HorizonNative:
    return HorizonNative(confidence=confidence, spread=spread)


# ---------------------------------------------------------------------------
# Singularity
# ---------------------------------------------------------------------------

class TestSingularityAdapter:
    adapter = SingularityAdapter()

    def test_normal(self):
        out = self.adapter.adapt(_singularity(), "singularity", TS)
        assert isinstance(out, StrategyOutput)
        assert out.strategy_id == "singularity"
        assert out.timestamp == TS
        assert out.forecast == pytest.approx(0.25)
        assert out.uncertainty == pytest.approx(0.35)
        assert out.cost_estimate == pytest.approx(0.30)

    def test_no_go_session(self):
        """kelly_fraction=0 when session is no-go -> forecast=0."""
        out = self.adapter.adapt(_singularity(kelly=0.0), "singularity", TS)
        assert out.forecast == pytest.approx(0.0)

    def test_perfect_win_rate(self):
        """win_rate=1.0 -> uncertainty=0."""
        out = self.adapter.adapt(_singularity(win_rate=1.0), "singularity", TS)
        assert out.uncertainty == pytest.approx(0.0)

    def test_zero_win_rate(self):
        """win_rate=0 -> uncertainty=1.0 (not negative, satisfies schema ge=0)."""
        out = self.adapter.adapt(_singularity(win_rate=0.0), "singularity", TS)
        assert out.uncertainty == pytest.approx(1.0)

    def test_metadata_fields(self):
        out = self.adapter.adapt(_singularity(kelly=0.3, win_rate=0.7), "singularity", TS)
        assert "kelly_fraction" in out.metadata
        assert "win_rate" in out.metadata

    def test_returns_new_object_each_call(self):
        native = _singularity()
        out1 = self.adapter.adapt(native, "singularity", TS)
        out2 = self.adapter.adapt(native, "singularity", TS)
        assert out1 is not out2


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

class TestDivergenceAdapter:
    adapter = DivergenceAdapter()

    def test_risk_on(self):
        out = self.adapter.adapt(_divergence(gamma=0.7, regime="RISK_ON"), "divergence", TS)
        assert out.forecast == pytest.approx(0.7)

    def test_risk_off_inverts_sign(self):
        """RISK_OFF -> negative forecast."""
        out = self.adapter.adapt(_divergence(gamma=0.7, regime="RISK_OFF"), "divergence", TS)
        assert out.forecast == pytest.approx(-0.7)

    def test_neutral_regime_zeroes_forecast(self):
        """NEUTRAL -> forecast=0 regardless of gamma."""
        out = self.adapter.adapt(_divergence(gamma=0.8, regime="NEUTRAL"), "divergence", TS)
        assert out.forecast == pytest.approx(0.0)

    def test_gated_gamma_zeroes_forecast(self):
        """gamma=0 (already gated upstream) -> forecast=0 even with RISK_ON."""
        out = self.adapter.adapt(_divergence(gamma=0.0, regime="RISK_ON"), "divergence", TS)
        assert out.forecast == pytest.approx(0.0)

    def test_gamma_not_rescaled(self):
        """gamma passes through unchanged (VIX-gated upstream, not re-normalised)."""
        out = self.adapter.adapt(_divergence(gamma=1.0, regime="RISK_ON"), "divergence", TS)
        assert out.forecast == pytest.approx(1.0)

    def test_uncertainty_is_divergence_score(self):
        out = self.adapter.adapt(_divergence(div_score=0.25), "divergence", TS)
        assert out.uncertainty == pytest.approx(0.25)

    def test_unknown_regime_treated_as_neutral(self):
        """Unknown regime string -> sign=0 -> forecast=0."""
        out = self.adapter.adapt(_divergence(gamma=0.5, regime="UNKNOWN"), "divergence", TS)
        assert out.forecast == pytest.approx(0.0)

    def test_metadata_fields(self):
        out = self.adapter.adapt(_divergence(), "divergence", TS)
        assert "gamma" in out.metadata
        assert "regime_sign" in out.metadata


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------

class TestHorizonAdapter:
    adapter = HorizonAdapter()

    def test_normal(self):
        out = self.adapter.adapt(_horizon(confidence=0.6), "horizon", TS)
        assert out.forecast == pytest.approx(0.6)
        assert out.uncertainty == pytest.approx(0.4)

    def test_zero_confidence(self):
        """confidence=0 -> uncertainty=1.0 (satisfies schema ge=0)."""
        out = self.adapter.adapt(_horizon(confidence=0.0), "horizon", TS)
        assert out.forecast == pytest.approx(0.0)
        assert out.uncertainty == pytest.approx(1.0)

    def test_full_confidence(self):
        """confidence=1 -> uncertainty=0."""
        out = self.adapter.adapt(_horizon(confidence=1.0), "horizon", TS)
        assert out.forecast == pytest.approx(1.0)
        assert out.uncertainty == pytest.approx(0.0)

    def test_cost_estimate_is_spread(self):
        out = self.adapter.adapt(_horizon(spread=0.50), "horizon", TS)
        assert out.cost_estimate == pytest.approx(0.50)

    def test_metadata_has_confidence(self):
        out = self.adapter.adapt(_horizon(confidence=0.75), "horizon", TS)
        assert out.metadata["confidence"] == pytest.approx(0.75)

    def test_output_is_strategy_output(self):
        out = self.adapter.adapt(_horizon(), "horizon", TS)
        assert isinstance(out, StrategyOutput)
