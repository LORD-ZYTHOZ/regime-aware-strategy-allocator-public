"""
Strategy adapters — translate each strategy's native output into
the StrategyOutput schema consumed by the regime-aware allocator.

Each adapter is stateless and immutable: adapt() always returns a new
StrategyOutput without modifying the input dict.

To implement a custom adapter, satisfy the StrategyAdapter Protocol:

    class MyAdapter:
        def adapt(self, native: Any, strategy_id: str, timestamp: datetime) -> StrategyOutput:
            ...
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Protocol, TypedDict

from allocator.schemas import StrategyOutput


# ---------------------------------------------------------------------------
# Protocol — structural contract for all adapters
# ---------------------------------------------------------------------------

class StrategyAdapter(Protocol):
    """
    Structural interface every strategy adapter must satisfy.

    `native` is typed as Any here because each concrete adapter accepts
    a different TypedDict.  Callers that want full type safety should use
    the concrete adapter classes directly; this Protocol is for consumers
    that hold adapters polymorphically (e.g. a future dynamic registry).
    """

    def adapt(
        self,
        native: Any,
        strategy_id: str,
        timestamp: datetime,
    ) -> StrategyOutput:
        """Translate native strategy output into StrategyOutput."""
        ...


# ---------------------------------------------------------------------------
# Singularity
# ---------------------------------------------------------------------------

class SingularityNative(TypedDict):
    """Native output produced by the Singularity session-profile strategy."""
    kelly_fraction: float   # 0.0 when session is no-go
    win_rate: float         # historical win rate for the active session
    spread: float           # current spread in price units


class SingularityAdapter:
    """Adapts Singularity session-profile output to StrategyOutput."""

    def adapt(
        self,
        native: SingularityNative,
        strategy_id: str,
        timestamp: datetime,
    ) -> StrategyOutput:
        uncertainty = max(0.0, 1.0 - native["win_rate"])
        return StrategyOutput(
            strategy_id=strategy_id,
            timestamp=timestamp,
            forecast=native["kelly_fraction"],
            uncertainty=uncertainty,
            cost_estimate=native["spread"],
            metadata={
                "kelly_fraction": native["kelly_fraction"],
                "win_rate": native["win_rate"],
            },
        )


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

_REGIME_SIGN: Dict[str, int] = {
    "RISK_ON": 1,
    "RISK_OFF": -1,
    "NEUTRAL": 0,
}


class DivergenceNative(TypedDict):
    """Native output produced by the Divergence LLM-persona strategy."""
    gamma: float                    # 0.0 if gated (conf < 0.6 or div_score > 0.4)
    regime: str                     # "RISK_ON" | "RISK_OFF" | "NEUTRAL"
    persona_divergence_score: float # disagreement score across personas (>= 0)
    spread: float


class DivergenceAdapter:
    """
    Adapts Divergence LLM-persona output to StrategyOutput.

    gamma is NOT re-scaled — it is already risk-adjusted (VIX-gated).
    Only the regime sign is applied.  NEUTRAL -> forecast=0 -> QP allocates
    zero weight to Divergence that cycle.  Unknown regime strings are treated
    as NEUTRAL (sign=0).
    """

    def adapt(
        self,
        native: DivergenceNative,
        strategy_id: str,
        timestamp: datetime,
    ) -> StrategyOutput:
        sign = _REGIME_SIGN.get(native["regime"], 0)
        forecast = native["gamma"] * sign
        uncertainty = max(0.0, native["persona_divergence_score"])
        return StrategyOutput(
            strategy_id=strategy_id,
            timestamp=timestamp,
            forecast=forecast,
            uncertainty=uncertainty,
            cost_estimate=native["spread"],
            metadata={
                "gamma": native["gamma"],
                "regime_sign": float(sign),
            },
        )


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------

class HorizonNative(TypedDict):
    """Native output produced by the Horizon LightGBM stacker strategy."""
    confidence: float   # 0-1 probability estimate from the stacker
    spread: float


class HorizonAdapter:
    """Adapts Horizon LightGBM stacker output to StrategyOutput."""

    def adapt(
        self,
        native: HorizonNative,
        strategy_id: str,
        timestamp: datetime,
    ) -> StrategyOutput:
        confidence = native["confidence"]
        uncertainty = max(0.0, 1.0 - confidence)
        return StrategyOutput(
            strategy_id=strategy_id,
            timestamp=timestamp,
            forecast=confidence,
            uncertainty=uncertainty,
            cost_estimate=native["spread"],
            metadata={"confidence": confidence},
        )
