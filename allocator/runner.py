"""
AllocatorRunner — coordination layer that bridges strategy native outputs
to the AllocatorEngine via typed adapters.

Usage
-----
from allocator.runner import AllocatorRunner
from allocator.adapters import SingularityNative, DivergenceNative, HorizonNative
from allocator.schemas import MarketState
from datetime import datetime

runner = AllocatorRunner()
result = runner.step(
    timestamp=datetime.utcnow(),
    market_state=market_state,
    singularity=SingularityNative(kelly_fraction=0.3, win_rate=0.65, spread=0.30),
    divergence=DivergenceNative(gamma=0.7, regime="RISK_ON", persona_divergence_score=0.15, spread=0.30),
    horizon=HorizonNative(confidence=0.6, spread=0.30),
)
# result.budgets -> {"singularity": 0.xx, "divergence": 0.xx, "horizon": 0.xx}
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from allocator.adapters import (
    DivergenceAdapter,
    DivergenceNative,
    HorizonAdapter,
    HorizonNative,
    SingularityAdapter,
    SingularityNative,
)
from allocator.engine import AllocatorEngine, EstimatorConfig, OptimizerConfig
from allocator.schemas import AllocatorInput, MarketState, OptimizerOutput, StrategyOutput

STRATEGY_IDS = ["singularity", "divergence", "horizon"]


class AllocatorRunner:
    """
    Wraps AllocatorEngine and applies per-strategy adapters each step.

    Strategies are identified by fixed IDs: singularity, divergence, horizon.
    Any strategy absent in a given step receives a zero-forecast StrategyOutput
    (uncertainty=1.0) so the QP can still run with partial data.
    """

    def __init__(
        self,
        est_cfg: Optional[EstimatorConfig] = None,
        opt_cfg: Optional[OptimizerConfig] = None,
    ) -> None:
        self._engine = AllocatorEngine(
            strategy_ids=STRATEGY_IDS,
            est_cfg=est_cfg,
            opt_cfg=opt_cfg,
        )
        self._singularity = SingularityAdapter()
        self._divergence = DivergenceAdapter()
        self._horizon = HorizonAdapter()

    def step(
        self,
        timestamp: datetime,
        market_state: MarketState,
        singularity: Optional[SingularityNative] = None,
        divergence: Optional[DivergenceNative] = None,
        horizon: Optional[HorizonNative] = None,
    ) -> OptimizerOutput:
        """Run one allocation step from native strategy outputs."""
        outputs = [
            self._to_singularity(singularity, timestamp, market_state),
            self._to_divergence(divergence, timestamp, market_state),
            self._to_horizon(horizon, timestamp, market_state),
        ]
        inp = AllocatorInput(
            timestamp=timestamp,
            market_state=market_state,
            strategy_outputs=outputs,
        )
        return self._engine.step(inp)

    # ------------------------------------------------------------------
    # Per-strategy helpers
    # ------------------------------------------------------------------

    def _to_singularity(
        self,
        native: Optional[SingularityNative],
        timestamp: datetime,
        market_state: MarketState,
    ) -> StrategyOutput:
        if native is None:
            return _absent_output("singularity", timestamp, market_state.slippage)
        return self._singularity.adapt(native, "singularity", timestamp)

    def _to_divergence(
        self,
        native: Optional[DivergenceNative],
        timestamp: datetime,
        market_state: MarketState,
    ) -> StrategyOutput:
        if native is None:
            return _absent_output("divergence", timestamp, market_state.slippage)
        return self._divergence.adapt(native, "divergence", timestamp)

    def _to_horizon(
        self,
        native: Optional[HorizonNative],
        timestamp: datetime,
        market_state: MarketState,
    ) -> StrategyOutput:
        if native is None:
            return _absent_output("horizon", timestamp, market_state.slippage)
        return self._horizon.adapt(native, "horizon", timestamp)


def _absent_output(strategy_id: str, timestamp: datetime, slippage: float) -> StrategyOutput:
    """Zero-forecast placeholder used when a strategy has no output this cycle."""
    return StrategyOutput(
        strategy_id=strategy_id,
        timestamp=timestamp,
        forecast=0.0,
        uncertainty=1.0,
        cost_estimate=slippage,
        metadata={"absent": 1.0},
    )
