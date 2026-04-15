"""
Data schemas for the regime-aware multi-strategy allocator.

Three schema groups:
  - StrategyOutput  : what each strategy engine produces
  - AllocatorInput  : what the allocator consumes each step
  - OptimizerOutput : what the QP solver returns
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Strategy layer
# ---------------------------------------------------------------------------

class StrategyOutput(BaseModel):
    """Standardised JSON contract every strategy engine must satisfy."""

    strategy_id: str
    timestamp: datetime
    forecast: float = Field(
        description="Expected return / edge (χ_t^s) over the next period"
    )
    uncertainty: float = Field(
        ge=0.0,
        description="Risk / uncertainty estimate (U_t^s); e.g. rolling vol or CVaR",
    )
    cost_estimate: float = Field(
        ge=0.0,
        description="Expected transaction cost + slippage (Ĉ_t^s)",
    )
    metadata: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional extra signals (regime scores, hit-rate, etc.)",
    )


# ---------------------------------------------------------------------------
# Allocator layer
# ---------------------------------------------------------------------------

class MarketState(BaseModel):
    """Market-state vector s_t^alloc fed into the allocator."""

    timestamp: datetime
    volatility: float = Field(ge=0.0, description="Realised / implied vol")
    dispersion: float = Field(ge=0.0, description="Cross-asset return dispersion")
    correlation: float = Field(
        ge=-1.0, le=1.0, description="Average pairwise correlation"
    )
    breadth: float = Field(
        ge=0.0, le=1.0, description="Fraction of strategies with positive edge"
    )
    trend: float = Field(description="Aggregate trend signal, signed [-1, 1]")
    slippage: float = Field(ge=0.0, description="Current market slippage estimate")
    liquidity: float = Field(ge=0.0, description="Liquidity score (higher = better)")


class AllocatorInput(BaseModel):
    """Full input packet consumed by the allocator at each time step."""

    timestamp: datetime
    market_state: MarketState
    strategy_outputs: List[StrategyOutput]
    prev_budgets: Dict[str, float] = Field(
        default_factory=dict,
        description="Budget vector b_{t-1} keyed by strategy_id",
    )

    @model_validator(mode="after")
    def timestamps_align(self) -> "AllocatorInput":
        for s in self.strategy_outputs:
            if s.timestamp != self.timestamp:
                raise ValueError(
                    f"Strategy {s.strategy_id} timestamp {s.timestamp} "
                    f"does not match allocator timestamp {self.timestamp}"
                )
        return self


# ---------------------------------------------------------------------------
# Optimizer layer
# ---------------------------------------------------------------------------

class RiskFlag(str, Enum):
    OK = "ok"
    DE_RISK = "de_risk"
    KILL = "kill"


class StrategyRiskFlag(BaseModel):
    strategy_id: str
    flag: RiskFlag
    reason: Optional[str] = None


class OptimizerOutput(BaseModel):
    """Result produced by the QP solver each step."""

    timestamp: datetime
    budgets: Dict[str, float] = Field(
        description="Optimal budget b_t* keyed by strategy_id (sums to ≤ 1)"
    )
    cash: float = Field(
        ge=0.0, description="Residual cash allocation c_t^alloc = 1 - sum(budgets)"
    )
    risk_flags: List[StrategyRiskFlag] = Field(default_factory=list)
    objective_value: Optional[float] = None
    solver_status: str = "unknown"

    @model_validator(mode="after")
    def budgets_sum_to_one(self) -> "OptimizerOutput":
        total = sum(self.budgets.values()) + self.cash
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"budgets + cash must sum to 1.0, got {total:.6f}")
        return self
