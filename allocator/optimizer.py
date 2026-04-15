"""
QP optimizer layer.

Solves:
    b_t* = argmax_b [ ν_t^T b  −  (γ_t/2) b^T Σ_t^slv b  −  λ_turn ‖b − b_{t-1}‖₁ ]
    subject to:  1^T b ≤ 1,   b ≥ 0

The L1 turnover term penalises large swings in allocations (transaction costs + slippage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False

from allocator.schemas import OptimizerOutput, RiskFlag, StrategyRiskFlag


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizerConfig:
    gamma: float = 1.0           # risk-aversion coefficient γ_t
    lambda_turn: float = 0.0001  # turnover penalty λ_turn (daily-unit scale)
    max_budget_per_strategy: float = 1.0   # upper bound per strategy
    de_risk_threshold: float = -0.5        # ν below this → DE_RISK flag
    kill_threshold: float = -1.0           # ν below this → KILL flag
    solver: str = "CLARABEL"               # cvxpy solver


# ---------------------------------------------------------------------------
# Core QP solver
# ---------------------------------------------------------------------------

def solve_qp(
    nu: np.ndarray,
    sigma: np.ndarray,
    b_prev: np.ndarray,
    strategy_ids: List[str],
    cfg: OptimizerConfig,
    timestamp: datetime,
) -> OptimizerOutput:
    """
    Solve the budget QP and return an OptimizerOutput.

    Parameters
    ----------
    nu           : (S,) net utility vector ν_t
    sigma        : (S, S) slippage-adjusted covariance Σ_t^slv
    b_prev       : (S,) previous budget vector b_{t-1}
    strategy_ids : list of strategy id strings, length S
    cfg          : optimizer configuration
    timestamp    : current time step
    """
    if not _CVXPY_AVAILABLE:
        raise ImportError("cvxpy is required for the QP optimizer. pip install cvxpy")

    n = len(nu)
    b = cp.Variable(n, nonneg=True)

    # Objective: maximise utility - risk - turnover
    objective = cp.Maximize(
        nu @ b
        - (cfg.gamma / 2.0) * cp.quad_form(b, sigma)
        - cfg.lambda_turn * cp.norm1(b - b_prev)
    )

    constraints = [
        cp.sum(b) <= 1.0,
        b <= cfg.max_budget_per_strategy,
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cfg.solver, warm_start=True)
    except Exception:
        # fallback solver
        problem.solve(solver="SCS")

    if b.value is None or problem.status in ("infeasible", "unbounded"):
        # safe fallback: equal-weight with cash buffer
        b_val = np.full(n, 1.0 / (n + 1))
        solver_status = f"fallback ({problem.status})"
    else:
        b_val = np.clip(b.value, 0.0, None)
        solver_status = problem.status

    # normalise so budgets sum to at most 1
    total = b_val.sum()
    if total > 1.0:
        b_val = b_val / total

    cash = float(max(0.0, 1.0 - b_val.sum()))
    budgets = {sid: float(bv) for sid, bv in zip(strategy_ids, b_val)}

    # risk flags
    risk_flags = _compute_risk_flags(nu, strategy_ids, cfg)

    return OptimizerOutput(
        timestamp=timestamp,
        budgets=budgets,
        cash=cash,
        risk_flags=risk_flags,
        objective_value=float(problem.value) if problem.value is not None else None,
        solver_status=solver_status,
    )


# ---------------------------------------------------------------------------
# Risk flag logic
# ---------------------------------------------------------------------------

def _compute_risk_flags(
    nu: np.ndarray,
    strategy_ids: List[str],
    cfg: OptimizerConfig,
) -> List[StrategyRiskFlag]:
    flags: List[StrategyRiskFlag] = []
    for sid, v in zip(strategy_ids, nu):
        if v <= cfg.kill_threshold:
            flags.append(
                StrategyRiskFlag(
                    strategy_id=sid,
                    flag=RiskFlag.KILL,
                    reason=f"ν={v:.4f} below kill threshold {cfg.kill_threshold}",
                )
            )
        elif v <= cfg.de_risk_threshold:
            flags.append(
                StrategyRiskFlag(
                    strategy_id=sid,
                    flag=RiskFlag.DE_RISK,
                    reason=f"ν={v:.4f} below de-risk threshold {cfg.de_risk_threshold}",
                )
            )
        else:
            flags.append(StrategyRiskFlag(strategy_id=sid, flag=RiskFlag.OK))
    return flags


# ---------------------------------------------------------------------------
# Regime-adaptive gamma
# ---------------------------------------------------------------------------

def adaptive_gamma(base_gamma: float, market_vol: float, vol_target: float = 0.15) -> float:
    """
    Scale risk aversion by realised vol relative to a target.
    High vol regime → higher γ → tighter risk control.
    """
    ratio = market_vol / max(vol_target, 1e-6)
    return float(base_gamma * ratio)
