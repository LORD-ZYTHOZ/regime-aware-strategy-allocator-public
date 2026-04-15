"""
Backtesting harness.

Ingests historical per-strategy returns, synthesises AllocatorInputs,
runs the engine step-by-step, and computes performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from allocator.engine import AllocatorEngine
from allocator.estimator import EstimatorConfig
from allocator.optimizer import OptimizerConfig
from allocator.schemas import (
    AllocatorInput,
    MarketState,
    StrategyOutput,
)


# ---------------------------------------------------------------------------
# Market state builder from raw returns
# ---------------------------------------------------------------------------

def _build_market_state(
    timestamp: datetime,
    returns_window: pd.DataFrame,
    slippage: float = 0.001,
) -> MarketState:
    """Derive market state features from a trailing returns window."""
    vols = returns_window.std()
    corr_matrix = returns_window.corr()
    corr_vals = corr_matrix.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, np.nan)
    off_diag = corr_vals[~np.isnan(corr_vals)]
    avg_corr = float(off_diag.mean()) if len(off_diag) > 0 else 0.0

    means = returns_window.mean()
    breadth = float((means > 0).mean())
    vol_scalar = float(vols.mean())

    return MarketState(
        timestamp=timestamp,
        volatility=vol_scalar,
        dispersion=float(vols.std()),
        correlation=avg_corr,
        breadth=breadth,
        trend=float(np.sign(means.mean())),
        slippage=slippage,
        liquidity=1.0 / max(slippage, 1e-6),
    )


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    lookback: int = 60          # rows used for market-state estimation
    slippage: float = 0.001
    est_cfg: EstimatorConfig = None
    opt_cfg: OptimizerConfig = None

    def __post_init__(self) -> None:
        if self.est_cfg is None:
            self.est_cfg = EstimatorConfig()
        if self.opt_cfg is None:
            self.opt_cfg = OptimizerConfig()


def run_backtest(
    returns: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full backtest over historical strategy returns.

    Parameters
    ----------
    returns : DataFrame [T × S]  — daily returns per strategy, indexed by datetime
    cfg     : BacktestConfig

    Returns
    -------
    allocations : DataFrame of budget weights + cash over time
    portfolio   : DataFrame with portfolio returns and cumulative PnL
    """
    cfg = cfg or BacktestConfig()
    strategy_ids = list(returns.columns)
    engine = AllocatorEngine(strategy_ids, cfg.est_cfg, cfg.opt_cfg)

    inputs: List[AllocatorInput] = []

    for t in range(cfg.lookback, len(returns)):
        ts = returns.index[t]
        window = returns.iloc[t - cfg.lookback : t]
        row = returns.iloc[t]

        market_state = _build_market_state(ts, window, cfg.slippage)

        strategy_outputs = [
            StrategyOutput(
                strategy_id=sid,
                timestamp=ts,
                forecast=float(row[sid]),
                uncertainty=float(window[sid].std()),
                cost_estimate=cfg.slippage,
            )
            for sid in strategy_ids
        ]

        inputs.append(
            AllocatorInput(
                timestamp=ts,
                market_state=market_state,
                strategy_outputs=strategy_outputs,
            )
        )

    allocations = engine.run(inputs)

    # Compute realised portfolio returns
    portfolio = _compute_portfolio_returns(allocations, returns, strategy_ids)

    return allocations, portfolio


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def _compute_portfolio_returns(
    allocations: pd.DataFrame,
    returns: pd.DataFrame,
    strategy_ids: List[str],
) -> pd.DataFrame:
    """Multiply budget weights by next-period realised returns."""
    budget_cols = [f"budget_{sid}" for sid in strategy_ids]
    weights = allocations[budget_cols].rename(
        columns={f"budget_{sid}": sid for sid in strategy_ids}
    )

    # align: weights at t applied to returns at t (same-bar, as a proxy)
    aligned_returns = returns.loc[weights.index]
    port_returns = (weights * aligned_returns).sum(axis=1)

    cum_pnl = (1 + port_returns).cumprod()

    return pd.DataFrame(
        {"portfolio_return": port_returns, "cumulative_pnl": cum_pnl}
    )


def compute_metrics(portfolio: pd.DataFrame) -> Dict[str, float]:
    """Annualised Sharpe, max drawdown, CAGR, hit rate."""
    r = portfolio["portfolio_return"].dropna()
    cum = portfolio["cumulative_pnl"].dropna()

    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = float(drawdown.min())

    n_years = len(r) / 252.0
    cagr = float(cum.iloc[-1] ** (1.0 / max(n_years, 1e-6)) - 1) if len(cum) > 0 else 0.0

    hit_rate = float((r > 0).mean())

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "hit_rate": hit_rate,
        "n_days": len(r),
    }
