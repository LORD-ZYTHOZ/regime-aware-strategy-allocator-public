"""
Stress-testing harness.

Runs the allocator through:
  1. Historical crisis scenarios (2008 GFC, 2020 COVID, 2022 rate shock)
  2. Synthetic stress scenarios (vol shock, correlation spike, liquidity crunch)

Each scenario returns allocations + portfolio metrics so you can validate
that the allocator de-risks or kills strategies before real money is at stake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from allocator.backtest import BacktestConfig, compute_metrics, run_backtest


# ---------------------------------------------------------------------------
# Historical crisis date ranges (approximate)
# ---------------------------------------------------------------------------

HISTORICAL_CRISES: Dict[str, Tuple[str, str]] = {
    "gfc_2008": ("2008-09-01", "2009-03-31"),
    "euro_debt_2011": ("2011-07-01", "2011-12-31"),
    "covid_2020": ("2020-02-01", "2020-04-30"),
    "rate_shock_2022": ("2022-01-01", "2022-12-31"),
}


# ---------------------------------------------------------------------------
# Synthetic scenario builder
# ---------------------------------------------------------------------------

@dataclass
class SyntheticScenario:
    name: str
    vol_multiplier: float = 1.0      # scale all strategy vols
    corr_addition: float = 0.0       # add to all pairwise correlations (clamp to ±1)
    return_shock: float = 0.0        # additive daily return shock (negative = drawdown)
    slippage_multiplier: float = 1.0
    n_days: int = 120


def apply_synthetic_scenario(
    returns: pd.DataFrame, scenario: SyntheticScenario
) -> pd.DataFrame:
    """
    Apply synthetic shocks to a returns DataFrame.

    Steps:
    1. Compute sample covariance and Cholesky factor.
    2. Reconstruct returns with inflated vol + added correlation.
    3. Add a flat return shock.
    """
    rng = np.random.default_rng(seed=42)
    n, s = scenario.n_days, len(returns.columns)

    # Base stats from original
    mu = returns.mean().values
    std = returns.std().values * scenario.vol_multiplier
    corr = returns.corr().values

    # Inflate correlations
    new_corr = np.clip(corr + scenario.corr_addition, -0.999, 0.999)
    np.fill_diagonal(new_corr, 1.0)

    # Rebuild covariance
    sigma = np.outer(std, std) * new_corr

    # Ensure PSD via nearest positive definite
    sigma = _nearest_psd(sigma)

    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        L = np.diag(std)

    # Simulate
    z = rng.standard_normal((n, s))
    sim = z @ L.T + mu + scenario.return_shock

    # Create new dates
    last_date = returns.index[-1] if len(returns) > 0 else pd.Timestamp("2020-01-01")
    new_dates = pd.bdate_range(start=last_date, periods=n + 1)[1:]

    return pd.DataFrame(sim, index=new_dates[: len(sim)], columns=returns.columns)


def _nearest_psd(A: np.ndarray) -> np.ndarray:
    """Higham's nearest positive semi-definite matrix."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if _is_pd(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not _is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
    return A3


def _is_pd(B: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------------------------
# Stress test runner
# ---------------------------------------------------------------------------

def run_historical_stress(
    full_returns: pd.DataFrame,
    crisis_name: str,
    cfg: Optional[BacktestConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Slice historical returns to a crisis window and backtest.

    Parameters
    ----------
    full_returns : full history DataFrame
    crisis_name  : key from HISTORICAL_CRISES
    cfg          : backtest config

    Returns
    -------
    allocations, portfolio, metrics
    """
    if crisis_name not in HISTORICAL_CRISES:
        raise ValueError(f"Unknown crisis: {crisis_name}. Choose from {list(HISTORICAL_CRISES)}")

    start, end = HISTORICAL_CRISES[crisis_name]
    window = full_returns.loc[start:end]

    if len(window) < 30:
        raise ValueError(f"Not enough data for {crisis_name} ({len(window)} rows). Provide full history.")

    allocations, portfolio = run_backtest(window, cfg)
    metrics = compute_metrics(portfolio)
    return allocations, portfolio, metrics


def run_synthetic_stress(
    base_returns: pd.DataFrame,
    scenario: SyntheticScenario,
    cfg: Optional[BacktestConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Generate a synthetic stress scenario and backtest through it.

    Returns
    -------
    allocations, portfolio, metrics
    """
    stressed = apply_synthetic_scenario(base_returns, scenario)
    allocations, portfolio = run_backtest(stressed, cfg)
    metrics = compute_metrics(portfolio)
    return allocations, portfolio, metrics


def run_all_synthetic_scenarios(
    base_returns: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    """
    Run a standard battery of synthetic stress scenarios.

    Returns a summary DataFrame with metrics for each scenario.
    """
    scenarios = [
        SyntheticScenario("baseline"),
        SyntheticScenario("vol_spike_2x", vol_multiplier=2.0),
        SyntheticScenario("vol_spike_3x", vol_multiplier=3.0),
        SyntheticScenario("corr_spike_0.5", corr_addition=0.5),
        SyntheticScenario("corr_spike_0.8", corr_addition=0.8),
        SyntheticScenario("return_shock_-1pct", return_shock=-0.01),
        SyntheticScenario("return_shock_-3pct", return_shock=-0.03),
        SyntheticScenario("liquidity_crunch", slippage_multiplier=5.0, vol_multiplier=2.0, corr_addition=0.4),
        SyntheticScenario("perfect_storm", vol_multiplier=3.0, corr_addition=0.7, return_shock=-0.02, slippage_multiplier=5.0),
    ]

    results = []
    for scenario in scenarios:
        try:
            _, _, metrics = run_synthetic_stress(base_returns, scenario, cfg)
            results.append({"scenario": scenario.name, **metrics})
        except Exception as e:
            results.append({"scenario": scenario.name, "error": str(e)})

    return pd.DataFrame(results).set_index("scenario")
