"""
Estimation layer.

Computes, for each step t:
  - η̂_t^s  : expected edge (rolling mean return)
  - U_t^s  : risk/uncertainty penalty (rolling vol or CVaR)
  - Σ_t^slv: slippage-adjusted covariance matrix
  - ν_t^s  : net utility = η̂ - λ_U * U - λ_C * Ĉ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EstimatorConfig:
    edge_window: int = 60        # rolling window for mean return (days)
    risk_window: int = 60        # rolling window for vol / CVaR
    cvar_alpha: float = 0.05     # CVaR tail probability
    ewm_halflife: int = 30       # half-life for EWM covariance (days)
    lambda_U: float = 0.05       # risk-penalty weight (tuned for daily-unit vol)
    lambda_C: float = 0.1        # cost-penalty weight
    use_cvar: bool = False        # True → CVaR, False → rolling vol


# ---------------------------------------------------------------------------
# Edge estimation  η̂_t^s
# ---------------------------------------------------------------------------

def estimate_edge(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling mean return as a proxy for expected edge.

    Parameters
    ----------
    returns : DataFrame [T × S]  (T time steps, S strategies)
    window  : look-back window in rows

    Returns
    -------
    DataFrame [T × S] — NaN for first (window-1) rows
    """
    return returns.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# Risk estimation  U_t^s
# ---------------------------------------------------------------------------

def estimate_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling daily volatility (same period units as edge)."""
    return returns.rolling(window=window, min_periods=window).std()


def estimate_cvar(returns: pd.DataFrame, window: int, alpha: float) -> pd.DataFrame:
    """
    Rolling historical CVaR (expected shortfall) at level alpha.
    More sensitive than vol in fat-tailed regimes.
    """
    def _cvar_row(x: np.ndarray) -> float:
        threshold = np.quantile(x, alpha)
        tail = x[x <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else np.nan

    return returns.rolling(window=window, min_periods=window).apply(
        _cvar_row, raw=True
    )


def estimate_risk(
    returns: pd.DataFrame, cfg: EstimatorConfig
) -> pd.DataFrame:
    """Dispatch to vol or CVaR based on config."""
    if cfg.use_cvar:
        return estimate_cvar(returns, cfg.risk_window, cfg.cvar_alpha)
    return estimate_vol(returns, cfg.risk_window)


# ---------------------------------------------------------------------------
# Covariance estimation  Σ_t^slv
# ---------------------------------------------------------------------------

def estimate_covariance(
    returns: pd.DataFrame,
    halflife: int,
    slippage: pd.Series | None = None,
) -> List[np.ndarray]:
    """
    Exponentially weighted covariance matrices, one per time step.

    Optionally inflates diagonal by slippage estimate to produce
    Σ_t^slv (slippage-adjusted covariance).

    Parameters
    ----------
    returns   : DataFrame [T × S]
    halflife  : EWM half-life in rows
    slippage  : Series [T] of scalar slippage multipliers (default 0)

    Returns
    -------
    List of S×S arrays, length T (first rows may be NaN-filled)
    """
    n_steps, n_strats = returns.shape
    covs: List[np.ndarray] = []

    ewm_alpha = 1 - np.exp(-np.log(2) / halflife)

    # accumulate EWM covariance incrementally
    weights = np.zeros(n_strats)
    S = np.zeros((n_strats, n_strats))
    mu = np.zeros(n_strats)
    total_w = 0.0

    for t in range(n_steps):
        r_t = returns.iloc[t].values
        if np.any(np.isnan(r_t)):
            covs.append(np.full((n_strats, n_strats), np.nan))
            continue

        w = ewm_alpha
        total_w = (1 - ewm_alpha) * total_w + w
        mu_new = mu + w / total_w * (r_t - mu)
        diff = r_t - mu_new
        diff_old = r_t - mu
        S = (1 - ewm_alpha) * S + ewm_alpha * np.outer(diff_old, diff)
        mu = mu_new

        cov_t = S / (1.0 if total_w == 0 else 1.0)

        # slippage adjustment: inflate diagonal
        if slippage is not None and not np.isnan(slippage.iloc[t]):
            slip = float(slippage.iloc[t])
            cov_t = cov_t + slip * np.diag(np.diag(cov_t))

        covs.append(cov_t.copy())

    return covs


# ---------------------------------------------------------------------------
# Net utility  ν_t^s
# ---------------------------------------------------------------------------

def compute_utility(
    eta_hat: np.ndarray,
    U: np.ndarray,
    C_hat: np.ndarray,
    lambda_U: float,
    lambda_C: float,
) -> np.ndarray:
    """
    ν_t^s = η̂_t^s − λ_U · U_t^s − λ_C · Ĉ_t^s

    All inputs are 1-D arrays of length S (number of strategies).
    Returns a 1-D array of net utilities.
    """
    return eta_hat - lambda_U * U - lambda_C * C_hat
