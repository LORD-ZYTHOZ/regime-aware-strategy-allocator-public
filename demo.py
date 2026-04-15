"""
Quick demo — run the allocator on synthetic strategy returns and print results.

Usage:
    .venv/bin/python demo.py
    .venv/bin/python demo.py --real   # pulls real ETF data via yfinance
"""

import argparse
import sys

import numpy as np
import pandas as pd

from allocator.backtest import BacktestConfig, compute_metrics, run_backtest
from allocator.stress import run_all_synthetic_scenarios


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_synthetic_returns(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Three synthetic strategies with distinct risk/return profiles."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)

    # ASIA-like: moderate edge, low vol
    asia = rng.normal(0.0008, 0.008, n)

    # LONDON-like: lower edge, moderate vol, some trending periods
    trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.001
    london = rng.normal(0.0003, 0.012, n) + trend

    # NY-like: high vol, inconsistent edge (simulate a broken strategy)
    ny_edge = np.where(np.arange(n) < 300, 0.0006, -0.0003)  # goes bad after day 300
    ny = rng.normal(ny_edge, 0.018, n)

    return pd.DataFrame(
        {"ASIA": asia, "LONDON": london, "NY": ny},
        index=dates,
    )


def make_real_returns() -> pd.DataFrame:
    """Pull ETF returns as strategy proxies (requires yfinance)."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed — pip install yfinance")
        sys.exit(1)

    tickers = ["GLD", "SPY", "TLT"]   # Gold, S&P500, Bonds as strategy proxies
    raw = yf.download(tickers, start="2020-01-01", progress=False)["Close"]
    returns = raw.pct_change().dropna()
    returns.columns = ["GOLD_strat", "EQUITY_strat", "BOND_strat"]
    return returns


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_allocations(allocs: pd.DataFrame, n: int = 10) -> None:
    budget_cols = [c for c in allocs.columns if c.startswith("budget_")]
    flag_cols   = [c for c in allocs.columns if c.startswith("flag_")]

    display = allocs[budget_cols + ["cash"]].tail(n).copy()
    display.columns = [c.replace("budget_", "") for c in display.columns]
    display = display.round(4)

    print(display.to_string())

    if flag_cols:
        print("\nRisk flags (last 5 rows):")
        print(allocs[flag_cols].tail(5).to_string())


def print_metrics(metrics: dict) -> None:
    print(f"  Sharpe ratio  : {metrics['sharpe']:>8.3f}")
    print(f"  CAGR          : {metrics['cagr']:>8.1%}")
    print(f"  Max drawdown  : {metrics['max_drawdown']:>8.1%}")
    print(f"  Hit rate      : {metrics['hit_rate']:>8.1%}")
    print(f"  Days traded   : {metrics['n_days']:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real ETF data")
    args = parser.parse_args()

    # ── Data ──────────────────────────────────────────────────────────────
    print_section("Loading data")
    if args.real:
        returns = make_real_returns()
        print(f"  Real ETF returns: {returns.shape[0]} days × {returns.shape[1]} strategies")
    else:
        returns = make_synthetic_returns()
        print(f"  Synthetic returns: {returns.shape[0]} days × {returns.shape[1]} strategies")
        print("  ASIA  → moderate edge, low vol")
        print("  LONDON→ lower edge, trending noise")
        print("  NY    → goes bad after day 300 (simulated broken strategy)")

    # ── Backtest ──────────────────────────────────────────────────────────
    print_section("Running backtest")
    cfg = BacktestConfig()
    allocations, portfolio = run_backtest(returns, cfg)
    metrics = compute_metrics(portfolio)

    print(f"\nLast {min(10, len(allocations))} allocation snapshots:")
    print_allocations(allocations)

    print("\nPerformance metrics:")
    print_metrics(metrics)

    # ── Allocation drift ──────────────────────────────────────────────────
    print_section("Allocation over time (sampled every 50 days)")
    budget_cols = [c for c in allocations.columns if c.startswith("budget_")]
    sample = allocations[budget_cols + ["cash"]].iloc[::50].round(3)
    sample.columns = [c.replace("budget_", "") for c in sample.columns]
    print(sample.to_string())

    # ── Stress tests ──────────────────────────────────────────────────────
    print_section("Stress test scenarios")
    summary = run_all_synthetic_scenarios(returns, cfg)
    if "sharpe" in summary.columns:
        print(summary[["sharpe", "max_drawdown", "cagr"]].round(3).to_string())
    else:
        print(summary.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
