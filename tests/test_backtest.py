"""Tests for the backtest and stress harnesses."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from allocator.backtest import BacktestConfig, compute_metrics, run_backtest
from allocator.stress import (
    SyntheticScenario,
    apply_synthetic_scenario,
    run_all_synthetic_scenarios,
    run_synthetic_stress,
)


def _make_returns(n: int = 300, s: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = rng.normal(0.0005, 0.01, size=(n, s))
    return pd.DataFrame(data, index=dates, columns=[f"strat_{i}" for i in range(s)])


class TestBacktest:
    def test_returns_dataframes(self):
        returns = _make_returns()
        allocs, portfolio = run_backtest(returns)
        assert isinstance(allocs, pd.DataFrame)
        assert isinstance(portfolio, pd.DataFrame)
        assert len(allocs) > 0

    def test_budget_columns_present(self):
        returns = _make_returns()
        allocs, _ = run_backtest(returns)
        for col in ["budget_strat_0", "budget_strat_1", "budget_strat_2"]:
            assert col in allocs.columns

    def test_budgets_non_negative(self):
        returns = _make_returns()
        allocs, _ = run_backtest(returns)
        budget_cols = [c for c in allocs.columns if c.startswith("budget_")]
        assert (allocs[budget_cols] >= -1e-6).all().all()

    def test_budgets_sum_to_at_most_one(self):
        returns = _make_returns()
        allocs, _ = run_backtest(returns)
        budget_cols = [c for c in allocs.columns if c.startswith("budget_")]
        row_sums = allocs[budget_cols].sum(axis=1) + allocs["cash"]
        assert (row_sums <= 1.0 + 1e-4).all()

    def test_metrics_keys(self):
        returns = _make_returns()
        _, portfolio = run_backtest(returns)
        metrics = compute_metrics(portfolio)
        for key in ["sharpe", "max_drawdown", "cagr", "hit_rate"]:
            assert key in metrics


class TestStress:
    def test_synthetic_scenario_shape(self):
        returns = _make_returns()
        scenario = SyntheticScenario("test", vol_multiplier=2.0)
        stressed = apply_synthetic_scenario(returns, scenario)
        assert stressed.shape == (scenario.n_days, returns.shape[1])

    def test_synthetic_stress_run(self):
        returns = _make_returns()
        scenario = SyntheticScenario("vol_spike", vol_multiplier=2.0)
        allocs, portfolio, metrics = run_synthetic_stress(returns, scenario)
        assert "sharpe" in metrics

    def test_all_scenarios_return_summary(self):
        returns = _make_returns()
        summary = run_all_synthetic_scenarios(returns)
        assert len(summary) > 0
        assert "sharpe" in summary.columns or "error" in summary.columns
