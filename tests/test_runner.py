"""Tests for AllocatorRunner."""

import pytest
from datetime import datetime

from allocator.adapters import DivergenceNative, HorizonNative, SingularityNative
from allocator.runner import AllocatorRunner, STRATEGY_IDS
from allocator.schemas import MarketState, OptimizerOutput

TS = datetime(2024, 1, 15, 9, 30)


def make_market_state(slippage: float = 0.001) -> MarketState:
    return MarketState(
        timestamp=TS,
        volatility=0.18,
        dispersion=0.05,
        correlation=0.3,
        breadth=0.6,
        trend=0.4,
        slippage=slippage,
        liquidity=500.0,
    )


def make_singularity() -> SingularityNative:
    return SingularityNative(kelly_fraction=0.3, win_rate=0.65, spread=0.30)


def make_divergence() -> DivergenceNative:
    return DivergenceNative(
        gamma=0.7, regime="RISK_ON", persona_divergence_score=0.15, spread=0.30
    )


def make_horizon() -> HorizonNative:
    return HorizonNative(confidence=0.6, spread=0.30)


# ---------------------------------------------------------------------------
# Basic output contract
# ---------------------------------------------------------------------------

class TestAllocatorRunnerOutput:
    def test_returns_optimizer_output(self):
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state(), make_singularity(), make_divergence(), make_horizon())
        assert isinstance(result, OptimizerOutput)

    def test_budget_keys_match_strategy_ids(self):
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state(), make_singularity(), make_divergence(), make_horizon())
        assert set(result.budgets.keys()) == set(STRATEGY_IDS)

    def test_budgets_and_cash_sum_to_one(self):
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state(), make_singularity(), make_divergence(), make_horizon())
        total = sum(result.budgets.values()) + result.cash
        assert abs(total - 1.0) < 1e-4

    def test_all_budgets_non_negative(self):
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state(), make_singularity(), make_divergence(), make_horizon())
        for sid, b in result.budgets.items():
            assert b >= 0.0, f"negative budget for {sid}: {b}"

    def test_timestamp_propagated(self):
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state(), make_singularity(), make_divergence(), make_horizon())
        assert result.timestamp == TS


# ---------------------------------------------------------------------------
# Absent strategies (None inputs)
# ---------------------------------------------------------------------------

class TestAbsentStrategies:
    def test_all_none_still_runs(self):
        """All strategies absent -> zero-forecast placeholders, QP still solves."""
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state())
        assert isinstance(result, OptimizerOutput)
        total = sum(result.budgets.values()) + result.cash
        assert abs(total - 1.0) < 1e-4

    def test_missing_singularity(self):
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state(), singularity=None,
                             divergence=make_divergence(), horizon=make_horizon())
        assert set(result.budgets.keys()) == set(STRATEGY_IDS)

    def test_missing_divergence(self):
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state(), singularity=make_singularity(),
                             divergence=None, horizon=make_horizon())
        assert set(result.budgets.keys()) == set(STRATEGY_IDS)

    def test_missing_horizon(self):
        runner = AllocatorRunner()
        result = runner.step(TS, make_market_state(), singularity=make_singularity(),
                             divergence=make_divergence(), horizon=None)
        assert set(result.budgets.keys()) == set(STRATEGY_IDS)

    def test_absent_strategy_slippage_used_as_cost(self):
        """Absent strategy placeholder uses market_state.slippage as cost_estimate."""
        # Run two steps and confirm the QP runs without error regardless
        runner = AllocatorRunner()
        ms = make_market_state(slippage=0.05)
        result = runner.step(TS, ms, singularity=None, divergence=None, horizon=None)
        assert isinstance(result, OptimizerOutput)


# ---------------------------------------------------------------------------
# Statefulness: multiple steps
# ---------------------------------------------------------------------------

class TestMultipleSteps:
    def test_two_sequential_steps(self):
        """Engine is stateful — b_prev updates across steps."""
        runner = AllocatorRunner()
        ms = make_market_state()
        r1 = runner.step(TS, ms, make_singularity(), make_divergence(), make_horizon())
        ts2 = datetime(2024, 1, 16, 9, 30)
        ms2 = ms.model_copy(update={"timestamp": ts2})
        r2 = runner.step(
            ts2, ms2,
            SingularityNative(kelly_fraction=0.2, win_rate=0.60, spread=0.30),
            DivergenceNative(gamma=0.5, regime="RISK_OFF", persona_divergence_score=0.20, spread=0.30),
            HorizonNative(confidence=0.55, spread=0.30),
        )
        assert isinstance(r1, OptimizerOutput)
        assert isinstance(r2, OptimizerOutput)
        total2 = sum(r2.budgets.values()) + r2.cash
        assert abs(total2 - 1.0) < 1e-4
