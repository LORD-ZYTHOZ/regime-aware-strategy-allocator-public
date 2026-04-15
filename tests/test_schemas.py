"""Tests for data schemas."""

import pytest
from datetime import datetime
from allocator.schemas import (
    StrategyOutput,
    MarketState,
    AllocatorInput,
    OptimizerOutput,
    RiskFlag,
    StrategyRiskFlag,
)

TS = datetime(2024, 1, 15, 9, 30)


def make_strategy_output(sid: str = "s1", forecast: float = 0.01) -> StrategyOutput:
    return StrategyOutput(
        strategy_id=sid,
        timestamp=TS,
        forecast=forecast,
        uncertainty=0.15,
        cost_estimate=0.001,
    )


def make_market_state() -> MarketState:
    return MarketState(
        timestamp=TS,
        volatility=0.18,
        dispersion=0.05,
        correlation=0.3,
        breadth=0.6,
        trend=0.4,
        slippage=0.001,
        liquidity=500.0,
    )


class TestStrategyOutput:
    def test_valid(self):
        s = make_strategy_output()
        assert s.strategy_id == "s1"
        assert s.forecast == 0.01

    def test_negative_uncertainty_rejected(self):
        with pytest.raises(Exception):
            StrategyOutput(
                strategy_id="s1", timestamp=TS,
                forecast=0.01, uncertainty=-0.1, cost_estimate=0.001,
            )

    def test_negative_cost_rejected(self):
        with pytest.raises(Exception):
            StrategyOutput(
                strategy_id="s1", timestamp=TS,
                forecast=0.01, uncertainty=0.1, cost_estimate=-0.001,
            )


class TestAllocatorInput:
    def test_valid(self):
        inp = AllocatorInput(
            timestamp=TS,
            market_state=make_market_state(),
            strategy_outputs=[make_strategy_output("s1"), make_strategy_output("s2")],
        )
        assert len(inp.strategy_outputs) == 2

    def test_timestamp_mismatch_rejected(self):
        wrong_ts = datetime(2024, 1, 16, 9, 30)
        bad_strategy = make_strategy_output("s1")
        bad_strategy = bad_strategy.model_copy(update={"timestamp": wrong_ts})
        with pytest.raises(Exception):
            AllocatorInput(
                timestamp=TS,
                market_state=make_market_state(),
                strategy_outputs=[bad_strategy],
            )


class TestOptimizerOutput:
    def test_valid(self):
        out = OptimizerOutput(
            timestamp=TS,
            budgets={"s1": 0.4, "s2": 0.3},
            cash=0.3,
            solver_status="optimal",
        )
        assert abs(out.cash - 0.3) < 1e-9

    def test_budgets_not_summing_to_one_rejected(self):
        with pytest.raises(Exception):
            OptimizerOutput(
                timestamp=TS,
                budgets={"s1": 0.6, "s2": 0.6},
                cash=0.3,
            )

    def test_risk_flags(self):
        out = OptimizerOutput(
            timestamp=TS,
            budgets={"s1": 0.5},
            cash=0.5,
            risk_flags=[StrategyRiskFlag(strategy_id="s1", flag=RiskFlag.KILL)],
        )
        assert out.risk_flags[0].flag == RiskFlag.KILL
