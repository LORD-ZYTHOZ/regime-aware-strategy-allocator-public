"""Tests for the QP optimizer."""

import numpy as np
import pytest
from datetime import datetime

from allocator.optimizer import OptimizerConfig, solve_qp, adaptive_gamma
from allocator.schemas import RiskFlag

TS = datetime(2024, 1, 15)
STRATS = ["s1", "s2", "s3"]


def _eye_sigma(n: int, scale: float = 0.01) -> np.ndarray:
    return np.eye(n) * scale


class TestSolveQP:
    def test_budgets_sum_to_at_most_one(self):
        nu = np.array([0.1, 0.05, 0.02])
        sigma = _eye_sigma(3)
        b_prev = np.zeros(3)
        out = solve_qp(nu, sigma, b_prev, STRATS, OptimizerConfig(), TS)
        assert sum(out.budgets.values()) + out.cash <= 1.0 + 1e-6

    def test_non_negative_budgets(self):
        nu = np.array([-0.5, -0.3, -0.1])
        sigma = _eye_sigma(3)
        b_prev = np.zeros(3)
        out = solve_qp(nu, sigma, b_prev, STRATS, OptimizerConfig(), TS)
        assert all(v >= -1e-6 for v in out.budgets.values())

    def test_high_utility_gets_more_budget(self):
        nu = np.array([1.0, 0.01, 0.01])
        sigma = _eye_sigma(3)
        b_prev = np.zeros(3)
        out = solve_qp(nu, sigma, b_prev, STRATS, OptimizerConfig(), TS)
        assert out.budgets["s1"] > out.budgets["s2"]

    def test_risk_flags_kill(self):
        nu = np.array([-2.0, 0.1, 0.1])
        sigma = _eye_sigma(3)
        b_prev = np.zeros(3)
        cfg = OptimizerConfig(kill_threshold=-1.0)
        out = solve_qp(nu, sigma, b_prev, STRATS, cfg, TS)
        flags = {rf.strategy_id: rf.flag for rf in out.risk_flags}
        assert flags["s1"] == RiskFlag.KILL

    def test_turnover_penalty_reduces_change(self):
        nu = np.array([0.5, 0.5, 0.5])
        sigma = _eye_sigma(3)
        b_prev = np.array([0.3, 0.3, 0.3])

        low_turn = solve_qp(nu, sigma, b_prev, STRATS, OptimizerConfig(lambda_turn=0.0), TS)
        high_turn = solve_qp(nu, sigma, b_prev, STRATS, OptimizerConfig(lambda_turn=10.0), TS)

        low_b = np.array(list(low_turn.budgets.values()))
        high_b = np.array(list(high_turn.budgets.values()))

        low_turnover = np.abs(low_b - b_prev).sum()
        high_turnover = np.abs(high_b - b_prev).sum()
        assert high_turnover <= low_turnover + 1e-4

    def test_cash_is_complement(self):
        nu = np.array([0.1, 0.1])
        sigma = _eye_sigma(2)
        b_prev = np.zeros(2)
        out = solve_qp(nu, sigma, b_prev, ["s1", "s2"], OptimizerConfig(), TS)
        total = sum(out.budgets.values()) + out.cash
        assert abs(total - 1.0) < 1e-4


class TestAdaptiveGamma:
    def test_high_vol_increases_gamma(self):
        g_low = adaptive_gamma(1.0, market_vol=0.10, vol_target=0.15)
        g_high = adaptive_gamma(1.0, market_vol=0.30, vol_target=0.15)
        assert g_high > g_low

    def test_zero_vol_does_not_crash(self):
        g = adaptive_gamma(1.0, market_vol=0.0)
        assert g >= 0
