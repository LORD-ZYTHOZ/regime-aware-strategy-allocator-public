"""
Allocator engine.

Ties the estimator and optimizer together into a single step-by-step loop.
Each call to `step()` consumes one AllocatorInput and produces one OptimizerOutput.
`run()` processes a full history of AllocatorInputs and returns a log.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from allocator.estimator import (
    EstimatorConfig,
    compute_utility,
    estimate_covariance,
    estimate_edge,
    estimate_risk,
)
from allocator.optimizer import OptimizerConfig, adaptive_gamma, solve_qp
from allocator.schemas import AllocatorInput, OptimizerOutput


class AllocatorEngine:
    """
    Stateful allocator that maintains b_{t-1} and runs the full
    estimate → utility → QP pipeline each step.
    """

    def __init__(
        self,
        strategy_ids: List[str],
        est_cfg: Optional[EstimatorConfig] = None,
        opt_cfg: Optional[OptimizerConfig] = None,
    ) -> None:
        self.strategy_ids = strategy_ids
        self.est_cfg = est_cfg or EstimatorConfig()
        self.opt_cfg = opt_cfg or OptimizerConfig()

        n = len(strategy_ids)
        self._b_prev = np.zeros(n)          # start with zero allocation
        self._return_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Single-step interface
    # ------------------------------------------------------------------

    def step(self, inp: AllocatorInput) -> OptimizerOutput:
        """Process one time step."""
        strategy_map = {s.strategy_id: s for s in inp.strategy_outputs}

        # Build per-strategy arrays in consistent order
        eta_hat = np.array([strategy_map[sid].forecast for sid in self.strategy_ids])
        U = np.array([strategy_map[sid].uncertainty for sid in self.strategy_ids])
        C_hat = np.array([strategy_map[sid].cost_estimate for sid in self.strategy_ids])

        # Accumulate return history (use forecast as proxy until realised returns arrive)
        self._return_history.append(eta_hat.copy())

        # Covariance: need at least 2 observations
        returns_arr = np.array(self._return_history)
        returns_df = pd.DataFrame(returns_arr, columns=self.strategy_ids)

        covs = estimate_covariance(
            returns_df,
            halflife=self.est_cfg.ewm_halflife,
            slippage=pd.Series([inp.market_state.slippage] * len(returns_arr)),
        )
        sigma = covs[-1]

        if np.any(np.isnan(sigma)):
            # not enough history yet — use identity scaled by vol
            sigma = np.eye(len(self.strategy_ids)) * max(inp.market_state.volatility ** 2, 1e-6)

        # Regime-adaptive risk aversion
        gamma = adaptive_gamma(self.opt_cfg.gamma, inp.market_state.volatility)

        # Net utility
        nu = compute_utility(eta_hat, U, C_hat, self.est_cfg.lambda_U, self.est_cfg.lambda_C)

        # Solve QP (create a local config with updated gamma)
        opt_cfg = OptimizerConfig(
            gamma=gamma,
            lambda_turn=self.opt_cfg.lambda_turn,
            max_budget_per_strategy=self.opt_cfg.max_budget_per_strategy,
            de_risk_threshold=self.opt_cfg.de_risk_threshold,
            kill_threshold=self.opt_cfg.kill_threshold,
            solver=self.opt_cfg.solver,
        )

        result = solve_qp(
            nu=nu,
            sigma=sigma,
            b_prev=self._b_prev,
            strategy_ids=self.strategy_ids,
            cfg=opt_cfg,
            timestamp=inp.timestamp,
        )

        # Update state
        self._b_prev = np.array([result.budgets[sid] for sid in self.strategy_ids])

        return result

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def run(self, inputs: List[AllocatorInput]) -> pd.DataFrame:
        """
        Process a list of AllocatorInputs sequentially.

        Returns
        -------
        DataFrame indexed by timestamp with columns:
          - budget_{strategy_id} for each strategy
          - cash
          - objective_value
          - solver_status
          - flag_{strategy_id} for each strategy
        """
        records = []
        for inp in inputs:
            out = self.step(inp)
            row: Dict = {
                "timestamp": out.timestamp,
                "cash": out.cash,
                "objective_value": out.objective_value,
                "solver_status": out.solver_status,
            }
            for sid, bval in out.budgets.items():
                row[f"budget_{sid}"] = bval
            for rf in out.risk_flags:
                row[f"flag_{rf.strategy_id}"] = rf.flag.value
            records.append(row)

        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).set_index("timestamp")
        return df
