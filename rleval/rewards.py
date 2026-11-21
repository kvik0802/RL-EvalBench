"""Reward Modeling Engine for Reinforcement Learning (PPO, GRPO, RLAIF)."""
from __future__ import annotations

from typing import Dict, Any
from rleval.schema import ExecutionResult, RewardSignal, TaskSpecification


class RewardEngine:
    """
    Computes fine-grained reward signals based on:
    - Functional Correctness (test coverage)
    - Memory & Latency Efficiency (penalties for slow/bloated code)
    - Numerical Stability & Floating-Point Drift
    """
    def __init__(
        self,
        pass_bonus: float = 1.0,
        fail_penalty: float = -0.5,
        latency_weight: float = 0.1,
        memory_weight: float = 0.05,
        numerical_weight: float = 0.5
    ):
        self.pass_bonus = pass_bonus
        self.fail_penalty = fail_penalty
        self.latency_weight = latency_weight
        self.memory_weight = memory_weight
        self.numerical_weight = numerical_weight

    def compute_reward(self, task: TaskSpecification, result: ExecutionResult) -> RewardSignal:
        if result.status == "passed":
            sparse = self.pass_bonus
        else:
            sparse = self.fail_penalty

        # Dense reward based on passed test ratio
        if result.total_tests > 0:
            test_ratio = result.passed_tests / result.total_tests
            dense = test_ratio * self.pass_bonus
        else:
            dense = 1.0 if result.status == "passed" else 0.0

        # Efficiency penalty: penalize approaching limits
        lat_ratio = min(result.duration_sec / max(task.time_limit_sec, 0.001), 1.0)
        mem_ratio = min(result.peak_memory_mb / max(task.memory_limit_mb, 1.0), 1.0)
        eff_penalty = (lat_ratio * self.latency_weight) + (mem_ratio * self.memory_weight)

        # Numerical stability penalty
        num_penalty = 0.0
        if result.numerical_error is not None:
            if result.numerical_error > task.numerical_tolerance:
                num_penalty = min(result.numerical_error * self.numerical_weight, 1.0)

        total = dense - eff_penalty - num_penalty

        breakdown = {
            "sparse_reward": round(sparse, 4),
            "test_dense_reward": round(dense, 4),
            "efficiency_penalty": round(eff_penalty, 4),
            "numerical_penalty": round(num_penalty, 4)
        }

        return RewardSignal(
            sparse_reward=sparse,
            dense_reward=dense,
            efficiency_penalty=eff_penalty,
            numerical_penalty=num_penalty,
            total_reward=round(total, 4),
            breakdown=breakdown
        )
