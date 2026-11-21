"""Unit & Integration tests for RL-EvalBench."""
import pytest
from rleval.schema import TaskSpecification
from rleval.sandbox import ExecutionSandbox
from rleval.rewards import RewardEngine
from rleval.tasks import FRONTIER_TASKS
from rleval.environment import MLEvalEnv


def test_sandbox_runs_and_passes_valid_code():
    task = FRONTIER_TASKS[0]  # Stable softmax
    sandbox = ExecutionSandbox()
    result = sandbox.run_task(task, task.reference_solution)
    assert result.status == "passed"
    assert result.exit_code == 0
    assert result.passed_tests == 3
    assert result.numerical_error is not None
    assert result.numerical_error < 1e-4


def test_sandbox_catches_overflow_and_bad_code():
    task = FRONTIER_TASKS[0]
    sandbox = ExecutionSandbox()
    bad_code = """import numpy as np
def stable_softmax(x):
    # Unstable standard exp(x) that overflows on 1000.0
    return np.exp(x) / np.sum(np.exp(x))
"""
    result = sandbox.run_task(task, bad_code)
    assert result.status == "failed"


def test_reward_engine_rewards_passing_penalizes_failing():
    task = FRONTIER_TASKS[0]
    sandbox = ExecutionSandbox()
    reward_engine = RewardEngine()

    good_res = sandbox.run_task(task, task.reference_solution)
    good_reward = reward_engine.compute_reward(task, good_res)
    assert good_reward.total_reward > 0.8
    assert good_reward.sparse_reward == 1.0

    bad_res = sandbox.run_task(task, "def stable_softmax(x): return x")
    bad_reward = reward_engine.compute_reward(task, bad_res)
    assert bad_reward.total_reward < 0.2


def test_gymnasium_environment_lifecycle():
    env = MLEvalEnv()
    obs, info = env.reset(options={"task_id": "task_01_numerically_stable_softmax"})
    assert obs["task_id"] == "task_01_numerically_stable_softmax"

    task = FRONTIER_TASKS[0]
    next_obs, reward, terminated, truncated, step_info = env.step(task.reference_solution)
    assert terminated is True
    assert reward > 0.8
    assert step_info["passed"] is True
