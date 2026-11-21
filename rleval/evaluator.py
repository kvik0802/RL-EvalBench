"""Automated Evaluation & Calibration Engine for frontier LLM & Coding Agent benchmarks."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from rleval.schema import ExecutionResult, RewardSignal, TaskSpecification
from rleval.tasks import FRONTIER_TASKS
from rleval.sandbox import ExecutionSandbox
from rleval.rewards import RewardEngine


def run_benchmark_suite(
    tasks: List[TaskSpecification] = FRONTIER_TASKS,
    agent_mode: str = "oracle"  # oracle (reference) | buggy (unoptimized) | baseline
) -> Dict[str, Any]:
    """
    Evaluates an agent on the full frontier task suite, computing:
    - Pass@1 accuracy
    - Mean reward signal (dense & sparse)
    - Average execution latency and peak memory
    - Numerical drift errors
    """
    sandbox = ExecutionSandbox()
    reward_engine = RewardEngine()

    records = []
    total_passed = 0
    total_reward = 0.0

    for task in tasks:
        # Determine code based on benchmark simulation mode
        if agent_mode == "oracle":
            code = task.reference_solution
        elif agent_mode == "buggy":
            # Simulate subtle numerical or memory bugs
            code = task.starter_code.replace("pass", "return np.zeros(10)")
        else:
            code = task.starter_code

        exec_res: ExecutionResult = sandbox.run_task(task, code)
        reward_sig: RewardSignal = reward_engine.compute_reward(task, exec_res)

        if exec_res.status == "passed":
            total_passed += 1
        total_reward += reward_sig.total_reward

        records.append({
            "task_id": task.task_id,
            "category": task.category,
            "status": exec_res.status,
            "exit_code": exec_res.exit_code,
            "duration_sec": round(exec_res.duration_sec, 3),
            "peak_memory_mb": round(exec_res.peak_memory_mb, 2),
            "passed_tests": f"{exec_res.passed_tests}/{exec_res.total_tests}",
            "numerical_error": exec_res.numerical_error,
            "reward": reward_sig.total_reward,
            "reward_breakdown": reward_sig.breakdown,
            "stderr": exec_res.stderr[:200] if exec_res.stderr else ""
        })

    n = len(tasks)
    pass_at_1 = total_passed / n if n > 0 else 0.0
    mean_reward = total_reward / n if n > 0 else 0.0

    return {
        "agent_mode": agent_mode,
        "total_tasks": n,
        "passed_tasks": total_passed,
        "pass_at_1": round(pass_at_1, 4),
        "mean_reward": round(mean_reward, 4),
        "tasks": records
    }
