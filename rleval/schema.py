"""Core schemas for tasks, execution traces, rewards, and evaluation metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskSpecification(BaseModel):
    task_id: str
    category: str = Field(description="coding | numerical_stability | memory_optimization | ml_pipeline")
    prompt: str
    starter_code: str
    reference_solution: str
    test_suite: str
    time_limit_sec: float = 5.0
    memory_limit_mb: float = 512.0
    numerical_tolerance: float = 1e-5


class ExecutionResult(BaseModel):
    status: str  # passed | failed | timeout | memory_exceeded | syntax_error
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_sec: float = 0.0
    peak_memory_mb: float = 0.0
    passed_tests: int = 0
    total_tests: int = 0
    numerical_error: Optional[float] = None


class RewardSignal(BaseModel):
    sparse_reward: float = 0.0  # 1.0 if all tests passed, 0.0 otherwise
    dense_reward: float = 0.0   # continuous function based on test coverage & efficiency
    efficiency_penalty: float = 0.0
    numerical_penalty: float = 0.0
    total_reward: float = 0.0
    breakdown: Dict[str, float] = {}
