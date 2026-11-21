"""Challenging Machine Learning & Systems Tasks for Evaluation & RL Training."""
from __future__ import annotations

from typing import List, Dict
from rleval.schema import TaskSpecification

FRONTIER_TASKS: List[TaskSpecification] = [
    TaskSpecification(
        task_id="task_01_numerically_stable_softmax",
        category="numerical_stability",
        prompt="Implement a numerically stable softmax function that avoids floating point overflow on large inputs.",
        starter_code="""import numpy as np

def stable_softmax(x: np.ndarray) -> np.ndarray:
    # TODO: Implement numerically stable softmax
    pass
""",
        reference_solution="""import numpy as np

def stable_softmax(x: np.ndarray) -> np.ndarray:
    shift_x = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=-1, keepdims=True)
""",
        test_suite="""import numpy as np

# Test 1: Shape & Normal sum
x1 = np.array([1.0, 2.0, 3.0])
out1 = stable_softmax(x1)
assert np.isclose(np.sum(out1), 1.0), "Probabilities must sum to 1"

# Test 2: Extreme values (overflow test)
x_large = np.array([1000.0, 1001.0, 1002.0])
out_large = stable_softmax(x_large)
assert not np.isnan(out_large).any(), "Overflow detected (NaNs returned)"
assert not np.isinf(out_large).any(), "Overflow detected (Infs returned)"
assert np.isclose(np.sum(out_large), 1.0), "Large sum must be 1.0"

# Test 3: Numerical error compared to reference
ref = np.array([0.09003057, 0.24472847, 0.66524096])
err = np.max(np.abs(out_large - ref))
print(f"NUMERICAL_ERROR: {err}")
print("TESTS_PASSED: 3/3")
""",
        numerical_tolerance=1e-5
    ),
    TaskSpecification(
        task_id="task_02_memory_efficient_attention_mask",
        category="memory_optimization",
        prompt="Implement a causal attention mask generator that allocates minimal memory without large intermediate dense matrices.",
        starter_code="""import numpy as np

def build_causal_mask(seq_len: int) -> np.ndarray:
    # TODO: Generate lower-triangular causal mask
    pass
""",
        reference_solution="""import numpy as np

def build_causal_mask(seq_len: int) -> np.ndarray:
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))
""",
        test_suite="""import numpy as np

mask = build_causal_mask(4)
expected = np.array([
    [True, False, False, False],
    [True, True, False, False],
    [True, True, True, False],
    [True, True, True, True]
])
assert np.array_equal(mask, expected), "Mask is not lower triangular"
print("TESTS_PASSED: 2/2")
""",
        time_limit_sec=3.0,
        memory_limit_mb=256.0
    ),
    TaskSpecification(
        task_id="task_03_streaming_moving_average",
        category="ml_pipeline",
        prompt="Implement an online O(1) space streaming exponentially weighted moving average (EWMA).",
        starter_code="""class OnlineEWMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        # TODO: update online moving average
        pass
""",
        reference_solution="""class OnlineEWMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value
""",
        test_suite="""ewma = OnlineEWMA(alpha=0.5)
assert ewma.update(10.0) == 10.0
assert ewma.update(20.0) == 15.0
assert ewma.update(30.0) == 22.5
print("TESTS_PASSED: 3/3")
"""
    )
]


def get_task_by_id(task_id: str) -> TaskSpecification | None:
    for t in FRONTIER_TASKS:
        if t.task_id == task_id:
            return t
    return None
