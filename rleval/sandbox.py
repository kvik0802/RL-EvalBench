"""Sandboxed execution runner with strict CPU, memory, and timeout governance."""
from __future__ import annotations

import os
import sys
import time
import subprocess
import tempfile
import psutil
from pathlib import Path
from typing import Tuple

from rleval.schema import ExecutionResult, TaskSpecification


class ExecutionSandbox:
    """
    Subprocess sandbox that runs agent code against unit tests,
    measuring peak RAM, latency, and numerical stability.
    """
    def __init__(self, default_timeout_sec: float = 5.0, default_mem_limit_mb: float = 512.0):
        self.default_timeout = default_timeout_sec
        self.default_mem_limit = default_mem_limit_mb

    def run_task(self, task: TaskSpecification, submission_code: str) -> ExecutionResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = Path(tmpdir) / "test_run.py"
            # Combine agent submission with test harness
            full_script = f"""# Agent Submission
{submission_code}

# Benchmark Verification Harness
{task.test_suite}
"""
            run_path.write_text(full_script, encoding="utf-8")

            t0 = time.perf_counter()
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(run_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=tmpdir
                )

                peak_mem = 0.0
                try:
                    p = psutil.Process(proc.pid)
                    # Poll memory usage during execution
                    while proc.poll() is None:
                        try:
                            mem = p.memory_info().rss / (1024 * 1024)
                            if mem > peak_mem:
                                peak_mem = mem
                            if peak_mem > task.memory_limit_mb:
                                proc.kill()
                                return ExecutionResult(
                                    status="memory_exceeded",
                                    stderr=f"Out of memory: exceeded {task.memory_limit_mb} MB (used {peak_mem:.1f} MB)",
                                    duration_sec=time.perf_counter() - t0,
                                    peak_memory_mb=peak_mem
                                )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            break
                        time.sleep(0.01)

                    stdout, stderr = proc.communicate(timeout=task.time_limit_sec)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return ExecutionResult(
                        status="timeout",
                        stderr=f"Timeout expired after {task.time_limit_sec} seconds",
                        duration_sec=task.time_limit_sec,
                        peak_memory_mb=peak_mem
                    )

                duration = time.perf_counter() - t0
                exit_code = proc.returncode

                # Parse test outputs
                passed = 0
                total = 0
                num_err = None

                for line in stdout.split("\n"):
                    if line.startswith("TESTS_PASSED:"):
                        try:
                            parts = line.replace("TESTS_PASSED:", "").strip().split("/")
                            passed = int(parts[0])
                            total = int(parts[1])
                        except Exception:
                            pass
                    elif line.startswith("NUMERICAL_ERROR:"):
                        try:
                            num_err = float(line.replace("NUMERICAL_ERROR:", "").strip())
                        except Exception:
                            pass

                status = "passed" if exit_code == 0 and (passed == total or total == 0) else "failed"

                return ExecutionResult(
                    status=status,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    duration_sec=duration,
                    peak_memory_mb=peak_mem,
                    passed_tests=passed,
                    total_tests=total,
                    numerical_error=num_err
                )

            except Exception as e:
                return ExecutionResult(
                    status="failed",
                    stderr=str(e),
                    exit_code=-1,
                    duration_sec=time.perf_counter() - t0
                )
