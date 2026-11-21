"""CLI interface for running RL-EvalBench."""
from __future__ import annotations

import argparse
import json
import sys
from rleval.evaluator import run_benchmark_suite


def main():
    parser = argparse.ArgumentParser(description="RL-EvalBench: Frontier RL Environment & Dynamic Verification Engine")
    parser.add_argument("--mode", choices=["oracle", "buggy"], default="oracle", help="Agent simulation mode")
    parser.add_argument("--out", type=str, default="report.json", help="Output path for JSON report")
    args = parser.parse_args()

    print(f"🚀 Running RL-EvalBench across frontier tasks in '{args.mode}' mode...")
    report = run_benchmark_suite(agent_mode=args.mode)

    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Pass@1 Accuracy:   {report['pass_at_1'] * 100:.2f}%")
    print(f"Mean RL Reward:    {report['mean_reward']:.4f}")
    print(f"Passed Tasks:      {report['passed_tasks']} / {report['total_tasks']}")
    print("=" * 60)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full report to {args.out}")


if __name__ == "__main__":
    main()
