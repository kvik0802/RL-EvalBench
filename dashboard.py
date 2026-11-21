"""Interactive Streamlit Dashboard for RL-EvalBench."""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Root setup
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rleval.tasks import FRONTIER_TASKS
from rleval.sandbox import ExecutionSandbox
from rleval.rewards import RewardEngine
from rleval.evaluator import run_benchmark_suite

st.set_page_config(
    page_title="RL-EvalBench — Frontier RL & Eval Platform",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 RL-EvalBench: Frontier RL Environment & Dynamic Verification Engine")
st.caption("Designed for training & evaluating autonomous agents on challenging numerical, memory, and ML engineering tasks.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Evaluation Control")
    mode = st.radio("Agent Mode", ["oracle (Reference Solutions)", "buggy (Degraded/Buggy Implementation)"])
    agent_mode_key = "oracle" if "oracle" in mode else "buggy"

    if st.button("🚀 Run Full Benchmark Suite", type="primary"):
        with st.spinner("Executing tasks in isolated sandbox..."):
            st.session_state["report"] = run_benchmark_suite(agent_mode=agent_mode_key)

if "report" not in st.session_state:
    st.session_state["report"] = run_benchmark_suite(agent_mode="oracle")

report = st.session_state["report"]

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pass@1 Accuracy", f"{report['pass_at_1'] * 100:.1f}%")
c2.metric("Mean RL Reward", f"{report['mean_reward']:.3f}")
c3.metric("Passed Tasks", f"{report['passed_tasks']} / {report['total_tasks']}")
c4.metric("Environment", "Gymnasium v0.29+")

st.divider()

# Interactive Task Explorer
st.subheader("📋 Task Execution Matrix")
df = pd.DataFrame(report["tasks"])
st.dataframe(df[["task_id", "category", "status", "duration_sec", "peak_memory_mb", "passed_tests", "reward"]], use_container_width=True)

# Visualizations
col_left, col_right = st.columns(2)
with col_left:
    fig_lat = px.bar(df, x="task_id", y="duration_sec", color="status", title="Execution Latency per Task (seconds)", text_auto=True)
    st.plotly_chart(fig_lat, use_container_width=True)

with col_right:
    fig_mem = px.bar(df, x="task_id", y="peak_memory_mb", color="status", title="Peak Memory Consumption (MB)", text_auto=True)
    st.plotly_chart(fig_mem, use_container_width=True)

# Code Runner Sandbox
st.divider()
st.subheader("🛠️ Interactive Sandbox Task Solver")
selected_id = st.selectbox("Select Task to Solve / Verify:", [t.task_id for t in FRONTIER_TASKS])
selected_task = next(t for t in FRONTIER_TASKS if t.task_id == selected_id)

st.markdown(f"**Prompt:** {selected_task.prompt}")
code_input = st.text_area("Python Implementation:", value=selected_task.reference_solution, height=220)

if st.button("⚡ Execute & Verify in Sandbox"):
    sandbox = ExecutionSandbox()
    reward_engine = RewardEngine()
    with st.spinner("Running in sandboxed environment..."):
        res = sandbox.run_task(selected_task, code_input)
        sig = reward_engine.compute_reward(selected_task, res)

    if res.status == "passed":
        st.success(f"✅ PASSED (Exit Code 0) | Duration: {res.duration_sec:.3f}s | Peak RAM: {res.peak_memory_mb:.1f}MB")
    else:
        st.error(f"❌ {res.status.upper()}: {res.stderr}")

    st.json({
        "status": res.status,
        "reward": sig.total_reward,
        "reward_breakdown": sig.breakdown,
        "numerical_error": res.numerical_error
    })
