"""Page 3: Benchmark Explorer — Plotly CI bars, ROUGE vs faithfulness, raw CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.data import get_eval_bundle, show_empty_state  # noqa: E402
from core.plotly_charts import (  # noqa: E402
    METRIC_OPTIONS,
    build_arm_bar_ci_figure,
    build_raw_metrics_table,
    build_rouge_vs_faithfulness_figure,
)
from core.ui import inject_css, render_footer, render_hero, render_metric_tile, render_sidebar  # noqa: E402

st.set_page_config(page_title="Benchmark Explorer", page_icon=":material/bar_chart:", layout="wide")
inject_css()
render_sidebar()

render_hero(
    title="Benchmark Explorer (Plotly)",
    subtitle=(
        "Interactive per-arm bootstrap CIs, ROUGE vs. numeric-faithfulness rank disagreement, "
        "and raw downloadable evaluation metrics."
    ),
    kicker="Phase 2 & Phase 6 · Interactive Plotly Explorer",
)

bundle = get_eval_bundle()
if bundle.is_empty or not bundle.arms:
    show_empty_state(bundle, context="benchmark arm results")
    st.stop()

first_arm = bundle.arm_results[bundle.arms[0]]
n_records = first_arm.n_records
n_resamples = first_arm.aggregate.rouge1.n_resamples

k1, k2, k3 = st.columns(3)
with k1:
    render_metric_tile("Arms Loaded", str(len(bundle.arms)))
with k2:
    render_metric_tile("Held-out Records", str(n_records), f"seed={first_arm.sample_seed}")
with k3:
    render_metric_tile("Bootstrap Resamples", str(n_resamples), "95% percentile CI")

st.markdown("#### Per-Arm Metric with Bootstrap CI Error Bars")
metric_key = st.selectbox(
    "Choose metric",
    options=list(METRIC_OPTIONS.keys()),
    format_func=lambda k: METRIC_OPTIONS[k],
    key="be_metric_select",
)
fig_ci = build_arm_bar_ci_figure(bundle, metric_key)
st.plotly_chart(fig_ci, width="stretch", alt=f"Plotly bar chart of {METRIC_OPTIONS[metric_key]} with 95% bootstrap CIs")

st.markdown("#### ROUGE vs. Numeric Faithfulness — Rank Disagreement")
fig_rank, rank_df = build_rouge_vs_faithfulness_figure(bundle)
st.plotly_chart(fig_rank, width="stretch", alt="Side-by-side arm rankings on ROUGE-1 vs numeric grounding recall")
st.dataframe(rank_df, hide_index=True, alt="Arm rank comparison between ROUGE-1 and numeric grounding recall")

st.markdown("#### Raw Numbers & CSV Export")
raw_df = build_raw_metrics_table(bundle)
st.dataframe(raw_df, hide_index=True, alt="Raw numeric evaluation metrics and bootstrap CIs per arm")
st.download_button(
    "Download Raw Metrics (CSV)",
    data=raw_df.to_csv(index=False).encode("utf-8"),
    file_name="benchmark_raw_metrics.csv",
    mime="text/csv",
)

render_footer()
