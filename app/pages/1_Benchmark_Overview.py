"""Page 1: Benchmark Overview — per-arm CIs, paired deltas, and serving cost."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.charts import build_arm_ci_chart, build_paired_delta_chart  # noqa: E402
from core.data_loader import (  # noqa: E402
    ARM_LABELS,
    ARM_ORDER,
    fmt_ci,
    fmt_num,
    fmt_pct,
    load_all_arms,
    load_comparison,
)
from core.ui import inject_css, render_hero, render_sidebar  # noqa: E402

st.set_page_config(page_title="Benchmark Overview", page_icon=":material/analytics:", layout="wide")
inject_css()
render_sidebar()

render_hero(
    title="Benchmark Overview",
    subtitle="Per-arm point estimates with 95% percentile bootstrap CIs, paired deltas, and GPU serving cost.",
    kicker="Phase 2 & Phase 6 · Aggregate Results",
)

comparison = load_comparison()
arms = load_all_arms()

if not comparison:
    st.warning("`artifacts/eval/comparison.json` not found. All metrics are unavailable (N/A).")
    st.stop()

METRICS = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore-F1",
    "numeric_grounding_recall": "Numeric Grounding Recall",
    "fabricated_number_rate": "Fabricated Number Rate",
}

selected_metric = st.segmented_control(
    "Select metric for CI comparison",
    options=list(METRICS.keys()),
    default="rouge1",
    format_func=lambda k: METRICS[k],
)
if selected_metric:
    chart = build_arm_ci_chart(comparison, selected_metric, METRICS[selected_metric])
    st.altair_chart(chart, alt=f"{METRICS[selected_metric]} across benchmark arms with 95% bootstrap CIs")

st.markdown("#### Per-Arm Summary Table (95% Bootstrap CI)")
summary_rows = []
for arm in ARM_ORDER:
    agg = comparison.get("arms", {}).get(arm)
    if not agg:
        continue
    summary_rows.append(
        {
            "Arm": ARM_LABELS.get(arm, arm),
            "ROUGE-1": fmt_ci(agg.get("rouge1")),
            "ROUGE-2": fmt_ci(agg.get("rouge2")),
            "ROUGE-L": fmt_ci(agg.get("rougeL")),
            "BERTScore-F1": fmt_ci(agg.get("bertscore_f1")),
            "Numeric Recall": fmt_ci(agg.get("numeric_grounding_recall")),
            "Fabricated Numbers": fmt_ci(agg.get("fabricated_number_rate")),
            "Turn-Format Valid": fmt_pct(agg.get("turn_format_valid_rate")),
        }
    )
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, alt="Per-arm evaluation metrics with 95% bootstrap CIs")

st.markdown("#### Paired Bootstrap Deltas")
pairs = list(comparison.get("comparisons", {}).keys())
default_pair = "finetuned vs zero-shot-14b" if "finetuned vs zero-shot-14b" in pairs else (pairs[0] if pairs else None)
if pairs and default_pair:
    pair_key = st.selectbox("Select arm comparison (A − B)", options=pairs, index=pairs.index(default_pair))
    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        delta_chart = build_paired_delta_chart(comparison, pair_key)
        st.altair_chart(delta_chart, alt=f"Paired bootstrap deltas for {pair_key}")
    with c_right:
        delta_rows = []
        for metric, d in comparison["comparisons"][pair_key].items():
            delta_rows.append(
                {
                    "Metric": metric,
                    "Δ (A − B)": fmt_num(d.get("delta"), decimals=4),
                    "95% CI": f"[{fmt_num(d.get('ci_low'), 4)}, {fmt_num(d.get('ci_high'), 4)}]",
                    "Excludes 0": "Yes" if d.get("excludes_zero") else "No",
                }
            )
        st.dataframe(pd.DataFrame(delta_rows), hide_index=True, alt=f"Paired bootstrap delta table for {pair_key}")

st.markdown("#### Serving Cost & Throughput")
cost_rows = []
for arm in ARM_ORDER:
    arm_data = arms.get(arm, {})
    agg = comparison.get("arms", {}).get(arm, {})
    cost_rows.append(
        {
            "Arm": ARM_LABELS.get(arm, arm),
            "Model": arm_data.get("model_name") or "N/A",
            "Peak VRAM (GB)": fmt_num(agg.get("peak_vram_gb"), 2),
            "Tokens / sec": fmt_num(agg.get("tokens_per_second"), 1),
            "Records / sec": fmt_num(agg.get("records_per_second"), 2),
            "Wall Clock (s)": fmt_num(agg.get("wall_clock_s"), 1),
        }
    )
st.dataframe(pd.DataFrame(cost_rows), hide_index=True, alt="Peak VRAM, throughput, and wall-clock time per arm")
