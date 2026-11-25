"""Altair chart builders styled for the dark clinical theme."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

from core.data_loader import ARM_LABELS, ARM_ORDER

ARM_COLORS = {
    "finetuned": "#14B8A6",
    "zero-shot-14b": "#38BDF8",
    "zero-shot": "#94A3B8",
    "classic-tfidf": "#F59E0B",
}


def _arm_label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


def build_arm_ci_chart(comparison: dict[str, Any], metric_key: str, metric_label: str) -> alt.Chart:
    """Horizontal point-estimate + 95% CI error-bar chart across arms."""
    arms_data = comparison.get("arms", {})
    rows = []
    for arm in ARM_ORDER:
        m = arms_data.get(arm, {}).get(metric_key)
        if isinstance(m, dict) and m.get("point_estimate") is not None:
            rows.append(
                {
                    "arm": arm,
                    "label": _arm_label(arm),
                    "point_estimate": m["point_estimate"],
                    "ci_low": m["ci_low"],
                    "ci_high": m["ci_high"],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    domain = [_arm_label(a) for a in ARM_ORDER if a in df["arm"].values]
    color_scale = alt.Scale(domain=domain, range=[ARM_COLORS.get(a, "#14B8A6") for a in ARM_ORDER if a in df["arm"].values])

    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=domain, title=None),
        color=alt.Color("label:N", scale=color_scale, legend=None),
    )
    bars = base.mark_errorbar(thickness=2.5).encode(
        x=alt.X("ci_low:Q", title=metric_label, scale=alt.Scale(zero=False)),
        x2="ci_high:Q",
    )
    pts = base.mark_point(filled=True, size=95).encode(
        x=alt.X("point_estimate:Q", title=metric_label),
        tooltip=[
            alt.Tooltip("label:N", title="Arm"),
            alt.Tooltip("point_estimate:Q", title="Point Est.", format=".3f"),
            alt.Tooltip("ci_low:Q", title="95% CI Low", format=".3f"),
            alt.Tooltip("ci_high:Q", title="95% CI High", format=".3f"),
        ],
    )
    return (bars + pts).properties(height=210, title=f"{metric_label} (95% bootstrap CI)")


def build_paired_delta_chart(comparison: dict[str, Any], pair_key: str) -> alt.Chart:
    """Plot paired bootstrap deltas with 95% CIs for a selected arm pair."""
    deltas = comparison.get("comparisons", {}).get(pair_key, {})
    rows = []
    for metric, d in deltas.items():
        rows.append(
            {
                "metric": metric,
                "delta": d["delta"],
                "ci_low": d["ci_low"],
                "ci_high": d["ci_high"],
                "excludes_zero": "Excludes 0" if d.get("excludes_zero") else "Includes 0",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    zero_rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="#64748B", strokeDash=[4, 4]).encode(x="x:Q")
    color_scale = alt.Scale(domain=["Excludes 0", "Includes 0"], range=["#14B8A6", "#F59E0B"])

    base = alt.Chart(df).encode(
        y=alt.Y("metric:N", title=None, sort=None),
        color=alt.Color("excludes_zero:N", scale=color_scale, title="95% CI"),
    )
    err = base.mark_errorbar(thickness=2.5).encode(
        x=alt.X("ci_low:Q", title="Paired difference (Δ)"),
        x2="ci_high:Q",
    )
    pts = base.mark_point(filled=True, size=90).encode(
        x="delta:Q",
        tooltip=[
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("delta:Q", title="Δ", format="+.4f"),
            alt.Tooltip("ci_low:Q", title="CI Low", format="+.4f"),
            alt.Tooltip("ci_high:Q", title="CI High", format="+.4f"),
            alt.Tooltip("excludes_zero:N", title="Significance"),
        ],
    )
    return (zero_rule + err + pts).properties(height=240, title=f"Paired deltas: {pair_key}")


def build_faithfulness_distribution(arms: dict[str, dict[str, Any]], metric_key: str, label: str) -> alt.Chart:
    """Histogram/layered density of per-record faithfulness across arms."""
    rows = []
    for arm, data in arms.items():
        for rec in data.get("records", []):
            if metric_key in rec and rec[metric_key] is not None:
                rows.append({"arm": _arm_label(arm), "arm_key": arm, "value": float(rec[metric_key])})
    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_bar()

    domain = [_arm_label(a) for a in ARM_ORDER if a in arms]
    color_scale = alt.Scale(domain=domain, range=[ARM_COLORS.get(a, "#14B8A6") for a in ARM_ORDER if a in arms])

    return (
        alt.Chart(df)
        .mark_bar(opacity=0.72)
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=20), title=label),
            y=alt.Y("count():Q", title="Records"),
            color=alt.Color("arm:N", scale=color_scale, legend=None),
            row=alt.Row("arm:N", sort=domain, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            tooltip=[alt.Tooltip("arm:N", title="Arm"), alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=68)
    )


def build_contamination_chart(contam: dict[str, Any]) -> alt.Chart:
    """Compare true continuation vs. deranged control ROUGE-L with 95% CIs."""
    rows = []
    for key, label, color in [
        ("rougeL_vs_true", "True continuation", "#14B8A6"),
        ("rougeL_vs_control", "Deranged control", "#94A3B8"),
    ]:
        m = contam.get(key)
        if isinstance(m, dict):
            rows.append(
                {
                    "comparison": label,
                    "point_estimate": m["point_estimate"],
                    "ci_low": m["ci_low"],
                    "ci_high": m["ci_high"],
                    "color": color,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    scale = alt.Scale(domain=["True continuation", "Deranged control"], range=["#14B8A6", "#94A3B8"])
    base = alt.Chart(df).encode(
        y=alt.Y("comparison:N", title=None, sort=["True continuation", "Deranged control"]),
        color=alt.Color("comparison:N", scale=scale, legend=None),
    )
    err = base.mark_errorbar(thickness=2.5).encode(
        x=alt.X("ci_low:Q", title="ROUGE-L", scale=alt.Scale(zero=False)),
        x2="ci_high:Q",
    )
    pts = base.mark_point(filled=True, size=100).encode(
        x="point_estimate:Q",
        tooltip=[
            alt.Tooltip("comparison:N", title="Comparison"),
            alt.Tooltip("point_estimate:Q", title="ROUGE-L", format=".4f"),
            alt.Tooltip("ci_low:Q", title="95% CI Low", format=".4f"),
            alt.Tooltip("ci_high:Q", title="95% CI High", format=".4f"),
        ],
    )
    return (err + pts).properties(height=150, title="Base model prefix-continuation ROUGE-L (95% CI)")
