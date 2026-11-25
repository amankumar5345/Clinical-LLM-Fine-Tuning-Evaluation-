"""Plotly chart and raw-table builders for 3_Benchmark_Explorer.py."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data import EvalBundle
from core.data_loader import ARM_LABELS, ARM_ORDER

ARM_COLORS = {
    "finetuned": "#14B8A6",
    "zero-shot-14b": "#38BDF8",
    "zero-shot": "#94A3B8",
    "classic-tfidf": "#F59E0B",
}

METRIC_OPTIONS: dict[str, str] = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore-F1",
    "numeric_grounding_recall": "Numeric Grounding Recall",
    "numeric_precision": "Numeric Precision",
    "fabricated_number_rate": "Fabricated Number Rate",
}


def build_arm_bar_ci_figure(bundle: EvalBundle, metric_key: str) -> go.Figure:
    """Plotly bar chart with asymmetric 95% bootstrap CI error bars."""
    ordered = [a for a in ARM_ORDER if a in bundle.metrics_per_arm] + [
        a for a in bundle.arms if a not in ARM_ORDER
    ]
    labels, vals, err_plus, err_minus, colors = [], [], [], [], []
    for arm in ordered:
        m = getattr(bundle.metrics_per_arm[arm], metric_key)
        labels.append(ARM_LABELS.get(arm, arm))
        vals.append(m.point_estimate)
        err_plus.append(max(0.0, m.ci_high - m.point_estimate))
        err_minus.append(max(0.0, m.point_estimate - m.ci_low))
        colors.append(ARM_COLORS.get(arm, "#14B8A6"))

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_plus,
                    arrayminus=err_minus,
                    color="#E6EDF7",
                    thickness=2,
                    width=6,
                ),
                hovertemplate="%{x}<br>Estimate: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"{METRIC_OPTIONS.get(metric_key, metric_key)} (95% Bootstrap CI)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 28, 51, 0.45)",
        font=dict(color="#E6EDF7"),
        yaxis=dict(gridcolor="#1E2D4A", zerolinecolor="#1E2D4A"),
        xaxis=dict(gridcolor="#1E2D4A"),
        margin=dict(l=40, r=20, t=48, b=40),
        height=380,
    )
    return fig


def build_rouge_vs_faithfulness_figure(bundle: EvalBundle) -> tuple[go.Figure, pd.DataFrame]:
    """Side-by-side Plotly comparison ranking arms on ROUGE-1 vs Numeric Grounding Recall."""
    rows = []
    for arm in bundle.arms:
        agg = bundle.metrics_per_arm[arm]
        rows.append(
            {
                "arm_key": arm,
                "Arm": ARM_LABELS.get(arm, arm),
                "ROUGE-1": agg.rouge1.point_estimate,
                "Numeric Recall": agg.numeric_grounding_recall.point_estimate,
                "Fabricated Rate": agg.fabricated_number_rate.point_estimate,
            }
        )
    df = pd.DataFrame(rows)
    df["ROUGE-1 Rank"] = df["ROUGE-1"].rank(ascending=False, method="min").astype(int)
    df["Faithfulness Rank"] = df["Numeric Recall"].rank(ascending=False, method="min").astype(int)
    df["Rank Shift"] = df["ROUGE-1 Rank"] - df["Faithfulness Rank"]

    by_rouge = df.sort_values("ROUGE-1", ascending=True)
    by_faith = df.sort_values("Numeric Recall", ascending=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Ranked by ROUGE-1 (style & overlap)", "Ranked by Numeric Grounding Recall"),
        horizontal_spacing=0.22,
    )
    fig.add_trace(
        go.Bar(
            y=by_rouge["Arm"],
            x=by_rouge["ROUGE-1"],
            orientation="h",
            marker_color=[ARM_COLORS.get(k, "#14B8A6") for k in by_rouge["arm_key"]],
            text=[f"#{r} ({v:.3f})" for r, v in zip(by_rouge["ROUGE-1 Rank"], by_rouge["ROUGE-1"], strict=True)],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=by_faith["Arm"],
            x=by_faith["Numeric Recall"],
            orientation="h",
            marker_color=[ARM_COLORS.get(k, "#14B8A6") for k in by_faith["arm_key"]],
            text=[
                f"#{r} ({v:.3f})"
                for r, v in zip(by_faith["Faithfulness Rank"], by_faith["Numeric Recall"], strict=True)
            ],
            textposition="auto",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 28, 51, 0.45)",
        font=dict(color="#E6EDF7"),
        margin=dict(l=20, r=20, t=48, b=30),
        height=320,
    )
    fig.update_xaxes(gridcolor="#1E2D4A", range=[0, 1.0])
    rank_df = df.sort_values("ROUGE-1 Rank")[
        ["Arm", "ROUGE-1 Rank", "ROUGE-1", "Faithfulness Rank", "Numeric Recall", "Fabricated Rate", "Rank Shift"]
    ]
    return fig, rank_df


def build_raw_metrics_table(bundle: EvalBundle) -> pd.DataFrame:
    """Flatten all arm aggregates and CIs into a raw numeric DataFrame for display & CSV export."""
    rows = []
    ordered = [a for a in ARM_ORDER if a in bundle.metrics_per_arm] + [
        a for a in bundle.arms if a not in ARM_ORDER
    ]
    for arm in ordered:
        agg = bundle.metrics_per_arm[arm]
        res = bundle.arm_results[arm]
        row: dict[str, object] = {
            "arm": arm,
            "model_name": res.model_name,
            "n_records": res.n_records,
            "turn_format_valid_rate": agg.turn_format_valid_rate,
            "peak_vram_gb": agg.peak_vram_gb,
            "tokens_per_second": agg.tokens_per_second,
            "records_per_second": agg.records_per_second,
            "wall_clock_s": agg.wall_clock_s,
        }
        for key in METRIC_OPTIONS:
            m = getattr(agg, key)
            row[f"{key}_point"] = m.point_estimate
            row[f"{key}_ci_low"] = m.ci_low
            row[f"{key}_ci_high"] = m.ci_high
            row[f"{key}_n_resamples"] = m.n_resamples
        rows.append(row)
    return pd.DataFrame(rows)


def build_rouge_misleads_figure(bundle: EvalBundle) -> go.Figure:
    """Compact grouped bar chart illustrating ROUGE-1 vs Fabricated Number Rate."""
    ordered = [a for a in ARM_ORDER if a in bundle.metrics_per_arm]
    labels = [ARM_LABELS.get(a, a) for a in ordered]
    rouge1_vals = [bundle.metrics_per_arm[a].rouge1.point_estimate for a in ordered]
    fab_vals = [bundle.metrics_per_arm[a].fabricated_number_rate.point_estimate for a in ordered]

    fig = go.Figure(
        data=[
            go.Bar(name="ROUGE-1 (higher = closer to reference)", x=labels, y=rouge1_vals, marker_color="#14B8A6"),
            go.Bar(name="Fabricated Number Rate (lower = safer)", x=labels, y=fab_vals, marker_color="#F43F5E"),
        ]
    )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 28, 51, 0.45)",
        font=dict(color="#E6EDF7"),
        yaxis=dict(gridcolor="#1E2D4A", range=[0, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=20, t=42, b=30),
        height=310,
    )
    return fig
