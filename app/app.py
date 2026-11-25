"""Landing page for the NoteChat QLoRA evaluation app."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.data import get_eval_bundle, show_empty_state  # noqa: E402
from core.data_loader import fmt_ci, fmt_num, load_contamination  # noqa: E402
from core.plotly_charts import build_rouge_misleads_figure  # noqa: E402
from core.ui import inject_css, render_card, render_footer, render_hero, render_metric_tile, render_sidebar  # noqa: E402

st.set_page_config(
    page_title="NoteChat QLoRA Eval",
    page_icon=":material/clinical_notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar()

render_hero(
    title="Clinical Note -> Doctor-Patient Dialogue Evaluation",
    subtitle=(
        "Can a 4-bit QLoRA fine-tuned 3B model match or beat a 14B zero-shot model and a "
        "non-neural TF-IDF retrieval baseline under strict local-only evaluation?"
    ),
    kicker="NoteChat QLoRA Eval · Local Benchmark Harness",
)

with st.skeleton():
    bundle = get_eval_bundle()
    contam = load_contamination()

if bundle.is_empty:
    show_empty_state(bundle)
else:
    ft = bundle.metrics_per_arm.get("finetuned")
    delta_ft_14b = (
        bundle.comparison.comparisons.get("finetuned vs zero-shot-14b", {})
        if bundle.comparison
        else {}
    )
    contam_delta = (contam or {}).get("delta_true_minus_control", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_tile(
            "Fine-tuned 3B ROUGE-1",
            fmt_num(ft.rouge1.point_estimate) if ft else "N/A",
            f"95% CI {fmt_ci(ft.rouge1.model_dump())}" if ft else "N/A",
        )
    with c2:
        render_metric_tile(
            "Fine-tuned 3B Numeric Recall",
            fmt_num(ft.numeric_grounding_recall.point_estimate) if ft else "N/A",
            f"95% CI {fmt_ci(ft.numeric_grounding_recall.model_dump())}" if ft else "N/A",
        )
    with c3:
        r1_d = delta_ft_14b.get("rouge1")
        render_metric_tile(
            "Δ ROUGE-1 (3B FT vs 14B ZS)",
            f"{r1_d.delta:+.3f}" if r1_d else "N/A",
            f"95% CI {fmt_ci(r1_d.model_dump(), signed=True)}" if r1_d else "N/A",
        )
    with c4:
        render_metric_tile(
            "Contamination Δ ROUGE-L",
            f"{contam_delta['delta']:+.4f}" if "delta" in contam_delta else "N/A",
            f"n={contam.get('n_records', 'N/A')} · {fmt_ci(contam_delta, 4, signed=True)}" if contam else "N/A",
        )

    st.markdown("#### :material/warning: Why ROUGE Misleads on Clinical Dialogue")
    e_left, e_right = st.columns([1, 1.35])
    with e_left:
        tfidf = bundle.metrics_per_arm.get("classic-tfidf")
        zs14 = bundle.metrics_per_arm.get("zero-shot-14b")
        render_card(
            "Fluent Style != Factual Grounding",
            f"A non-neural TF-IDF retrieval baseline scores ROUGE-1 {fmt_num(tfidf.rouge1.point_estimate) if tfidf else 'N/A'} "
            f"(ahead of zero-shot 14B at {fmt_num(zs14.rouge1.point_estimate) if zs14 else 'N/A'}), yet fabricates "
            f"{fmt_num(tfidf.fabricated_number_rate.point_estimate * 100, 1, '%') if tfidf else 'N/A'} of the numbers it states "
            "because it retrieves a fluent dialogue from a different patient. Numeric faithfulness against the source note "
            "separates true grounding from style mimicry.",
            badges=[("No-Model Retrieval Trap", "amber"), ("Numeric Faithfulness Proxy", "teal")],
        )
    with e_right:
        st.plotly_chart(
            build_rouge_misleads_figure(bundle),
            width="stretch",
            alt="Grouped bar chart comparing ROUGE-1 and fabricated number rate across arms",
        )

st.markdown("#### :material/route: How It Works")
s1, s2, s3 = st.columns(3)
with s1:
    render_card(
        "1 · Local Data & Split",
        "NoteChat clinical notes are deduplicated by SHA-256, stripped of prompt preambles, and split by note_id.",
        badges=[("src/data/build_dataset.py", "sky")],
    )
with s2:
    render_card(
        "2 · 4-Bit QLoRA Fine-Tuning",
        "Qwen2.5-3B-Instruct is adapted with NF4 4-bit quantization and LoRA on attention + MLP projections.",
        badges=[("notebooks/finetune.ipynb", "teal")],
    )
with s3:
    render_card(
        "3 · Paired Eval & Faithfulness",
        "All 4 arms decode deterministically on identical held-out records and are scored with 95% bootstrap CIs.",
        badges=[("src/eval/run_eval.py", "teal")],
    )

render_footer()
