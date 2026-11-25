"""Page 5: Method — QLoRA config, hardware, splits, contamination, CI, and limitations."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.data import get_eval_bundle  # noqa: E402
from core.data_loader import (  # noqa: E402
    fmt_ci,
    fmt_num,
    load_configs,
    load_contamination,
    load_data_report_stats,
    load_method_metadata,
)
from core.ui import inject_css, render_card, render_footer, render_hero, render_metric_tile, render_sidebar  # noqa: E402

st.set_page_config(page_title="Method & Limitations", page_icon=":material/science:", layout="wide")
inject_css()
render_sidebar()

render_hero(
    title="Methodology, Setup & Limitations",
    subtitle="Reproducible QLoRA configuration, split design, contamination audit, CI status, and scope boundaries.",
    kicker="Phase 1–7 · Technical Specification",
)

with st.skeleton():
    configs = load_configs()
    data_stats = load_data_report_stats()
    contam = load_contamination()
    meta = load_method_metadata()
    bundle = get_eval_bundle()

train_cfg = configs.get("train", {})
lora_cfg = train_cfg.get("lora", {})
quant_cfg = train_cfg.get("quantization", {})
data_cfg = configs.get("data", {})

st.markdown("#### :material/tune: QLoRA Settings & Hardware")
c1, c2 = st.columns(2)
with c1:
    render_card(
        "QLoRA Hyperparameters (configs/train.yaml)",
        f"Quantization: 4-bit {quant_cfg.get('bnb_4bit_quant_type', 'N/A')} "
        f"(double quant={quant_cfg.get('bnb_4bit_use_double_quant', 'N/A')}, "
        f"compute={quant_cfg.get('bnb_4bit_compute_dtype', 'N/A')}). "
        f"LoRA r={lora_cfg.get('r', 'N/A')}, alpha={lora_cfg.get('lora_alpha', 'N/A')}, "
        f"dropout={lora_cfg.get('lora_dropout', 'N/A')}. "
        f"Epochs={train_cfg.get('num_train_epochs', 'N/A')}, "
        f"lr={train_cfg.get('learning_rate', 'N/A')}, "
        f"max_seq_len={train_cfg.get('max_seq_len', 'N/A')}, "
        f"optim={train_cfg.get('optim', 'N/A')}.",
        badges=[(f"Trainable: {meta.get('trainable_params', 'N/A')}", "teal")],
    )
with c2:
    ft_agg = bundle.metrics_per_arm.get("finetuned")
    zs14_agg = bundle.metrics_per_arm.get("zero-shot-14b")
    render_card(
        "Hardware & Serving Footprint",
        f"Training hardware: {meta.get('hardware', 'N/A')}. "
        f"Eval peak VRAM — Fine-tuned 3B: {fmt_num(ft_agg.peak_vram_gb, 2, ' GB') if ft_agg else 'N/A'} "
        f"({fmt_num(ft_agg.tokens_per_second, 1, ' tok/s') if ft_agg else 'N/A'}) vs. "
        f"Zero-shot 14B: {fmt_num(zs14_agg.peak_vram_gb, 2, ' GB') if zs14_agg else 'N/A'} "
        f"({fmt_num(zs14_agg.tokens_per_second, 1, ' tok/s') if zs14_agg else 'N/A'}).",
        badges=[("Local-Only Inference", "sky"), ("Greedy Decoding", "teal")],
    )

st.markdown("#### :material/dataset: Split Design & Contamination Probe")
s1, s2 = st.columns(2)
with s1:
    render_card(
        "Dataset Split Design (configs/data.yaml)",
        f"Source: {data_cfg.get('source_file', 'N/A')}. "
        f"Total notes: {data_stats.get('total_notes', 'N/A')} "
        f"(duplicates dropped: {data_stats.get('duplicates_dropped', 'N/A')}). "
        f"Train / Val / Test: {data_stats.get('train_split', 'N/A')} / "
        f"{data_stats.get('val_split', 'N/A')} / {data_stats.get('test_split', 'N/A')} "
        f"split by `{data_cfg.get('split', {}).get('key', 'N/A')}` (seed={data_cfg.get('split', {}).get('seed', 'N/A')}).",
        badges=[("Zero Leakage Verified", "teal")],
    )
with s2:
    cdelta = (contam or {}).get("delta_true_minus_control", {})
    render_card(
        "Phase 3 Contamination Probe",
        f"Model: {(contam or {}).get('model_name', 'N/A')} on {(contam or {}).get('n_records', 'N/A')} test dialogues. "
        f"ROUGE-L vs True: {fmt_ci((contam or {}).get('rougeL_vs_true'), 4)} · "
        f"vs Deranged Control: {fmt_ci((contam or {}).get('rougeL_vs_control'), 4)} · "
        f"Paired Δ: {fmt_ci(cdelta, 4, signed=True)}.",
        badges=[("Sattolo Derangement Control", "sky")],
    )

st.markdown("#### :material/verified_user: Test & CI Status")
t1, t2, t3 = st.columns(3)
with t1:
    render_metric_tile("Test Files", meta.get("test_files", "N/A"), f"{meta.get('test_functions', 'N/A')} test functions")
with t2:
    render_metric_tile("CI Python", meta.get("ci_python", "N/A"), meta.get("ci_workflow", "N/A"))
with t3:
    render_metric_tile("CI Checks", "ruff + pytest", "CPU-only GitHub Actions")

st.markdown("#### :material/gpp_maybe: Limitations & Scope Boundaries")
l1, l2 = st.columns(2)
with l1:
    render_card("Research Demo", "Built as a methodology evaluation harness; not a clinical product or medical device.", [("Evaluation Harness", "slate")])
    render_card("Synthetic Reference Data", "NoteChat dialogues are LLM-synthesized backwards from PMC-Patients notes, not human transcripts.", [("Synthetic Targets", "amber")])
with l2:
    render_card("Single-GPU Training", "Trained on a single consumer GPU with one default LoRA configuration; rank/epoch ablations remain future work.", [("No Ablations Yet", "amber")])
    render_card("Not for Clinical Use", "Models can fabricate clinical values; never use for patient care, charting, or medical advice.", [("No Clinical Use", "rose")])

render_footer()
