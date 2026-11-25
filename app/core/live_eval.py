"""Live scoring & optional local adapter inference bridge."""

from __future__ import annotations

import html
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.faithfulness import NUMBER_RE, extract_numbers, numeric_faithfulness  # noqa: E402

_TURN_START_RE = re.compile(r"^(Doctor|Patient):", re.MULTILINE)
_TURN_SPLIT_RE = re.compile(r"(?=^(?:Doctor|Patient):)", re.MULTILINE)

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
DEFAULT_ADAPTER_REL = "artifacts/adapters/unsloth__Qwen2.5-3B-Instruct-bnb-4bit/final_adapter"


def check_inference_capabilities() -> dict[str, Any]:
    """Check whether local GPU + Unsloth + saved adapter are present."""
    adapter_dir = REPO_ROOT / DEFAULT_ADAPTER_REL
    has_adapter = adapter_dir.exists()
    has_unsloth = importlib.util.find_spec("unsloth") is not None
    has_cuda = False
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
    except ImportError:
        has_cuda = False

    return {
        "adapter_path": str(adapter_dir),
        "has_adapter": has_adapter,
        "has_unsloth": has_unsloth,
        "has_cuda": has_cuda,
        "ready": has_adapter and has_unsloth and has_cuda,
    }


@st.cache_resource(show_spinner="Loading Qwen2.5-3B + QLoRA adapter...")
def get_cached_model(use_adapter: bool = True, max_seq_len: int = 2048):
    from src.inference.local_hf import load_model

    adapter = str(REPO_ROOT / DEFAULT_ADAPTER_REL) if use_adapter else None
    return load_model(
        model_name=DEFAULT_BASE_MODEL,
        max_seq_len=max_seq_len,
        load_in_4bit=True,
        seed=42,
        adapter_path=adapter,
    )


def score_dialogue(generated: str, clinical_note: str, reference: str | None = None) -> dict[str, Any]:
    """Score any dialogue against a clinical note (and optional reference)."""
    faith = numeric_faithfulness(generated, clinical_note)
    try:
        from src.eval.metrics import rouge, turn_format_validity

        fmt = turn_format_validity(generated)
        rouge_scores = rouge(generated, reference) if reference else {}
    except ImportError:
        stripped = generated.strip()
        turns = [p.strip() for p in _TURN_SPLIT_RE.split(stripped) if p.strip()] if _TURN_START_RE.search(stripped) else []
        fmt = {
            "starts_with_turn_marker": bool(_TURN_START_RE.match(stripped)),
            "num_turns": len(turns),
            "has_doctor_turn": any(t.startswith("Doctor:") for t in turns),
            "has_patient_turn": any(t.startswith("Patient:") for t in turns),
        }
        rouge_scores = {}

    note_nums = extract_numbers(clinical_note)
    gen_nums = extract_numbers(generated)
    shared = sorted(note_nums & gen_nums, key=float)
    invented = sorted(gen_nums - note_nums, key=float)
    missed = sorted(note_nums - gen_nums, key=float)
    return {
        **faith,
        **fmt,
        **rouge_scores,
        "shared_numbers": shared,
        "invented_numbers": invented,
        "missed_numbers": missed,
    }


SYNTHETIC_EXAMPLES: dict[str, dict[str, str]] = {
    "Synthetic Case 1 · Grounded Vitals & Labs": {
        "note": (
            "Synthetic 58-year-old adult presented with 3 days of fatigue. "
            "Vitals: BP 142/88 mmHg, HR 84 bpm, Temp 37.2 C. "
            "Labs: serum sodium 134 mmol/L, potassium 4.1 mmol/L."
        ),
        "dialogue": (
            "Doctor: Hello, I see you have had fatigue for 3 days.\n"
            "Patient: Yes, that's right.\n"
            "Doctor: Your blood pressure is 142/88 mmHg, heart rate is 84 bpm, and temperature is 37.2 C.\n"
            "Patient: How were my lab tests?\n"
            "Doctor: Your sodium is 134 mmol/L and potassium is 4.1 mmol/L."
        ),
    },
    "Synthetic Case 2 · Fabricated Dosage & Lab": {
        "note": (
            "Synthetic 64-year-old adult with hypertension. "
            "BP 158/84 mmHg, HR 88 bpm. Prescribed lisinopril 10 mg daily for 14 days."
        ),
        "dialogue": (
            "Doctor: Your blood pressure today is 158/84 mmHg and your pulse is 88 bpm.\n"
            "Patient: Do I need medication?\n"
            "Doctor: Yes, take lisinopril 20 mg daily for 30 days, and your hemoglobin is 11.5 g/dL.\n"
            "Patient: Understood, thank you."
        ),
    },
    "Synthetic Case 3 · Number-Free Dialogue": {
        "note": (
            "Synthetic 45-year-old adult with 5 days of cough. "
            "Temp 38.1 C, SpO2 97% on room air."
        ),
        "dialogue": (
            "Doctor: Good morning, what brings you in today?\n"
            "Patient: I have had a dry cough and mild fever.\n"
            "Doctor: Let's listen to your lungs and keep you hydrated."
        ),
    },
}


def _normalize_single_number(raw: str) -> str:
    value = float(raw)
    return str(int(value)) if value.is_integer() else str(value)


def highlight_dialogue_html(dialogue: str, source_note: str) -> str:
    """Render dialogue HTML with green/red number highlights and hover tooltips."""
    note_nums = extract_numbers(source_note)
    lines_html: list[str] = []
    for raw_line in dialogue.splitlines():
        parts: list[str] = []
        last = 0
        for m in NUMBER_RE.finditer(raw_line):
            parts.append(html.escape(raw_line[last : m.start()]))
            raw_num = m.group(0)
            norm = _normalize_single_number(raw_num)
            if norm in note_nums:
                tip = html.escape(f"Supported: '{norm}' found in source note")
                parts.append(f'<span class="num-supported" title="{tip}">{html.escape(raw_num)}</span>')
            else:
                tip = html.escape(f"Unsupported: '{norm}' not found in source note")
                parts.append(f'<span class="num-unsupported" title="{tip}">{html.escape(raw_num)}</span>')
            last = m.end()
        parts.append(html.escape(raw_line[last:]))
        lines_html.append(f"<div style='margin-bottom:0.35rem'>{''.join(parts)}</div>")
    return f'<div class="dialogue-box">{"".join(lines_html)}</div>'


def get_unsupported_rows(dialogue: str, source_note: str) -> list[dict[str, str]]:
    """Return table rows for numbers present in dialogue but absent from source_note."""
    note_nums = extract_numbers(source_note)
    rows: list[dict[str, str]] = []
    for m in NUMBER_RE.finditer(dialogue):
        raw_num = m.group(0)
        norm = _normalize_single_number(raw_num)
        if norm not in note_nums:
            rows.append(
                {
                    "Raw Text": raw_num,
                    "Normalized Value": norm,
                    "Status": "Not found in source note",
                }
            )
    return rows
