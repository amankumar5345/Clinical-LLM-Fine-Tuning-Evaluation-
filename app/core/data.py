"""Validated eval-artifact loader with st.cache_data and empty-state handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import BaseModel, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = REPO_ROOT / "artifacts" / "eval"
EXPECTED_ARMS = ("zero-shot", "zero-shot-14b", "finetuned", "classic-tfidf")


class BootstrapMetric(BaseModel):
    point_estimate: float
    ci_low: float
    ci_high: float
    n: int
    n_resamples: int


class PairedDelta(BaseModel):
    delta: float
    ci_low: float
    ci_high: float
    n: int
    n_resamples: int
    excludes_zero: bool


class ArmAggregate(BaseModel):
    rouge1: BootstrapMetric
    rouge2: BootstrapMetric
    rougeL: BootstrapMetric
    bertscore_f1: BootstrapMetric
    numeric_grounding_recall: BootstrapMetric
    numeric_precision: BootstrapMetric
    fabricated_number_rate: BootstrapMetric
    turn_format_valid_rate: float
    tokens_per_second: float | None = None
    records_per_second: float
    peak_vram_gb: float | None = None
    wall_clock_s: float


class EvalRecord(BaseModel):
    note_id: str
    generated: str
    reference: str
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_f1: float
    starts_with_turn_marker: bool
    num_turns: int
    has_doctor_turn: bool
    has_patient_turn: bool
    numeric_grounding_recall: float
    numeric_precision: float
    fabricated_number_rate: float
    n_note_numbers: int
    n_generated_numbers: int
    latency_s: float
    gen_tokens: int | None = None
    retrieved_note_id: str | None = None
    retrieval_similarity: float | None = None


class ArmResult(BaseModel):
    arm: str
    model_name: str | None = None
    adapter_path: str | None = None
    baseline: str | None = None
    n_records: int
    decoding: dict[str, Any] = Field(default_factory=dict)
    sample_seed: int
    aggregate: ArmAggregate
    records: list[EvalRecord]


class ComparisonResult(BaseModel):
    n_records: int
    arms: dict[str, ArmAggregate]
    comparisons: dict[str, dict[str, PairedDelta]]


class ExampleNote(BaseModel):
    note_id: str
    reference: str
    generations: dict[str, EvalRecord]


class EvalBundle(BaseModel):
    arms: list[str] = Field(default_factory=list)
    metrics_per_arm: dict[str, ArmAggregate] = Field(default_factory=dict)
    arm_results: dict[str, ArmResult] = Field(default_factory=dict)
    comparison: ComparisonResult | None = None
    example_notes: dict[str, ExampleNote] = Field(default_factory=dict)
    missing_files: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.arms and self.comparison is None


def _resolve_eval_dir(eval_dir: str | Path | None) -> Path:
    return Path(eval_dir) if eval_dir is not None else DEFAULT_EVAL_DIR


@st.cache_data(show_spinner=False)
def load_eval_bundle(eval_dir: str | None = None) -> dict[str, Any]:
    """Load and pydantic-validate saved eval results from disk (cached)."""
    root = _resolve_eval_dir(eval_dir)
    missing: list[str] = []
    errors: list[str] = []
    arm_results: dict[str, ArmResult] = {}

    if not root.exists():
        missing.append(str(root))
        return EvalBundle(missing_files=missing).model_dump()

    discovered_dirs = [p.parent.name for p in sorted(root.glob("*/results.json"))]
    ordered_arms = [a for a in EXPECTED_ARMS if a in discovered_dirs] + [
        a for a in discovered_dirs if a not in EXPECTED_ARMS
    ]
    for expected in EXPECTED_ARMS:
        if expected not in ordered_arms:
            missing.append(str(root / expected / "results.json"))

    for arm in ordered_arms:
        path = root / arm / "results.json"
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            parsed = ArmResult.model_validate(raw)
            arm_results[parsed.arm] = parsed
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"{path}: {exc}")

    comparison: ComparisonResult | None = None
    comp_path = root / "comparison.json"
    if not comp_path.exists():
        missing.append(str(comp_path))
    else:
        try:
            raw_comp = json.loads(comp_path.read_text(encoding="utf-8"))
            comparison = ComparisonResult.model_validate(raw_comp)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"{comp_path}: {exc}")

    example_notes: dict[str, ExampleNote] = {}
    for arm_name, result in arm_results.items():
        for rec in result.records:
            if rec.note_id not in example_notes:
                example_notes[rec.note_id] = ExampleNote(
                    note_id=rec.note_id,
                    reference=rec.reference,
                    generations={},
                )
            example_notes[rec.note_id].generations[arm_name] = rec

    bundle = EvalBundle(
        arms=list(arm_results.keys()),
        metrics_per_arm={name: res.aggregate for name, res in arm_results.items()},
        arm_results=arm_results,
        comparison=comparison,
        example_notes=example_notes,
        missing_files=missing,
        validation_errors=errors,
    )
    return bundle.model_dump()


def get_eval_bundle(eval_dir: str | Path | None = None) -> EvalBundle:
    """Return a typed EvalBundle from the cached loader."""
    raw = load_eval_bundle(str(eval_dir) if eval_dir is not None else None)
    return EvalBundle.model_validate(raw)


def show_empty_state(bundle: EvalBundle, context: str = "evaluation results") -> None:
    """Render a clear non-error empty state when result files are missing."""
    missing_list = "\n".join(f"- `{p}`" for p in bundle.missing_files) if bundle.missing_files else "- None listed"
    st.info(
        f"No {context} are available yet (N/A). Run `bash scripts/run_all_arms.sh` to generate them.\n\n"
        f"Missing paths:\n{missing_list}",
        title="No saved results found",
    )
