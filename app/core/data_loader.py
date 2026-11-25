"""Data loading helpers — every metric is read from repo artifacts or shown as N/A."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "artifacts" / "eval"
CONFIGS_DIR = REPO_ROOT / "configs"
DOCS_DIR = REPO_ROOT / "docs"

ARM_ORDER = ["finetuned", "zero-shot-14b", "zero-shot", "classic-tfidf"]
ARM_LABELS = {
    "zero-shot": "Arm 1 · Zero-shot 3B",
    "zero-shot-14b": "Arm 2 · Zero-shot 14B",
    "finetuned": "Arm 3 · QLoRA 3B",
    "classic-tfidf": "Arm 4 · TF-IDF Retrieval",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def load_comparison() -> dict[str, Any] | None:
    return _read_json(EVAL_DIR / "comparison.json")


@st.cache_data(show_spinner=False)
def load_arm_results(arm: str) -> dict[str, Any] | None:
    return _read_json(EVAL_DIR / arm / "results.json")


@st.cache_data(show_spinner=False)
def load_all_arms() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    if not EVAL_DIR.exists():
        return loaded
    for arm in ARM_ORDER:
        data = load_arm_results(arm)
        if data is not None:
            loaded[arm] = data
    for path in sorted(EVAL_DIR.glob("*/results.json")):
        arm_name = path.parent.name
        if arm_name not in loaded:
            data = _read_json(path)
            if data is not None:
                loaded[arm_name] = data
    return loaded


@st.cache_data(show_spinner=False)
def load_contamination() -> dict[str, Any] | None:
    return _read_json(EVAL_DIR / "contamination.json")


@st.cache_data(show_spinner=False)
def load_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for name in ("data", "eval", "train"):
        path = CONFIGS_DIR / f"{name}.yaml"
        if path.exists():
            configs[name] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return configs


@st.cache_data(show_spinner=False)
def load_data_report_stats() -> dict[str, str]:
    """Parse dataset summary values from docs/data_report.md without hardcoding."""
    path = DOCS_DIR / "data_report.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    stats: dict[str, str] = {}
    patterns = {
        "total_notes": r"Notes in the final dataset:\s*(\d+)",
        "duplicates_dropped": r"Exact-duplicate notes dropped.*?:\s*(\d+)",
        "train_split": r"\|\s*train\s*\|\s*(\d+)\s*\|",
        "val_split": r"\|\s*val\s*\|\s*(\d+)\s*\|",
        "test_split": r"\|\s*test\s*\|\s*(\d+)\s*\|",
        "preamble_stripped": r"stripped at build time\):\s*([^\n]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            stats[key] = m.group(1).strip()
    return stats


@st.cache_data(show_spinner=False)
def load_method_metadata() -> dict[str, str]:
    """Read hardware, CI workflow, and test-suite counts from repo files."""
    meta: dict[str, str] = {}
    card_path = DOCS_DIR / "MODEL_CARD.md"
    if card_path.exists():
        card_text = card_path.read_text(encoding="utf-8")
        hw = re.search(r"\*\*Hardware\*\*\s*\|\s*([^\n|]+)", card_text)
        tp = re.search(r"\*\*Trainable parameters\*\*\s*\|\s*([^\n|]+)", card_text)
        if hw:
            meta["hardware"] = hw.group(1).strip()
        if tp:
            meta["trainable_params"] = tp.group(1).strip()

    ci_path = REPO_ROOT / ".github" / "workflows" / "tests.yml"
    if ci_path.exists():
        ci_text = ci_path.read_text(encoding="utf-8")
        py_m = re.search(r"uv python install\s+([0-9.]+)", ci_text)
        if py_m:
            meta["ci_python"] = py_m.group(1).strip()
        meta["ci_workflow"] = ".github/workflows/tests.yml"

    tests_dir = REPO_ROOT / "tests"
    if tests_dir.exists():
        test_files = list(tests_dir.glob("test_*.py"))
        test_fn_count = sum(
            len(re.findall(r"^def test_", f.read_text(encoding="utf-8", errors="ignore"), re.MULTILINE))
            for f in test_files
        )
        meta["test_files"] = str(len(test_files))
        meta["test_functions"] = str(test_fn_count)
    return meta


def fmt_num(value: Any, decimals: int = 3, suffix: str = "") -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_pct(value: Any, decimals: int = 1) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_ci(metric_dict: dict[str, Any] | None, decimals: int = 3, signed: bool = False) -> str:
    if not isinstance(metric_dict, dict):
        return "N/A"
    val = metric_dict.get("point_estimate", metric_dict.get("delta"))
    lo = metric_dict.get("ci_low")
    hi = metric_dict.get("ci_high")
    if val is None or lo is None or hi is None:
        return "N/A"
    sign = "+" if signed else ""
    return f"{val:{sign}.{decimals}f} [{lo:{sign}.{decimals}f}, {hi:{sign}.{decimals}f}]"
