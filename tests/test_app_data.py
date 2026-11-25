"""Offline tests for app/core/data.py pydantic validation and empty states."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.data import get_eval_bundle  # noqa: E402

FIXTURE_EVAL_DIR = Path(__file__).resolve().parent / "fixtures" / "eval"


def test_loads_offline_fixture_with_pydantic_validation():
    bundle = get_eval_bundle(FIXTURE_EVAL_DIR)
    assert not bundle.is_empty
    assert bundle.validation_errors == []
    assert set(bundle.arms) == {"finetuned", "zero-shot"}

    ft_metrics = bundle.metrics_per_arm["finetuned"]
    assert ft_metrics.rouge1.point_estimate == 0.65
    assert ft_metrics.rouge1.ci_low == 0.60
    assert ft_metrics.rouge1.ci_high == 0.70

    assert "note-fixture-001" in bundle.example_notes
    ex = bundle.example_notes["note-fixture-001"]
    assert "finetuned" in ex.generations
    assert "zero-shot" in ex.generations
    assert ex.generations["finetuned"].generated.startswith("Doctor:")

    assert bundle.comparison is not None
    assert "finetuned vs zero-shot" in bundle.comparison.comparisons


def test_missing_directory_returns_empty_state_without_error(tmp_path: Path):
    missing_dir = tmp_path / "does_not_exist"
    bundle = get_eval_bundle(missing_dir)
    assert bundle.is_empty
    assert bundle.arms == []
    assert bundle.metrics_per_arm == {}
    assert bundle.example_notes == {}
    assert str(missing_dir) in bundle.missing_files
    assert bundle.validation_errors == []


def test_invalid_json_records_validation_error_without_crashing(tmp_path: Path):
    bad_arm = tmp_path / "finetuned"
    bad_arm.mkdir(parents=True)
    (bad_arm / "results.json").write_text("{not valid json", encoding="utf-8")

    bundle = get_eval_bundle(tmp_path)
    assert bundle.is_empty
    assert len(bundle.validation_errors) == 1
    assert "results.json" in bundle.validation_errors[0]


def test_real_repo_eval_artifacts_validate_against_schema():
    real_dir = REPO_ROOT / "artifacts" / "eval"
    if not real_dir.exists():
        return
    bundle = get_eval_bundle(real_dir)
    assert bundle.validation_errors == []
    assert set(bundle.arms) == {"zero-shot", "zero-shot-14b", "finetuned", "classic-tfidf"}
    assert len(bundle.example_notes) == 200
    assert bundle.comparison is not None
