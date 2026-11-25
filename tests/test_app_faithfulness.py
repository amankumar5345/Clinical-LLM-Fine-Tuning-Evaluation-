"""Tests for the Faithfulness Checker highlight and unsupported-value helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.live_eval import (  # noqa: E402
    SYNTHETIC_EXAMPLES,
    get_unsupported_rows,
    highlight_dialogue_html,
)


def test_highlight_marks_supported_green_and_unsupported_red():
    note = "BP 120/80 mmHg, sodium 135 mmol/L."
    dialogue = "Doctor: BP is 120/80 and sodium is 140."
    rendered = highlight_dialogue_html(dialogue, note)

    assert '<span class="num-supported" title="Supported: &#x27;120&#x27; found in source note">120</span>' in rendered
    assert '<span class="num-supported" title="Supported: &#x27;80&#x27; found in source note">80</span>' in rendered
    assert '<span class="num-unsupported" title="Unsupported: &#x27;140&#x27; not found in source note">140</span>' in rendered


def test_highlight_normalizes_decimals_and_leading_zeros():
    note = "Duration: 5 days, temp 37.0 C."
    dialogue = "Patient: I had symptoms for 05 days and temp 37 C."
    rendered = highlight_dialogue_html(dialogue, note)

    assert "num-unsupported" not in rendered
    assert rendered.count("num-supported") == 2


def test_highlight_escapes_html_injection():
    note = "Age 40."
    dialogue = "Doctor: <script>alert(1)</script> Age 40."
    rendered = highlight_dialogue_html(dialogue, note)

    assert "<script>" not in rendered
    assert "&lt;script&gt;alert(" in rendered
    assert '<span class="num-unsupported"' in rendered


def test_get_unsupported_rows_extracts_exact_mismatches():
    case2 = SYNTHETIC_EXAMPLES["Synthetic Case 2 · Fabricated Dosage & Lab"]
    rows = get_unsupported_rows(case2["dialogue"], case2["note"])
    raw_vals = [r["Raw Text"] for r in rows]

    assert raw_vals == ["20", "30", "11.5"]
    assert all(r["Status"] == "Not found in source note" for r in rows)
