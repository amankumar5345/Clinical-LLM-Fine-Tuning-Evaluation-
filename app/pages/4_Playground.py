"""Page 4: Playground — multi-backend dialogue generation + automatic faithfulness."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.backends import get_backends  # noqa: E402
from core.data import get_eval_bundle  # noqa: E402
from core.data_loader import fmt_num  # noqa: E402
from core.live_eval import SYNTHETIC_EXAMPLES, highlight_dialogue_html, score_dialogue  # noqa: E402
from core.ui import inject_css, render_badge, render_dialogue, render_footer, render_hero, render_metric_tile, render_sidebar  # noqa: E402

st.set_page_config(page_title="Playground", page_icon=":material/play_circle:", layout="wide")
inject_css()
render_sidebar()

render_hero(
    title="Dialogue Generation Playground",
    subtitle=(
        "Generate or replay doctor-patient dialogues using Cached Examples, Local QLoRA Adapter, "
        "or a configured Remote Endpoint — with automatic numeric-faithfulness scoring."
    ),
    kicker="Inference Playground · Automatic Faithfulness Audit",
)

backends = get_backends()
statuses = {k: b.status() for k, b in backends.items()}

badges_html = "".join(
    render_badge(f"{b.label}: {'Ready' if statuses[k][0] else 'Unavailable'}", "teal" if statuses[k][0] else "slate")
    for k, b in backends.items()
)
st.html(f"<div class='clinical-card'>{badges_html}</div>")

c_sel, c_opt = st.columns([2, 1])
with c_sel:
    selected_key = st.selectbox(
        "Select inference backend",
        options=list(backends.keys()),
        format_func=lambda k: f"{backends[k].label} — {statuses[k][1]}",
        key="pg_backend_select",
    )
with c_opt:
    do_sample = st.toggle("Stochastic sampling (default: greedy)", value=False, key="pg_sample")

backend = backends[selected_key]
is_ready, status_msg = statuses[selected_key]

bundle = get_eval_bundle()
note_id: str | None = None
reference_text: str | None = None

if selected_key == "cached" and bundle.example_notes:
    note_ids = sorted(bundle.example_notes.keys())
    note_id = st.selectbox("Select held-out test note_id", options=note_ids[:50], key="pg_note_id")
    ex_note = bundle.example_notes[note_id]
    reference_text = ex_note.reference
    note_input = st.text_area(
        "Source note (unavailable in public results.json — paste synthetic note or score against reference)",
        value="",
        height=110,
        placeholder="Optional: paste a synthetic note to run numeric_faithfulness against it.",
        key="pg_cached_note",
    )
else:
    default_ex = next(iter(SYNTHETIC_EXAMPLES.values()))
    note_input = st.text_area(
        "Clinical Note (synthetic or de-identified text only — never paste PHI)",
        value=default_ex["note"],
        height=140,
        key="pg_note_input",
    )

run_disabled = not is_ready
if not is_ready:
    st.info(f"Selected backend is unavailable: {status_msg}", title="Backend disabled")

if st.button("Generate Dialogue", type="primary", disabled=run_disabled, width="stretch"):
    out_col, faith_col = st.columns([1.25, 1])
    with out_col:
        st.markdown("#### Generated Dialogue")
        stream = backend.generate(note_input, note_id=note_id, do_sample=do_sample)
        if backend.supports_streaming:
            generated_text = st.write_stream(stream)
            if not isinstance(generated_text, str):
                generated_text = str(generated_text)
        else:
            generated_text = "".join(stream)
            render_dialogue(generated_text)

    with faith_col:
        st.markdown("#### Automatic Faithfulness Check")
        audit_source = note_input.strip() or (reference_text or "")
        scores = score_dialogue(generated_text, audit_source, reference=reference_text)
        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_tile("Numeric Recall", fmt_num(scores["numeric_grounding_recall"]))
        with m2:
            render_metric_tile("Fabricated Rate", fmt_num(scores["fabricated_number_rate"]))
        with m3:
            render_metric_tile("Turns", str(scores["num_turns"]))
        if note_input.strip():
            st.html(highlight_dialogue_html(generated_text, note_input))
        elif reference_text:
            st.caption("Scored against reference dialogue (source clinical note is not stored in `results.json`).")

render_footer()
