"""Page 2: Faithfulness Checker — interactive numeric grounding & highlight view."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.data_loader import fmt_num  # noqa: E402
from core.live_eval import SYNTHETIC_EXAMPLES, get_unsupported_rows, highlight_dialogue_html  # noqa: E402
from core.ui import inject_css, render_footer, render_hero, render_metric_tile, render_sidebar  # noqa: E402
from src.eval.faithfulness import extract_numbers, numeric_faithfulness  # noqa: E402

st.set_page_config(page_title="Faithfulness Checker", page_icon=":material/verified:", layout="wide")
inject_css()
render_sidebar()

render_hero(
    title="Numeric Faithfulness Checker",
    subtitle=(
        "Check every number in a generated dialogue against its source clinical note using "
        "`src.eval.faithfulness.numeric_faithfulness`."
    ),
    kicker="Interactive Faithfulness Audit · Ephemeral In-Memory Check",
)

st.warning(
    "Use only synthetic or de-identified text. Do NOT paste real patient data or PHI. "
    "Inputs are processed in memory only and are never logged or stored.",
    title="Synthetic or de-identified text only",
)

example_names = list(SYNTHETIC_EXAMPLES.keys())
if "fc_note_input" not in st.session_state:
    first = SYNTHETIC_EXAMPLES[example_names[0]]
    st.session_state["fc_note_input"] = first["note"]
    st.session_state["fc_dialogue_input"] = first["dialogue"]


def _apply_example() -> None:
    chosen = st.session_state["fc_example_select"]
    st.session_state["fc_note_input"] = SYNTHETIC_EXAMPLES[chosen]["note"]
    st.session_state["fc_dialogue_input"] = SYNTHETIC_EXAMPLES[chosen]["dialogue"]


st.selectbox(
    "Load synthetic example",
    options=example_names,
    key="fc_example_select",
    on_change=_apply_example,
)

c_in1, c_in2 = st.columns(2)
with c_in1:
    source_note = st.text_area("Source Clinical Note (synthetic / de-identified)", key="fc_note_input", height=175)
with c_in2:
    dialogue = st.text_area("Dialogue to Check", key="fc_dialogue_input", height=175)

faith = numeric_faithfulness(dialogue, source_note)
supported_count = len(extract_numbers(dialogue) & extract_numbers(source_note))
total_count = int(faith["n_generated_numbers"])

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_tile("Supported / Total Numbers", f"{supported_count} / {total_count}")
with m2:
    render_metric_tile("Numeric Precision", fmt_num(faith["numeric_precision"]))
with m3:
    render_metric_tile("Grounding Recall", fmt_num(faith["numeric_grounding_recall"]))
with m4:
    render_metric_tile("Fabricated Rate", fmt_num(faith["fabricated_number_rate"]))

st.markdown("#### Highlighted Dialogue (hover numbers for details)")
st.caption("Green = supported by source note · Red = unsupported / not found in source note")
st.html(highlight_dialogue_html(dialogue, source_note))

st.markdown("#### Unsupported Values")
unsupported_rows = get_unsupported_rows(dialogue, source_note)
if unsupported_rows:
    st.dataframe(
        pd.DataFrame(unsupported_rows),
        hide_index=True,
        alt="Unsupported numeric values found in dialogue",
    )
else:
    st.success("All numbers in the dialogue are supported by the source note.", title="0 unsupported numbers")

render_footer()
