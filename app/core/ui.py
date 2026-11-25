"""Shared UI components, sidebar, and single inject_css() helper."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

GITHUB_URL = "https://github.com/amankumar5345/Clinical-LLM-Fine-Tuning-Evaluation-"
DOCS_URLS = {
    "Project Spec": f"{GITHUB_URL}/blob/master/docs/PROJECT_SPEC.md",
    "Model Card": f"{GITHUB_URL}/blob/master/docs/MODEL_CARD.md",
    "Decisions Log": f"{GITHUB_URL}/blob/master/docs/DECISIONS.md",
    "Architecture": f"{GITHUB_URL}/blob/master/docs/ARCHITECTURE.md",
}
LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "logo.svg"


def inject_css() -> None:
    """Inject the dark clinical theme CSS once per page render."""
    st.html(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.8rem;
            max-width: 1240px;
        }
        /* Hero header */
        .clinical-hero {
            background: linear-gradient(135deg, #0F1A30 0%, #0D2338 55%, #0A1526 100%);
            border: 1px solid rgba(20, 184, 166, 0.30);
            border-radius: 14px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px rgba(3, 8, 20, 0.45);
        }
        .clinical-hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #2DD4BF;
            margin-bottom: 0.45rem;
        }
        .clinical-hero h1 {
            margin: 0 0 0.4rem 0;
            font-size: 1.72rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #F1F5F9;
            line-height: 1.2;
        }
        .clinical-hero p {
            margin: 0;
            color: #94A3B8;
            font-size: 0.95rem;
            line-height: 1.55;
            max-width: 72ch;
        }

        /* Soft cards */
        .clinical-card {
            background: rgba(17, 28, 51, 0.85);
            border: 1px solid #1E2D4A;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 4px 16px rgba(3, 8, 20, 0.25);
        }
        .clinical-card-title {
            font-size: 0.98rem;
            font-weight: 600;
            color: #F1F5F9;
            margin: 0 0 0.35rem 0;
        }
        .clinical-card-body {
            font-size: 0.87rem;
            color: #94A3B8;
            line-height: 1.5;
            margin: 0;
        }

        /* Metric tiles */
        .metric-tile {
            background: linear-gradient(180deg, #111C33 0%, #0D1629 100%);
            border: 1px solid #1E2D4A;
            border-top: 2px solid #14B8A6;
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
            height: 100%;
        }
        .metric-tile-label {
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 0.35rem;
        }
        .metric-tile-value {
            font-size: 1.48rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }
        .metric-tile-sub {
            font-size: 0.78rem;
            color: #2DD4BF;
            margin-top: 0.35rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }

        /* Badges */
        .clinical-badge {
            display: inline-block;
            padding: 0.18rem 0.58rem;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-right: 0.35rem;
            margin-bottom: 0.25rem;
        }
        .badge-teal {
            background: rgba(20, 184, 166, 0.15);
            color: #2DD4BF;
            border: 1px solid rgba(20, 184, 166, 0.38);
        }
        .badge-sky {
            background: rgba(56, 189, 248, 0.14);
            color: #7DD3FC;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }
        .badge-amber {
            background: rgba(245, 158, 11, 0.14);
            color: #FCD34D;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .badge-rose {
            background: rgba(244, 63, 94, 0.14);
            color: #FDA4AF;
            border: 1px solid rgba(244, 63, 94, 0.35);
        }
        .badge-slate {
            background: rgba(148, 163, 184, 0.14);
            color: #CBD5E1;
            border: 1px solid rgba(148, 163, 184, 0.30);
        }

        /* Dialogue viewer */
        .dialogue-box {
            background: #080F1D;
            border: 1px solid #1E2D4A;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            max-height: 460px;
            overflow-y: auto;
            font-size: 0.87rem;
            line-height: 1.5;
        }
        .turn-row {
            padding: 0.45rem 0.65rem;
            border-radius: 8px;
            margin-bottom: 0.45rem;
        }
        .turn-doc {
            background: rgba(20, 184, 166, 0.09);
            border-left: 3px solid #14B8A6;
            color: #E2E8F0;
        }
        .turn-pat {
            background: rgba(56, 189, 248, 0.08);
            border-left: 3px solid #38BDF8;
            color: #E2E8F0;
        }
        .turn-other {
            background: rgba(148, 163, 184, 0.08);
            border-left: 3px solid #64748B;
            color: #CBD5E1;
        }
        .turn-speaker {
            font-weight: 700;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-right: 0.4rem;
            color: #2DD4BF;
        }
        .turn-pat .turn-speaker {
            color: #7DD3FC;
        }
        .num-supported {
            background: rgba(16, 185, 129, 0.22);
            color: #6EE7B7;
            border: 1px solid rgba(16, 185, 129, 0.55);
            border-radius: 5px;
            padding: 0.05rem 0.32rem;
            font-weight: 700;
            cursor: help;
        }
        .num-unsupported {
            background: rgba(244, 63, 94, 0.24);
            color: #FDA4AF;
            border: 1px solid rgba(244, 63, 94, 0.60);
            border-radius: 5px;
            padding: 0.05rem 0.32rem;
            font-weight: 700;
            cursor: help;
            text-decoration: underline wavy rgba(244, 63, 94, 0.7);
        }

        /* Sidebar notice */
        .sidebar-notice {
            background: rgba(245, 158, 11, 0.10);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 10px;
            padding: 0.7rem 0.85rem;
            font-size: 0.8rem;
            color: #FDE68A;
            line-height: 1.45;
            margin-top: 0.5rem;
        }

        /* Polish native widgets */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid rgba(20, 184, 166, 0.35);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid #1E2D4A;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.45rem 0.9rem;
            background: rgba(17, 28, 51, 0.55);
        }
        div[data-testid="stExpander"] {
            background: rgba(17, 28, 51, 0.55);
            border: 1px solid #1E2D4A;
            border-radius: 10px;
        }
        </style>
        """
    )


def render_badge(text: str, variant: str = "teal") -> str:
    return f'<span class="clinical-badge badge-{html.escape(variant)}">{html.escape(text)}</span>'


def render_hero(title: str, subtitle: str, kicker: str = "NoteChat · Local Clinical LLM Eval") -> None:
    st.html(
        f"""
        <div class="clinical-hero">
            <div class="clinical-hero-kicker">{html.escape(kicker)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """
    )


def render_metric_tile(label: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="metric-tile-sub">{html.escape(sub)}</div>' if sub else ""
    st.html(
        f"""
        <div class="metric-tile">
            <div class="metric-tile-label">{html.escape(label)}</div>
            <div class="metric-tile-value">{html.escape(value)}</div>
            {sub_html}
        </div>
        """
    )


def render_card(title: str, body: str, badges: list[tuple[str, str]] | None = None) -> None:
    badges_html = "".join(render_badge(t, v) for t, v in (badges or []))
    st.html(
        f"""
        <div class="clinical-card">
            <div class="clinical-card-title">{html.escape(title)}</div>
            {badges_html}
            <p class="clinical-card-body">{html.escape(body)}</p>
        </div>
        """
    )


def render_dialogue(text: str) -> None:
    """Render a Doctor:/Patient: dialogue inside a styled scrollable container."""
    if not text:
        st.info("No dialogue text available.")
        return
    rows: list[str] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Doctor:"):
            body = html.escape(line[len("Doctor:") :].strip())
            rows.append(f'<div class="turn-row turn-doc"><span class="turn-speaker">Doctor</span>{body}</div>')
        elif line.startswith("Patient:"):
            body = html.escape(line[len("Patient:") :].strip())
            rows.append(f'<div class="turn-row turn-pat"><span class="turn-speaker">Patient</span>{body}</div>')
        else:
            rows.append(f'<div class="turn-row turn-other">{html.escape(line)}</div>')
    st.html(f'<div class="dialogue-box">{"".join(rows)}</div>')


def render_sidebar() -> None:
    """Render shared sidebar branding, links, and clinical disclaimer."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=44, alt="NoteChat QLoRA Eval logo")
        st.markdown("### NoteChat QLoRA Eval")
        st.caption("On-prem QLoRA vs. large zero-shot & retrieval baselines on clinical dialogue generation.")
        st.divider()
        st.markdown("**Repository & Docs**")
        st.markdown(f"- [GitHub Repository]({GITHUB_URL})")
        for label, url in DOCS_URLS.items():
            st.markdown(f"- [{label}]({url})")
        st.html(
            """
            <div class="sidebar-notice">
                <strong>Notice:</strong> Research demo, not medical advice. Outputs are synthetic
                dialogues for methodology evaluation and must not be used clinically.
            </div>
            """
        )


def render_footer() -> None:
    """Render a consistent footer with repository link."""
    st.divider()
    st.caption(
        f"NoteChat QLoRA Eval · Research demo, not medical advice · "
        f"[GitHub Repository]({GITHUB_URL})"
    )
