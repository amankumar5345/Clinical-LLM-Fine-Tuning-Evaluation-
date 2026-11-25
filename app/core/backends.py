"""Backend interface and implementations for 4_Playground.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data import get_eval_bundle  # noqa: E402
from core.live_eval import SYNTHETIC_EXAMPLES  # noqa: E402
from src.inference.local_hf import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
DEFAULT_ADAPTER_DIR = REPO_ROOT / "artifacts" / "adapters" / "unsloth__Qwen2.5-3B-Instruct-bnb-4bit" / "final_adapter"


class Backend(ABC):
    """Abstract generation backend."""

    key: str
    label: str
    supports_streaming: bool = False

    @abstractmethod
    def status(self) -> tuple[bool, str]:
        """Return (is_available, honest_status_message)."""

    @abstractmethod
    def generate(self, note: str, *, note_id: str | None = None, max_new_tokens: int = 512, do_sample: bool = False) -> Iterator[str]:
        """Yield generated text chunks (or a single full string if non-streaming)."""


class CachedExamplesBackend(Backend):
    """Default backend: replays saved fine-tuned outputs or synthetic examples (no model)."""

    key = "cached"
    label = "Cached Examples (Default · No Model)"
    supports_streaming = False

    def status(self) -> tuple[bool, str]:
        bundle = get_eval_bundle()
        if bundle.example_notes:
            return True, f"Ready · {len(bundle.example_notes)} saved test records"
        return True, "Ready · Synthetic fallback examples only"

    def generate(
        self, note: str, *, note_id: str | None = None, max_new_tokens: int = 512, do_sample: bool = False
    ) -> Iterator[str]:
        bundle = get_eval_bundle()
        if note_id and note_id in bundle.example_notes:
            ex = bundle.example_notes[note_id]
            rec = ex.generations.get("finetuned") or next(iter(ex.generations.values()), None)
            if rec is not None:
                yield rec.generated
                return
        for ex in SYNTHETIC_EXAMPLES.values():
            if ex["note"].strip() == note.strip():
                yield ex["dialogue"]
                return
        yield "N/A — no cached generation matches this custom note. Select a saved note_id or enable a model backend."


@st.cache_resource(show_spinner="Loading base model + LoRA adapter (transformers + peft + bitsandbytes)...")
def _load_peft_model(base_model: str, adapter_dir: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


class LocalAdapterBackend(Backend):
    """Local 4-bit base model + LoRA adapter via transformers + peft + bitsandbytes."""

    key = "local"
    label = "Local QLoRA Adapter (GPU)"
    supports_streaming = True

    def status(self) -> tuple[bool, str]:
        try:
            import torch

            has_gpu = bool(torch.cuda.is_available())
        except ImportError:
            has_gpu = False
        if not has_gpu:
            return False, "Disabled · No CUDA GPU detected"

        missing_pkgs = [p for p in ("transformers", "peft", "bitsandbytes") if importlib.util.find_spec(p) is None]
        if missing_pkgs:
            return False, f"Disabled · Missing packages: {', '.join(missing_pkgs)}"

        if not DEFAULT_ADAPTER_DIR.exists():
            return False, f"Disabled · Adapter not found at {DEFAULT_ADAPTER_DIR.name}"

        return True, "Ready · CUDA GPU + transformers/peft/bitsandbytes"

    def generate(
        self, note: str, *, note_id: str | None = None, max_new_tokens: int = 512, do_sample: bool = False
    ) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        ok, msg = self.status()
        if not ok:
            raise RuntimeError(msg)

        model, tokenizer = _load_peft_model(DEFAULT_BASE_MODEL, str(DEFAULT_ADAPTER_DIR))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt({"clinical_note": note})},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs: dict[str, Any] = {
            "input_ids": inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if do_sample:
            gen_kwargs["temperature"] = 0.7

        def _run() -> None:
            with torch.no_grad():
                model.generate(**gen_kwargs)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        yield from streamer
        thread.join(timeout=5)


class RemoteEndpointBackend(Backend):
    """Remote OpenAI-compatible or custom JSON inference endpoint configured via st.secrets."""

    key = "remote"
    label = "Remote Endpoint (st.secrets)"
    supports_streaming = True

    def _read_secrets(self) -> tuple[str | None, str | None, str]:
        try:
            section = st.secrets.get("remote_llm", {})
            url = section.get("url") or st.secrets.get("REMOTE_LLM_URL")
            token = section.get("token") or st.secrets.get("REMOTE_LLM_TOKEN")
            model = section.get("model") or "qwen2.5-3b-qlora"
            return (str(url) if url else None, str(token) if token else None, str(model))
        except Exception:
            return None, None, "qwen2.5-3b-qlora"

    def status(self) -> tuple[bool, str]:
        url, token, _ = self._read_secrets()
        if not url or not token:
            return False, "Disabled · Set remote_llm.url and remote_llm.token in st.secrets"
        return True, f"Ready · Endpoint configured ({url})"

    def generate(
        self, note: str, *, note_id: str | None = None, max_new_tokens: int = 512, do_sample: bool = False
    ) -> Iterator[str]:
        url, token, model_id = self._read_secrets()
        if not url or not token:
            raise RuntimeError("Remote endpoint URL/token missing in st.secrets.")

        payload = json.dumps(
            {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt({"clinical_note": note})},
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0.7 if do_sample else 0.0,
                "stream": True,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue


def get_backends() -> dict[str, Backend]:
    backends: list[Backend] = [CachedExamplesBackend(), LocalAdapterBackend(), RemoteEndpointBackend()]
    return {b.key: b for b in backends}
