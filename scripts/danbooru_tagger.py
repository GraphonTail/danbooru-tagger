"""
danbooru_tagger.py — ForgeNeo / A1111 Forge extension.

Pipeline: Tagger → prompt → diffusion
  process() is called by Forge BEFORE diffusion — tags are merged into the
  prompt, then generation runs normally.

Mode is always "abstract": seeds come from the current prompt, generated
tags are new contextual tags (seeds themselves are not included in output).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import gradio as gr
from modules import scripts

try:
    from modules.ui_components import InputAccordion
    _HAS_INPUT_ACCORDION = True
except ImportError:
    _HAS_INPUT_ACCORDION = False

# ─── paths ─────────────────────────────────────────────────────────────────
EXT_DIR      = Path(__file__).resolve().parent.parent
TAGGER_LIB   = EXT_DIR / "tagger_lib"
VOCAB_PATH   = EXT_DIR / "data"  / "vocab_clean.json"
MODEL_DIR    = EXT_DIR / "model"
PROFILES_DIR = EXT_DIR / "data"  / "profiles"

# ─── compact-slider CSS (injected once per tab) ─────────────────────────────
_SLIDER_CSS = """
<style>
.dt-slider { margin: 0 0 4px 0 !important; }
.dt-slider .wrap { padding: 4px 0 2px !important; gap: 4px !important; }
.dt-slider input[type=number] { max-width: 64px !important; }
</style>
"""


def _ensure_lib_in_path() -> None:
    lib_str = str(TAGGER_LIB)
    if not any(os.path.normcase(p) == os.path.normcase(lib_str) for p in sys.path):
        sys.path.insert(0, lib_str)


_ensure_lib_in_path()


# ─── utilities ──────────────────────────────────────────────────────────────

def _scan_models() -> list[str]:
    if not MODEL_DIR.exists():
        return []
    return sorted(p.name for p in MODEL_DIR.glob("*.pth"))


def _scan_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.txt"))


def _files_ok(model_name: str) -> tuple[bool, str]:
    if not VOCAB_PATH.exists():
        return False, f"vocab not found: {VOCAB_PATH}"
    ckpt = MODEL_DIR / model_name
    if not ckpt.exists():
        return False, f"checkpoint not found: {ckpt}"
    return True, ""


def _get_tagger(model_name: str):
    _ensure_lib_in_path()
    ckpt = str(MODEL_DIR / model_name)
    try:
        from inference import TaggerInference
        return TaggerInference.get_instance(str(VOCAB_PATH), ckpt)
    except ModuleNotFoundError as e:
        print(f"[DanbooruTagger] Import error: {e}")
        traceback.print_exc()
        return None
    except Exception:
        traceback.print_exc()
        return None


def _seeds_from_prompt(prompt: str) -> list[str]:
    return [t.strip() for t in prompt.split(",") if t.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# EXTENSION
# ══════════════════════════════════════════════════════════════════════════════

class DanbooruTaggerScript(scripts.Script):

    def title(self) -> str:
        return "Danbooru Tagger"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    # ── UI ──────────────────────────────────────────────────────────────────

    def ui(self, is_img2img: bool):
        tab = "i2i" if is_img2img else "t2i"
        prompt_selector = f"#{'img2img' if is_img2img else 'txt2img'}_prompt textarea"

        _accordion = (
            InputAccordion(False, label="🏷️ Danbooru Tagger", elem_id=f"dt_acc_{tab}")
            if _HAS_INPUT_ACCORDION
            else gr.Accordion("🏷️ Danbooru Tagger", open=False, elem_id=f"dt_acc_{tab}")
        )

        with _accordion as enabled:

            if not _HAS_INPUT_ACCORDION:
                enabled = gr.Checkbox(label="Enable", value=False, elem_id=f"dt_en_{tab}")

            # compact-slider CSS
            gr.HTML(_SLIDER_CSS)

            # ── Model picker ───────────────────────────────────────────────
            initial_models = _scan_models()
            with gr.Row():
                model_sel = gr.Dropdown(
                    choices   = initial_models,
                    value     = initial_models[0] if initial_models else None,
                    label     = "Model",
                    scale     = 1,
                    elem_id   = f"dt_mdl_{tab}",
                )
                refresh_models_btn = gr.Button(
                    "⟳", scale=0, min_width=40, elem_id=f"dt_mref_{tab}",
                )
            refresh_models_btn.click(
                fn=lambda: gr.update(choices=_scan_models()),
                outputs=model_sel,
            )

            # ── Insert position ────────────────────────────────────────────
            insert_pos = gr.Radio(
                choices=["prepend", "append", "replace"],
                value="prepend",
                label="Insert",
                elem_id=f"dt_ins_{tab}",
            )

            # ── Sliders ────────────────────────────────────────────────────
            count = gr.Slider(
                minimum=1, maximum=500, value=25, step=1,
                label="Tag count",
                elem_id=f"dt_cnt_{tab}",
                elem_classes=["dt-slider"],
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=50.0, value=1.0, step=0.05,
                label="Temperature  (rec: 0.75–1.25)",
                elem_id=f"dt_tmp_{tab}",
                elem_classes=["dt-slider"],
            )
            spread = gr.Slider(
                minimum=0.1, maximum=50.0, value=1.5, step=0.05,
                label="Spread  (rec: 1.0–2.0)",
                elem_id=f"dt_spd_{tab}",
                elem_classes=["dt-slider"],
            )

            # ── Safety profiles ────────────────────────────────────────────
            with gr.Group(elem_id=f"dt_prof_grp_{tab}"):
                with gr.Row():
                    gr.Markdown("**Safety profiles** — `data/profiles/*.txt`")
                    refresh_prof_btn = gr.Button(
                        "⟳ Refresh", scale=0, min_width=90, elem_id=f"dt_ref_{tab}",
                    )
                initial_profiles = _scan_profiles()
                profiles_sel = gr.CheckboxGroup(
                    choices=initial_profiles,
                    value=[],
                    label="",
                    show_label=False,
                    elem_id=f"dt_prof_{tab}",
                )
                gr.HTML(
                    "<p style='margin:2px 0 0;color:#888;font-size:0.82em'>"
                    "No profiles — add <code>.txt</code> files to "
                    "<code>data/profiles/</code> and click ⟳"
                    "</p>"
                    if not initial_profiles else
                    f"<p style='margin:2px 0 0;color:#888;font-size:0.82em'>"
                    f"{len(initial_profiles)} profile(s) loaded</p>"
                )
            refresh_prof_btn.click(
                fn=lambda: gr.update(choices=_scan_profiles()),
                outputs=profiles_sel,
            )

            # ── Preview ────────────────────────────────────────────────────
            preview_btn = gr.Button(
                "🔍  Preview tags",
                variant="primary",
                elem_id=f"dt_pbtn_{tab}",
            )
            preview_out = gr.Textbox(
                label="Generated tags",
                placeholder="click Preview to generate...",
                interactive=False,
                lines=3,
                elem_id=f"dt_pout_{tab}",
            )
            with gr.Row():
                copy_btn = gr.Button(
                    "📋 Copy",
                    scale=1,
                    elem_id=f"dt_copy_{tab}",
                )
                inject_btn = gr.Button(
                    "⬆️ Inject into prompt",
                    scale=1,
                    elem_id=f"dt_inj_{tab}",
                )

            preview_btn.click(
                fn=self._preview,
                inputs=[model_sel, count, temperature, spread, profiles_sel],
                outputs=preview_out,
            )

            # Copy: write to clipboard via JS (no Python roundtrip needed)
            copy_btn.click(
                fn=None,
                inputs=[preview_out],
                outputs=[],
                _js="""(tags) => {
                    if (tags) navigator.clipboard.writeText(tags);
                }""",
            )

            # Inject: push tags directly into the active prompt textarea
            inject_btn.click(
                fn=None,
                inputs=[preview_out],
                outputs=[],
                _js=f"""(tags) => {{
                    if (!tags) return;
                    const el = document.querySelector('{prompt_selector}');
                    if (!el) return;
                    const setter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value').set;
                    setter.call(el, tags);
                    el.dispatchEvent(new Event('input', {{bubbles: true}}));
                }}""",
            )

        return [
            enabled, model_sel,
            insert_pos,
            count, temperature, spread,
            profiles_sel,
        ]

    # ── Preview ─────────────────────────────────────────────────────────────

    @staticmethod
    def _preview(model_sel, count, temperature, spread, profiles_sel) -> str:
        if not model_sel:
            return "❌ No model selected — add .pth files to model/"
        ok, err = _files_ok(model_sel)
        if not ok:
            return f"❌ {err}"
        tagger = _get_tagger(model_sel)
        if tagger is None:
            return "❌ Model not loaded — check console"
        try:
            tags = tagger.generate(
                mode        = "abstract",
                seed_tags   = [],          # no prompt context in preview
                count       = int(count),
                temperature = float(temperature),
                spread      = float(spread),
                sort        = True,
                profiles    = list(profiles_sel) if profiles_sel else [],
            )
            return ", ".join(tags) if tags else "(no result)"
        except Exception as exc:
            traceback.print_exc()
            return f"❌ {exc}"

    # ── Process ─────────────────────────────────────────────────────────────

    def process(self, p,
                enabled, model_sel,
                insert_pos,
                count, temperature, spread,
                profiles_sel):

        if not enabled:
            return

        if not model_sel:
            print("[DanbooruTagger] ⚠ No model selected — skipping")
            return

        ok, err = _files_ok(model_sel)
        if not ok:
            print(f"[DanbooruTagger] ⚠ {err} — skipping")
            return

        tagger = _get_tagger(model_sel)
        if tagger is None:
            print("[DanbooruTagger] ⚠ Model not loaded — skipping")
            return

        profiles = list(profiles_sel) if profiles_sel else []

        for i, prompt in enumerate(p.all_prompts):
            seeds = _seeds_from_prompt(prompt)

            try:
                tags = tagger.generate(
                    mode        = "abstract",
                    seed_tags   = seeds,
                    count       = int(count),
                    temperature = float(temperature),
                    spread      = float(spread),
                    sort        = True,
                    profiles    = profiles,
                )
            except Exception:
                traceback.print_exc()
                continue

            if not tags:
                continue

            gen_str = ", ".join(tags)
            preview = gen_str[:120] + ("…" if len(gen_str) > 120 else "")
            print(f"[DanbooruTagger] [{insert_pos}] #{i}: {preview}")

            if insert_pos == "prepend":
                p.all_prompts[i] = gen_str + (", " + prompt if prompt.strip() else "")
            elif insert_pos == "append":
                p.all_prompts[i] = (prompt + ", " if prompt.strip() else "") + gen_str
            else:  # replace
                p.all_prompts[i] = gen_str

        # keep p.prompt in sync with the first image slot
        if p.all_prompts:
            p.prompt = p.all_prompts[0]
