"""TextForge — GPT-2 Text Studio

Streamlit application for:
- Text generation with pretrained GPT-2
- Text generation with locally fine-tuned GPT-2
- Model status and training workflow documentation

Run with: streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.textforge` imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hfcache"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.textforge.config import (
    PROJECT_ROOT,
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_NUM_RETURN_SEQUENCES,
    MAX_MAX_NEW_TOKENS,
    MAX_NUM_RETURN_SEQUENCES,
    check_fine_tuned_checkpoint,
    DEFAULT_TRAINING_OUTPUT,
    TRAIN_FILE,
    MODELS_DIR,
)
from src.textforge.generation import (
    generate_text,
    GenerationError,
    _cached_model,
)
from src.textforge.model_manager import load_model, get_device, ModelError

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TextForge — GPT-2 Text Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS for polished look
# -----------------------------------------------------------------------------
st.markdown("""
<style>
:root {
  --primary: #2E3A59;
  --secondary: #5B6A8A;
  --accent: #7A3FC7;
  --bg: #FAFAFA;
  --card-bg: #FFFFFF;
  --border: #DEE2E6;
  --text: #1F2937;
  --muted: #6B7280;
}
.app-header {
  background: linear-gradient(135deg, #2E3A59 0%, #5B6A8A 100%);
  color: white;
  padding: 2rem;
  border-radius: 12px;
  margin-bottom: 2rem;
}
.app-header h1 { margin: 0 0 0.5rem 0; font-size: 2rem; }
.app-header p { margin: 0; opacity: 0.9; font-size: 1rem; }
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.card h3 {
  margin-top: 0;
  font-size: 1.1rem;
  color: var(--primary);
  border-bottom: 2px solid var(--accent);
  display: inline-block;
  padding-bottom: 0.4rem;
}
.model-badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  margin-right: 0.5rem;
}
.badge-pretrained { background: #E5E7EB; color: #374151; border: 1px solid #D1D5DB; }
.badge-finetuned { background: #E6E9FF; color: #4B2E99; border: 1px solid #A5B4FC; }
.badge-missing { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
.muted { color: var(--muted); font-size: 0.9rem; }
.small { font-size: 0.85rem; color: var(--muted); }
.stButton > button {
  background: var(--accent);
  color: white;
  border: none;
  padding: 0.6rem 1.2rem;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
}
.stButton > button:hover { opacity: 0.9; }
.stButton > button:disabled { opacity: 0.6; cursor: not-allowed; }
.output-box {
  background: #1E1E2E;
  color: #CDD6F4;
  border-radius: 8px;
  padding: 1rem;
  font-family: ui-monospace, Menlo, Monaco, "Cascadia Mono", monospace;
  font-size: 0.9rem;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-x: auto;
}
.label-small { font-size: 0.85rem; font-weight: 600; color: var(--secondary); }
.help-text { font-size: 0.8rem; color: var(--muted); margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar — model selection & status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Selection")
    st.markdown("---")

    checkpoint = check_fine_tuned_checkpoint()

    mode_options = ["Pretrained GPT-2"]
    if checkpoint is not None:
        mode_options.append("Fine-tuned GPT-2 (local)")
    else:
        mode_options.append("Fine-tuned GPT-2 (not available)")

    selected_mode = st.selectbox(
        "Model mode",
        options=mode_options,
        index=0,
        help="Pretrained GPT-2 uses the base OpenAI model. Fine-tuned uses your trained checkpoint.",
    )

    is_finetuned_mode = "Fine-tuned" in selected_mode and "(not available)" not in selected_mode

    st.markdown("### Status")
    st.markdown("---")

    model_status_placeholder = st.empty()
    if is_finetuned_mode:
        model_status_placeholder.success("Fine-tuned checkpoint found")
        st.markdown(f"**Checkpoint:** `{DEFAULT_TRAINING_OUTPUT}`")
    else:
        model_status_placeholder.info("Pretrained GPT-2 (base model)")
        st.markdown("Using base model from HuggingFace Hub")

    if checkpoint is None:
        st.markdown("---")
        st.markdown("### Train your own model")
        st.markdown(
            """
To enable fine-tuned mode, run:

```bash
python train.py --data-path data/train.txt --output-dir models/gpt2-finetuned --epochs 3 --batch-size 2 --block-size 128 --max-steps 500
```

Then restart the app and select "Fine-tuned GPT-2".
""",
        )

    st.markdown("---")
    st.markdown("### Device")
    device = get_device()
    st.markdown(f"**Detected:** `{device.upper()}`")
    st.caption("macOS: CUDA is not available; CPU or MPS may be used.")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
  <h1>🤖 TextForge</h1>
  <p>GPT-2 Text Studio &mdash; real pretrained and fine-tuned text generation</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**TextForge** lets you generate text with OpenAI's GPT-2 language model — either the
publicly available pretrained version or a version fine-tuned on your own corpus.

> First-time users: select "Pretrained GPT-2" from the sidebar, type a prompt, and click Generate.
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Text generation workspace
# -----------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h3>Text Generation</h3>', unsafe_allow_html=True)

prompt = st.text_area(
    "Enter your prompt",
    placeholder="Type your prompt here... e.g. The future of technology belongs to",
    height=160,
    help="The model will continue from this text.",
)

c1, c2 = st.columns([1, 4])
generate_btn = c1.button("Generate", type="primary", use_container_width=True)

gen_placeholder = st.empty()

# -----------------------------------------------------------------------------
# Generation settings
# -----------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h3>Generation Settings</h3>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

max_new_tokens = col1.slider(
    "Max new tokens",
    min_value=8,
    max_value=MAX_MAX_NEW_TOKENS,
    value=DEFAULT_MAX_NEW_TOKENS,
    step=8,
    help="Maximum number of tokens to generate (avoid extremely long outputs).",
)

temperature = col2.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=DEFAULT_TEMPERATURE,
    step=0.1,
    help="Lower = more deterministic; higher = more creative/random.",
)

top_p = col3.slider(
    "Top-p",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_TOP_P,
    step=0.05,
    help="Nucleus sampling threshold (0 = disabled).",
)

repetition_penalty = col4.slider(
    "Repetition penalty",
    min_value=1.0,
    max_value=2.0,
    value=DEFAULT_REPETITION_PENALTY,
    step=0.05,
    help="Penalty for repeating tokens (>1 discourages repetition).",
)

num_outputs = st.select_slider(
    "Number of outputs",
    options=list(range(1, MAX_NUM_RETURN_SEQUENCES + 1)),
    value=DEFAULT_NUM_RETURN_SEQUENCES,
    help="How many completions to generate per prompt (max 3).",
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Generation logic
# -----------------------------------------------------------------------------
if generate_btn:
    if not prompt.strip():
        st.error("Please enter a prompt before generating.")
    else:
        # Check fine-tuned checkpoint availability
        if is_finetuned_mode:
            local_checkpoint = DEFAULT_TRAINING_OUTPUT
        else:
            local_checkpoint = None

        model_name = DEFAULT_MODEL_NAME

        with st.spinner("Loading model and generating text..."):
            try:
                results = generate_text(
                    prompt=prompt,
                    model_name=model_name,
                    local_checkpoint=local_checkpoint,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_outputs,
                )
                gen_placeholder.success("Generation complete!")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>Generated Output</h3>', unsafe_allow_html=True)
                for i, out_text in enumerate(results):
                    st.markdown(f"**Output {i+1}:**")
                    st.markdown(f'<div class="output-box">{out_text}</div>', unsafe_allow_html=True)
                    st.download_button(
                        label=f"Download Output {i+1} (.txt)",
                        data=out_text,
                        file_name=f"textforge_output_{i+1}.txt",
                        mime="text/plain",
                        key=f"download_{i}",
                        use_container_width=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            except GenerationError as exc:
                gen_placeholder.error(str(exc))
                st.info("Try selecting Pretrained GPT-2 in the sidebar if fine-tuned mode is unavailable.")
            except Exception as exc:
                gen_placeholder.error(f"Generation failed: {exc}")
                st.info("Check your internet connection if the model needs to be downloaded.")

# -----------------------------------------------------------------------------
# Training information
# -----------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h3>Training Information</h3>', unsafe_allow_html=True)

st.markdown('<span class="label-small">Configuration</span>', unsafe_allow_html=True)
train_info = {
    "Base model": DEFAULT_MODEL_NAME,
    "Training dataset": str(TRAIN_FILE),
    "Fine-tuned checkpoint": str(DEFAULT_TRAINING_OUTPUT),
    "Checkpoint exists": "Yes" if check_fine_tuned_checkpoint() is not None else "No",
}
for k, v in train_info.items():
    st.markdown(f"**{k}:** {v}")

st.markdown('<span class="label-small">Training workflow</span>', unsafe_allow_html=True)
st.markdown(
    """
1. **Prepare data** — Place a UTF-8 `.txt` file at `data/train.txt` (or any path you choose).
2. **Train** — Run the training script:

```bash
python train.py --data-path data/train.txt \
  --output-dir models/gpt2-finetuned \
  --epochs 3 --batch-size 2 --block-size 128 --max-steps 500
```

3. **Use** — Restart this app and select "Fine-tuned GPT-2 (local)" from the sidebar.

> A real fine-tuning run downloads `gpt2` (~500 MB) and trains on CPU/MPS/CUDA.
> Keep `--max-steps` small for a smoke test; full training may take hours.
""",
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="card muted small" style="margin-top: 2rem;">
TextForge — GPT-2 Text Studio &nbsp;|&nbsp; Project for Prodigy InfoTech Generative AI Internship — Task 01<br>
Base model: GPT-2 (OpenAI) via Hugging Face Transformers.
</div>
""", unsafe_allow_html=True)
