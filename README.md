# TextForge — GPT-2 Text Studio

**Prodigy InfoTech Generative AI Internship — Task 01**
*Fine-tuning and generation with OpenAI's GPT-2 language model.*

---

## What this project does

TextForge builds a real, working pipeline for:

1. **Text generation** with the pretrained `gpt2` model from Hugging Face.
2. **Custom fine-tuning** of `gpt2` on a small demonstration text dataset.
3. A polished **Streamlit UI** for interactive generation and model comparison.

This is not a simulated demo — the generation uses the real `model.generate()` API, and fine-tuning uses the real Hugging Face `Trainer`.

---

## High-level concept: how GPT-2 text generation works

GPT-2 is a causal language model. Given a sequence of tokens (words/sub-words), it predicts the next token one at a time. We feed it a user prompt, and the model continues the sequence. Key controls:

- **Temperature** — lowers randomness (closer to 0), increases creativity (closer to 2).
- **Top-p (nucleus sampling)** — selects from the smallest set of tokens whose cumulative probability exceeds `p`.
- **Repetition penalty** — discourages repeating words already in the prompt/response.
- **Max new tokens** — limits output length.

Fine-tuning adjusts the model's weights so that generated text matches the style of the training dataset.

---

## Project architecture

```text
.
├── app.py                      # Streamlit UI
├── train.py                    # Real fine-tuning script
├── requirements.txt            # Dependency list
├── README.md                   # This file
├── data/
│   ├── train.txt               # Small demonstration training corpus
│   └── README.md               # Notes on dataset customization
├── models/
│   └── .gitkeep                 # Fine-tuned checkpoints saved here
├── src/
│   └── textforge/
│       ├── __init__.py         # Package root
│       ├── config.py           # Paths, defaults, device detection
│       ├── dataset.py          # Data loading, validation, chunking
│       ├── generation.py       # Real generation with model.generate
│       └── model_manager.py    # Model loading and device handling
├── tests/
│   ├── test_dataset.py         # Dataset validation tests
│   ├── test_generation.py      # Generation logic tests (with mocks)
│   └── test_app.py             # App structure tests
└── outputs/
    └── .gitkeep                 # Generated text downloads
```

---

## Python environment setup

This project is tested on **macOS** with **Python 3.12** (also works with 3.11).

### Using a virtual environment (recommended)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Installing dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs the verified dependency versions pinned in `requirements.txt`:

- `torch` 2.2.2
- `transformers` 4.46.3
- `tokenizers` 0.20.3
- `huggingface-hub` 0.36.2
- `numpy` 1.26.4
- `accelerate` 1.15.0
- `streamlit` 1.66.1
- `pytest` 9.1.1

No NVIDIA GPU is required — the code detects CPU/MPS/CUDA automatically.

---

## Launching the Streamlit application

```bash
source .venv/bin/activate
streamlit run app.py
```

Then open the local URL shown (usually `http://localhost:8501`).

---

## How to generate text with the base model

1. Open the app (`streamlit run app.py`).
2. In the sidebar, select **"Pretrained GPT-2"** (default).
3. Type a prompt in the text area (e.g. `The future of technology belongs to`).
4. Adjust settings (temperature, max tokens, etc.).
5. Click **Generate**.
6. Copy the output or download it as `.txt`.

The generation is real — it loads the `gpt2` model weights and calls `model.generate()`.

---

## How to fine-tune GPT-2 on the provided dataset

### Short CPU smoke test (verified)

```bash
python train.py \
  --model-name gpt2 \
  --data-path data/train.txt \
  --output-dir outputs/gpt2-smoke-test-fresh \
  --epochs 1 \
  --batch-size 1 \
  --block-size 32 \
  --max-steps 2 --device cpu
```

This command runs a two-step CPU smoke test and saves its checkpoint to `outputs/gpt2-smoke-test-fresh/`. It verifies the training pipeline; it is not a full fine-tuning run.

### Repeat the smoke test in a separate directory

```bash
python train.py \
  --model-name gpt2 \
  --data-path data/train.txt \
  --output-dir outputs/gpt2-longer-smoke-fresh \
  --epochs 1 \
  --batch-size 1 \
  --block-size 32 \
  --max-steps 2 --device cpu
```

> **Note:** Choose a new, empty output directory for each run. The training script refuses to run if the destination already contains files.

---

## How to fine-tune on a custom `.txt` dataset

1. Create your `.txt` file (UTF-8, at least ~500 characters, ~5 non-empty lines):

```bash
# Example custom dataset
cat > data/my_custom.txt << 'EOF'
Custom language for my project.
Every line should be original text.
Include varied sentence structures.
Keep paragraphs separated by blank lines.
EOF
```

2. Train with your file:

```bash
python train.py \
  --data-path data/my_custom.txt \
  --output-dir outputs/my-custom-smoke-test \
  --epochs 1 --batch-size 1 --block-size 32 --max-steps 2 --device cpu
```

3. The script validates the dataset before starting — empty or whitespace-only files produce clear errors instead of silent failures.

---

## How to launch the app with the fine-tuned model

The app looks for a fine-tuned checkpoint in `models/gpt2-finetuned/`. The short smoke test above writes to `outputs/gpt2-smoke-test-fresh/` and validates training; it does not automatically make the app use that checkpoint.

On a fresh clone, create a checkpoint at the path expected by the app with this short CPU run:

```bash
python train.py --model-name gpt2 --data-path data/train.txt --output-dir models/gpt2-finetuned --epochs 1 --batch-size 1 --block-size 32 --max-steps 2 --device cpu
```

The generated checkpoint stays local and is excluded from Git by `.gitignore`.

After training, continue with these steps:

1. Restart the app: `streamlit run app.py`
2. In the sidebar, select **"Fine-tuned GPT-2 (local)"**.
3. Generate text. The model now reflects the style of your custom dataset.

If the fine-tuned checkpoint is missing, the UI clearly reports this and explains how to train one.

---

## How to run automated tests

```bash
python -m pytest tests/ -v
```

Tests cover:

- Dataset validation (valid, missing, empty, whitespace-only, short)
- Generation parameter validation (invalid tokens, temperature, top-p, etc.)
- Mocked generation tests (no model download needed for ordinary test runs)
- App structure and headless Streamlit execution

No external API access or large model downloads are required for ordinary unit tests.

---

## Troubleshooting

### Model download fails
- Check internet connection.
- Verify `HF_HOME` / `.cache/huggingface/transformers/` is not corrupt.
- Restart the app and try the pretrained mode first.

### Memory / out-of-memory errors
- Use `--batch-size 1` for training.
- Use CPU mode (`device` auto-detects; GPU is not forced).
- Reduce `--max-steps` or `--block-size`.

### Dependency conflicts
- Always use the provided `.venv` and `requirements.txt`.
- If `torch` installation fails on macOS, use the CPU-only index:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Checkpoint errors
- Verify the checkpoint directory exists (`models/gpt2-finetuned/`).
- Ensure `pytorch_model.bin` (or `config.json` + weights) is present.
- Never delete an existing checkpoint; the app will not remove it automatically.

---

## Known limitations

- This is a **demonstration corpus** (`data/train.txt`), not production data. Fine-tuning on such a small dataset improves style slightly but does not create a high-quality domain model.
- The model runs on CPU/MPS by default; no NVIDIA GPU is required but training is slower than a GPU.
- The Streamlit UI is for interactive exploration, not production deployment.
- Full fine-tuning may take substantial time and disk space (~500 MB model + output files).

---

## Acknowledgements

- The base model (`gpt2`) is a pretrained language model released by OpenAI.
- Before fine-tuning completes successfully, the model remains the **pretrained** `gpt2`. Only after the training script saves a valid checkpoint is the fine-tuned mode available.
- This project is for educational demonstration purposes only.

---

## Quick reference commands

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py

# Run a two-step CPU smoke test
python train.py --data-path data/train.txt --output-dir outputs/gpt2-smoke-test-fresh --epochs 1 --batch-size 1 --block-size 32 --max-steps 2 --device cpu

# Run tests
python -m pytest tests/ -v
```

---

*TextForge — built for the Prodigy InfoTech Generative AI Internship, Task 01.*
