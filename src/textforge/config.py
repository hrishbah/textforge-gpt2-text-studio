"""Configuration, paths and constants used across the TextForge project.

This module centralises:
- Filesystem locations (datasets, checkpoints, cache)
- Default generation hyperparameters
- Validation thresholds for user input
- Device detection for CPU/GPU/MPS

All paths are relative to the project root unless explicitly absolute.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ==============================================================================
# Filesystem paths (all resolved relative to project root)
# ==============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
TRAIN_FILE: Final[Path] = DATA_DIR / "train.txt"

OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

# Default location where a fine-tuned checkpoint is saved
DEFAULT_TRAINING_OUTPUT: Final[Path] = MODELS_DIR / "gpt2-finetuned"

# HuggingFace cache directory (respects HF_HOME or TRANSFORMERS_CACHE)
HF_CACHE: Final[Path] = Path(
    os.environ.get("HF_HOME")
    or os.environ.get("TRANSFORMERS_CACHE")
    or Path.home() / ".cache" / "huggingface" / "transformers"
)

# ==============================================================================
# Model selection constants
# ==============================================================================

# Base pretrained model name from HuggingFace hub
DEFAULT_MODEL_NAME: Final[str] = "gpt2"

# The exact model class used for causal language modelling
MODEL_CLASS_NAME: Final[str] = "GPT2LMHeadModel"
TOKENIZER_CLASS_NAME: Final[str] = "GPT2Tokenizer"

# ==============================================================================
# Default generation hyperparameters
# ==============================================================================

# These are the default values shown in the Streamlit UI
DEFAULT_MAX_NEW_TOKENS: Final[int] = 256
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_TOP_P: Final[float] = 0.95
DEFAULT_REPETITION_PENALTY: Final[float] = 1.1
DEFAULT_NUM_RETURN_SEQUENCES: Final[int] = 1

# Safe upper bounds (user cannot exceed)
MAX_MAX_NEW_TOKENS: Final[int] = 1024
MAX_NUM_RETURN_SEQUENCES: Final[int] = 3

# ==============================================================================
# Training configuration defaults
# ==============================================================================

DEFAULT_TRAIN_EPOCHS: Final[int] = 3
DEFAULT_TRAIN_BATCH_SIZE: Final[int] = 2
DEFAULT_TRAIN_BLOCK_SIZE: Final[int] = 128
DEFAULT_MAX_TRAIN_STEPS: Final[int] = 500  # ~smoke-test size

# Minimum dataset size required for training (non-empty lines)
MIN_DATASET_LINES: Final[int] = 5
MIN_DATASET_CHARS: Final[int] = 500

# ==============================================================================
# Dataset file requirements
# ==============================================================================

# Acceptable file extensions for training data
VALID_DATASET_EXTENSIONS: Final[tuple[str, ...]] = (".txt",)


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def check_fine_tuned_checkpoint() -> Path | None:
    """Return the path to the fine-tuned checkpoint if it exists.

    Checks ``DEFAULT_TRAINING_OUTPUT`` by default. Returns ``None`` if
    no valid checkpoint directory with model files is found.
    """
    checkpoint = DEFAULT_TRAINING_OUTPUT
    if checkpoint.is_dir():
        # GPT-2 checkpoint needs at least pytorch_model.bin
        if (checkpoint / "pytorch_model.bin").exists():
            return checkpoint
        # Or config.json + tokenizer files
        if any(
            (checkpoint / name).exists()
            for name in ("config.json", "pytorch_model.bin", "model.safetensors")
        ):
            return checkpoint
    return None