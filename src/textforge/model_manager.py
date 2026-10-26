"""Model manager: loads and caches GPT-2 models for TextForge."""
from __future__ import annotations
from pathlib import Path

class ModelError(Exception):
    pass

def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch.backends.mps as mps
        if mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def load_model(model_name: str = "gpt2", local_checkpoint: Path | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .config import DEFAULT_MODEL_NAME
    model_id = model_name
    if local_checkpoint is not None:
        checkpoint = Path(local_checkpoint)
        if checkpoint.is_dir():
            model_id = str(checkpoint)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if model_id == DEFAULT_MODEL_NAME:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as exc:
        raise ModelError(
            f"Failed to load model '{model_id}'. Check internet or checkpoint."
        ) from exc
    return model, tokenizer, model_id
