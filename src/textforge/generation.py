"""Text generation logic for TextForge.

This module wraps the real Hugging Face ``model.generate()`` API. It is
deliberately free of heavy imports at module load time so that unit tests can
exercise the validation and orchestration logic without downloading GPT-2.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_RETURN_SEQUENCES,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_MAX_NEW_TOKENS,
    MAX_NUM_RETURN_SEQUENCES,
)
from .model_manager import ModelError, get_device, load_model


class GenerationError(Exception):
    """Raised when generation cannot proceed (bad input or model failure)."""


@lru_cache(maxsize=4)
def _cached_model(model_name: str = "gpt2", local_checkpoint: str | None = None):
    """Load a model once and reuse it across requests.

    Returns
    -------
    tuple
        ``(model, tokenizer, resolved_model_id)``
    """
    checkpoint = Path(local_checkpoint) if local_checkpoint else None
    model, tokenizer, resolved_id = load_model(model_name, checkpoint)
    return model, tokenizer, resolved_id


def validate_generation_settings(
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    num_return_sequences: int = DEFAULT_NUM_RETURN_SEQUENCES,
) -> None:
    """Validate generation hyperparameters.

    Raises
    ------
    GenerationError
        If any setting is outside its supported range.
    """
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise GenerationError("max_new_tokens must be an integer.")
    if not 1 <= max_new_tokens <= MAX_MAX_NEW_TOKENS:
        raise GenerationError(
            f"max_new_tokens must be between 1 and {MAX_MAX_NEW_TOKENS}, "
            f"got {max_new_tokens}."
        )

    if not 0.1 <= float(temperature) <= 2.0:
        raise GenerationError(
            f"temperature must be between 0.1 and 2.0, got {temperature}."
        )

    if not 0.0 <= float(top_p) <= 1.0:
        raise GenerationError(f"top_p must be between 0.0 and 1.0, got {top_p}.")

    if not 1.0 <= float(repetition_penalty) <= 2.0:
        raise GenerationError(
            f"repetition_penalty must be between 1.0 and 2.0, "
            f"got {repetition_penalty}."
        )

    if not 1 <= int(num_return_sequences) <= MAX_NUM_RETURN_SEQUENCES:
        raise GenerationError(
            f"num_return_sequences must be between 1 and "
            f"{MAX_NUM_RETURN_SEQUENCES}, got {num_return_sequences}."
        )


def validate_prompt(prompt: str) -> str:
    """Validate a user prompt and return its stripped form."""
    if prompt is None or not isinstance(prompt, str):
        raise GenerationError("Prompt must be a non-empty string.")
    cleaned = prompt.strip()
    if not cleaned:
        raise GenerationError(
            "Prompt is empty. Please enter some text before generating."
        )
    return cleaned


def generate_text(
    prompt: str,
    model_name: str = "gpt2",
    local_checkpoint: Path | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    num_return_sequences: int = DEFAULT_NUM_RETURN_SEQUENCES,
) -> list[str]:
    """Generate text continuations for ``prompt`` using a real GPT-2 model.

    The model is loaded once and cached, so repeated calls do not re-download
    or re-load hundreds of megabytes of weights.

    Returns
    -------
    list[str]
        Decoded completions (the prompt is included at the start of each).

    Raises
    ------
    GenerationError
        For invalid input or a failure to load/run the model.
    """
    cleaned_prompt = validate_prompt(prompt)
    validate_generation_settings(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
    )

    checkpoint_str = str(local_checkpoint) if local_checkpoint else None

    try:
        model, tokenizer, _resolved_id = _cached_model(model_name, checkpoint_str)
    except ModelError as exc:
        raise GenerationError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise GenerationError(
            f"Could not load the model: {exc}. Check your internet connection "
            "if the model has not been downloaded yet."
        ) from exc

    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        raise GenerationError(f"PyTorch is unavailable: {exc}") from exc

    device = get_device()

    try:
        model = model.to(device)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        encoded = tokenizer(
            cleaned_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                do_sample=True,
                num_return_sequences=int(num_return_sequences),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        return [tokenizer.decode(row, skip_special_tokens=True) for row in outputs]
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(f"Generation failed: {exc}") from exc


def clear_model_cache() -> None:
    """Drop any cached model instances (used by tests and manual reloads)."""
    _cached_model.cache_clear()
