"""Tests for text generation logic with mocks to avoid downloads."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.textforge.generation import (
    generate_text,
    GenerationError,
    _cached_model,
)
from src.textforge.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_NUM_RETURN_SEQUENCES,
    MAX_MAX_NEW_TOKENS,
    MAX_NUM_RETURN_SEQUENCES,
)
from src.textforge.model_manager import ModelError

def test_empty_prompt_rejected():
    with pytest.raises(GenerationError):
        generate_text("")

def test_whitespace_prompt_rejected():
    with pytest.raises(GenerationError):
        generate_text("   \n\t  ")

def test_invalid_max_new_tokens_too_low():
    with pytest.raises(GenerationError):
        generate_text("prompt", max_new_tokens=0)

def test_invalid_max_new_tokens_too_high():
    with pytest.raises(GenerationError):
        generate_text("prompt", max_new_tokens=MAX_MAX_NEW_TOKENS + 1)

def test_invalid_temperature_low():
    with pytest.raises(GenerationError):
        generate_text("prompt", temperature=0.0)

def test_invalid_temperature_high():
    with pytest.raises(GenerationError):
        generate_text("prompt", temperature=2.1)

def test_invalid_top_p_low():
    with pytest.raises(GenerationError):
        generate_text("prompt", top_p=-0.1)

def test_invalid_top_p_high():
    with pytest.raises(GenerationError):
        generate_text("prompt", top_p=1.1)

def test_invalid_repetition_penalty_low():
    with pytest.raises(GenerationError):
        generate_text("prompt", repetition_penalty=0.9)

def test_invalid_repetition_penalty_high():
    with pytest.raises(GenerationError):
        generate_text("prompt", repetition_penalty=2.1)

def test_invalid_num_outputs_low():
    with pytest.raises(GenerationError):
        generate_text("prompt", num_return_sequences=0)

def test_invalid_num_outputs_high():
    with pytest.raises(GenerationError):
        generate_text("prompt", num_return_sequences=MAX_NUM_RETURN_SEQUENCES + 1)

def test_model_loading_failure_handled():
    with patch("src.textforge.generation.load_model") as mock_load:
        mock_load.side_effect = ModelError("Model not found")
        with pytest.raises(GenerationError, match="Model not found"):
            generate_text("prompt")

def test_generation_calls_model_generate():
    mock_tokenizer = Mock()
    mock_tokenizer.pad_token = "<|pad|>"
    mock_tokenizer.eos_token = "<|eos|>"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.decode = Mock(return_value="generated text")
    mock_tokenizer.return_value = {
        "input_ids": Mock(to=Mock(return_value=Mock())),
        "attention_mask": Mock(to=Mock(return_value=Mock())),
    }

    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.generate = Mock(return_value=[[1, 2, 3]])

    with patch("src.textforge.generation.load_model", return_value=(mock_model, mock_tokenizer, "gpt2")):
        with patch("src.textforge.generation.get_device", return_value="cpu"):
            # Clear cache
            from src.textforge.generation import clear_model_cache
            clear_model_cache()
            results = generate_text("test prompt", num_return_sequences=2)
            assert isinstance(results, list)
            assert mock_model.generate.called
            # Check generation was called with our params
            call_kwargs = mock_model.generate.call_args.kwargs
            assert "max_new_tokens" in call_kwargs
            assert "temperature" in call_kwargs
            assert "top_p" in call_kwargs
            assert "repetition_penalty" in call_kwargs
            assert "num_return_sequences" in call_kwargs

def test_missing_fine_tuned_reported(tmp_path, monkeypatch):
    from src.textforge import config

    # Isolate this test from any real checkpoint in the project.
    missing_checkpoint = tmp_path / "missing-checkpoint"
    monkeypatch.setattr(config, "DEFAULT_TRAINING_OUTPUT", missing_checkpoint)

    assert config.check_fine_tuned_checkpoint() is None

def test_base_model_not_mislabeled():
    from src.textforge.config import DEFAULT_MODEL_NAME
    assert DEFAULT_MODEL_NAME == "gpt2"
