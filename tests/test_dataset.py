"""Tests for the dataset loading and validation module."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.textforge.dataset import (
    load_dataset,
    chunk_texts,
    build_token_blocks,
    TextChunk,
    DatasetError,
)
from src.textforge.config import MIN_DATASET_CHARS, MIN_DATASET_LINES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VALID_DATA = DATA_DIR / "train.txt"

def test_valid_dataset_loads():
    content, paragraphs = load_dataset(VALID_DATA)
    assert isinstance(content, str)
    assert len(content) > 0
    assert isinstance(paragraphs, list)
    assert len(paragraphs) > 0
    assert all(isinstance(p, str) for p in paragraphs)

def test_missing_path_raises():
    with pytest.raises(DatasetError):
        load_dataset(Path("/nonexistent/path/to/file.txt"))

def test_empty_dataset_rejected(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(DatasetError):
        load_dataset(p)

def test_whitespace_only_rejected(tmp_path):
    p = tmp_path / "ws.txt"
    p.write_text("   \n\n   \t\t\n")
    with pytest.raises(DatasetError):
        load_dataset(p)

def test_short_dataset_rejected(tmp_path):
    p = tmp_path / "short.txt"
    p.write_text("hello\nworld\n")
    with pytest.raises(DatasetError):
        load_dataset(p)

def test_chunk_texts_basic():
    paragraphs = ["This is a test paragraph with enough words."]
    chunks = chunk_texts(paragraphs, block_size=128)
    assert len(chunks) > 0
    assert all(isinstance(c, TextChunk) for c in chunks)

def test_chunk_texts_invalid_block_size():
    with pytest.raises(ValueError):
        chunk_texts(["test"], block_size=0)

def test_build_token_blocks_fixed_size():
    # Use a deterministic mock tokenizer so this test never requires
    # an internet download.
    class MockTokenizer:
        def __call__(self, content, return_tensors="pt", **kwargs):
            # Deterministic 134-token sequence (enough for 4 blocks of 32)
            import torch
            return {"input_ids": torch.tensor([list(range(1, 135))])}

    text = "Hello world. " * 20
    blocks = build_token_blocks(text, MockTokenizer(), block_size=32)
    assert len(blocks) > 0
    for b in blocks:
        assert len(b["input_ids"]) == 32
        assert len(b["labels"]) == 32
        assert b["labels"] == b["input_ids"]
