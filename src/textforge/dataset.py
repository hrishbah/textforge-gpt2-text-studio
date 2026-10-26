"""Dataset loading, cleaning and chunking for TextForge.

Handles plain-text training corpora: reads a ``.txt`` file, validates that it
contains enough content, normalises whitespace, and converts the text into
fixed-size token blocks suitable for causal language modelling with GPT-2.

Two chunking strategies are provided:

``chunk_texts``
    Character-level blocks. Lightweight and used for quick inspection.

``build_token_blocks``
    The real training path. The whole corpus is tokenised once and then split
    into contiguous blocks of exactly ``block_size`` tokens, which guarantees
    uniform batches (required by the training collator).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import MIN_DATASET_CHARS, MIN_DATASET_LINES


class DatasetError(Exception):
    """Raised when a dataset is missing, empty or otherwise invalid."""


@dataclass
class TextChunk:
    """A single training chunk: the tokenised input and its causal labels."""

    text: str
    input_ids: list[int]
    labels: list[int]
    source_text: str


def load_dataset(path) -> tuple[str, list[str]]:
    """Load and validate a plain-text dataset.

    Parameters
    ----------
    path : str | Path
        Path to a UTF-8 ``.txt`` file.

    Returns
    -------
    tuple[str, list[str]]
        The normalised text and the list of paragraph chunks.

    Raises
    ------
    DatasetError
        If the file is missing, empty, whitespace-only or too small.
    """
    if path is None:
        raise DatasetError("No dataset path provided.")

    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    if not p.is_file():
        raise DatasetError(f"Dataset path is not a file: {path}")

    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(
            f"Dataset file could not be decoded as UTF-8 text: {path}"
        ) from exc

    # Normalise line endings and drop blank/whitespace-only lines.
    lines = [line.strip() for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]

    if len(lines) < MIN_DATASET_LINES:
        raise DatasetError(
            f"Dataset is too small: found {len(lines)} usable line(s), "
            f"but at least {MIN_DATASET_LINES} are required."
        )

    content = "\n".join(lines)
    if len(content) < MIN_DATASET_CHARS:
        raise DatasetError(
            f"Dataset content is too short: {len(content)} characters, "
            f"minimum required is {MIN_DATASET_CHARS}."
        )

    return content, _split_paragraphs(content, lines)


def _split_paragraphs(content: str, lines: list[str]) -> list[str]:
    """Split content into paragraphs.

    Blank-line separated paragraphs are preferred. If the file contains no
    blank lines (a common hand-written corpus layout), each non-empty line is
    treated as its own paragraph so that chunking never cuts across unrelated
    sentences.
    """
    paragraphs = [block.strip() for block in content.split("\n\n") if block.strip()]

    if len(paragraphs) <= 1 and len(lines) > 1:
        # No blank-line separation present -> treat each line as a paragraph.
        paragraphs = list(lines)

    return paragraphs


def chunk_texts(
    paragraphs: list[str],
    block_size: int = 128,
    overlap: int = 0,
) -> list[TextChunk]:
    """Split paragraphs into fixed-size character blocks.

    Parameters
    ----------
    paragraphs : list[str]
        Paragraphs to chunk.
    block_size : int
        Number of characters per block (must be positive).
    overlap : int
        Overlap in characters between consecutive blocks.

    Returns
    -------
    list[TextChunk]
        Chunks with empty token ids (character-level preview only).
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    chunks: list[TextChunk] = []
    overlap_chars = max(0, min(overlap, block_size))

    for paragraph in paragraphs:
        text = paragraph.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + block_size
            chunk_text = text[start:end]
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    input_ids=[],
                    labels=[],
                    source_text=chunk_text,
                )
            )
            start = end - overlap_chars if overlap_chars else end

    return chunks


def build_token_blocks(
    content: str,
    tokenizer,
    block_size: int = 128,
) -> list[dict]:
    """Tokenise ``content`` and split it into fixed-size token blocks.

    This is the real training path. Every returned example has exactly
    ``block_size`` tokens, so batches are rectangular and can be collated
    without ragged tensors.

    Parameters
    ----------
    content : str
        Normalised dataset text.
    tokenizer :
        A Hugging Face tokenizer (e.g. ``GPT2Tokenizer``).
    block_size : int
        Number of tokens per training example.

    Returns
    -------
    list[dict]
        Each item is ``{"input_ids": [...], "labels": [...]}``.

    Raises
    ------
    DatasetError
        If the content tokenises to fewer tokens than one block.
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if not content.strip():
        raise DatasetError("Cannot build token blocks from empty content.")

    encoded = tokenizer(content, return_tensors="pt")
    input_ids = encoded["input_ids"][0]

    total_tokens = int(input_ids.shape[0])
    if total_tokens < block_size:
        raise DatasetError(
            f"Dataset tokenises to only {total_tokens} tokens, which is fewer "
            f"than a single block of {block_size}. Add more text or lower "
            f"--block-size."
        )

    num_blocks = total_tokens // block_size
    usable = num_blocks * block_size

    examples: list[dict] = []
    for start in range(0, usable, block_size):
        block = input_ids[start : start + block_size].tolist()
        examples.append(
            {
                "input_ids": block,
                # Causal LM: the model predicts the next token from each position.
                "labels": block.copy(),
            }
        )

    return examples
