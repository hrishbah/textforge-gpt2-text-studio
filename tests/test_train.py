"""Safety tests for the GPT-2 training entry point."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import main


def test_training_refuses_nonempty_output_directory(tmp_path, capsys):
    dataset = PROJECT_ROOT / "data" / "train.txt"
    output_dir = tmp_path / "existing-checkpoint"
    output_dir.mkdir()
    sentinel = output_dir / "keep-me.txt"
    sentinel.write_text("preserve existing content", encoding="utf-8")

    result = main([
        "--data-path", str(dataset),
        "--output-dir", str(output_dir),
        "--max-steps", "2",
        "--device", "cpu",
    ])

    captured = capsys.readouterr()
    assert result == 1
    assert "output directory is not empty" in captured.err.lower()
    assert sentinel.read_text(encoding="utf-8") == "preserve existing content"
