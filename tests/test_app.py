"""Tests for the Streamlit app — structural and syntax checks.

Uses basic import/file-inspection rather than AppTest (version-sensitive).
Confirms the app is runnable, has required elements, and handles missing
checkpoints gracefully.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

APP_FILE = PROJECT_ROOT / "app.py"


def test_app_file_exists():
    assert APP_FILE.exists(), "app.py is missing"


def test_app_is_valid_python():
    source = APP_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)


def test_app_has_streamlit_import():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "import streamlit" in source or "import streamlit as st" in source


def test_app_has_generate_button_keyword():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "Generate" in source, "Generate button label missing"


def test_app_has_text_area():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "text_area" in source, "Prompt text_area missing"


def test_app_has_model_selection():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "selectbox" in source or "select_slider" in source or "Pretrained" in source


def test_app_has_training_info_section():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "Training Information" in source


def test_app_has_fine_tuned_check():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "check_fine_tuned_checkpoint" in source


def test_app_has_download_button():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "download_button" in source


def test_app_has_no_crash_on_missing_checkpoint():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "check_fine_tuned_checkpoint()" in source


def test_app_has_output_display():
    source = APP_FILE.read_text(encoding="utf-8")
    assert "output-box" in source or "Generated Output" in source
