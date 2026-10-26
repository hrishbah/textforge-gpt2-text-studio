"""TextForge — GPT-2 Text Studio.

A small, dependency-light package that powers the TextForge Streamlit app:

* :mod:`textforge.config`        — paths, defaults and validation constants.
* :mod:`textforge.dataset`       — dataset loading, cleaning and chunking.
* :mod:`textforge.model_manager` — model loading and caching.
* :mod:`textforge.generation`    — prompt validation and text generation.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
