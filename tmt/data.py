"""Dataset loaders, preprocessors, and sample data generators."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")

DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
DEFAULT_SAMPLE_TEXT = DEFAULT_SAMPLE_DIR / "sample_text.txt"
DEFAULT_SAMPLE_COLA = DEFAULT_SAMPLE_DIR / "sample_cola.tsv"


def get_sample_text_path() -> str:
    """Return absolute path to default bundled sample text dataset."""
    return str(DEFAULT_SAMPLE_TEXT)


def get_sample_cola_path() -> str:
    """Return absolute path to default bundled sample CoLA dataset."""
    return str(DEFAULT_SAMPLE_COLA)


def load_text_lines(
    path_or_pattern: str,
    min_bytes: int = 2,
    recursive: bool = True,
) -> Iterator[str]:
    """Yield non-empty lines from files matching a path or glob pattern.

    Raises:
        FileNotFoundError: If no files match the path or pattern.
    """
    files = sorted(glob.glob(path_or_pattern, recursive=recursive))
    if not files:
        if os.path.isfile(path_or_pattern):
            files = [path_or_pattern]
        else:
            raise FileNotFoundError(
                f"No files matched path or pattern {path_or_pattern!r}. "
                "Check the file path or pass --sample to use bundled sample data."
            )

    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(line.encode("utf-8")) >= min_bytes:
                    yield line


def load_cola(filepath: str) -> list[tuple[bytes, int]]:
    """Load CoLA benchmark dataset TSV.

    Expected format: 4 tab-separated columns:
        [0] source / code
        [1] label (0 = unacceptable, 1 = acceptable)
        [2] author judgment / annotation notes
        [3] sentence

    Returns:
        List of (sentence_bytes, label_int).
    """
    if not os.path.exists(filepath):
        return []

    data: list[tuple[bytes, int]] = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                sentence = parts[3].strip()
                if not sentence:
                    continue
                try:
                    label = int(parts[1])
                    data.append((sentence.encode("utf-8"), label))
                except ValueError:
                    continue
    return data


def create_train_dev_split(
    items: Sequence[T],
    dev_fraction: float = 0.1,
) -> tuple[list[T], list[T]]:
    """Split items into train and dev (heldout) slices.

    Raises:
        ValueError: If dev_fraction is not in [0.0, 1.0) or leaves unusable splits.
    """
    if not 0.0 <= dev_fraction < 1.0:
        raise ValueError(f"dev_fraction must be in [0.0, 1.0), got {dev_fraction!r}.")
    if len(items) < 2:
        raise ValueError(f"Need at least 2 items to split, got {len(items)}.")

    split_idx = int(len(items) * (1.0 - dev_fraction))
    if not 1 <= split_idx < len(items):
        raise ValueError(f"dev_fraction={dev_fraction!r} leaves no usable split for {len(items)} items.")

    return list(items[:split_idx]), list(items[split_idx:])


def generate_synthetic_corpus(num_lines: int = 100, seed: int = 42) -> list[str]:
    """Generate deterministic synthetic text lines for testing."""
    import numpy as np

    rng = np.random.default_rng(seed)
    nouns = ["model", "network", "dataset", "system", "layer", "trace", "optimizer", "token"]
    verbs = ["trains", "predicts", "adapts", "updates", "evaluates", "encodes", "processes"]
    adjectives = ["recurrent", "latent", "streaming", "continuous", "modular", "frozen", "fast"]
    templates = [
        "The {adj} {noun} {verb} smoothly.",
        "Every {noun} requires careful validation and {adj} tuning.",
        "In {adj} mode, the {noun} {verb} the input bytes directly.",
        "Testing {noun} behavior verifies stability under stress.",
    ]

    lines = []
    for _ in range(num_lines):
        tpl = str(rng.choice(templates))
        line = tpl.format(
            noun=str(rng.choice(nouns)),
            verb=str(rng.choice(verbs)),
            adj=str(rng.choice(adjectives)),
        )
        lines.append(line + "\n")
    return lines
