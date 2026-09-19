"""Pytest fixtures for Test-Model-Thing (TMT)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tmt.data import get_sample_cola_path, get_sample_text_path
from tmt.model import Model


@pytest.fixture
def small_model() -> Model:
    """Create a lightweight model for fast testing."""
    return Model(dim=32, layers=2, temp=0.75, lr=1e-3, seed=42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_text_file() -> str:
    """Path to verified bundled sample text file."""
    p = get_sample_text_path()
    assert Path(p).exists()
    return p


@pytest.fixture
def sample_cola_file() -> str:
    """Path to verified bundled sample CoLA TSV file."""
    p = get_sample_cola_path()
    assert Path(p).exists()
    return p
