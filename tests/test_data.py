"""Unit tests for tmt.data."""

from __future__ import annotations

from pathlib import Path

import pytest

from tmt.data import (
    create_train_dev_split,
    generate_synthetic_corpus,
    get_sample_cola_path,
    get_sample_text_path,
    load_cola,
    load_text_lines,
)


def test_bundled_sample_paths():
    text_path = get_sample_text_path()
    cola_path = get_sample_cola_path()

    assert Path(text_path).is_file()
    assert Path(cola_path).is_file()


def test_load_text_lines(sample_text_file: str):
    lines = list(load_text_lines(sample_text_file, min_bytes=2))
    assert len(lines) > 0
    for line in lines:
        assert len(line.encode("utf-8")) >= 2


def test_load_text_lines_not_found():
    with pytest.raises(FileNotFoundError):
        list(load_text_lines("non_existent_folder/**/*.xyz"))


def test_load_cola(sample_cola_file: str):
    rows = load_cola(sample_cola_file)
    assert len(rows) >= 10
    for b_s, label in rows:
        assert isinstance(b_s, bytes)
        assert label in (0, 1)
        assert len(b_s) > 0


def test_load_cola_missing_file():
    rows = load_cola("non_existent_file.tsv")
    assert rows == []


def test_create_train_dev_split():
    items = list(range(100))
    train, dev = create_train_dev_split(items, dev_fraction=0.1)

    assert len(train) == 90
    assert len(dev) == 10
    assert train == list(range(90))
    assert dev == list(range(90, 100))


def test_create_train_dev_split_validation():
    with pytest.raises(ValueError, match="dev_fraction must be in"):
        create_train_dev_split([1, 2, 3], dev_fraction=1.5)

    with pytest.raises(ValueError, match="Need at least 2 items"):
        create_train_dev_split([1], dev_fraction=0.1)


def test_generate_synthetic_corpus():
    corpus1 = generate_synthetic_corpus(num_lines=20, seed=123)
    corpus2 = generate_synthetic_corpus(num_lines=20, seed=123)

    assert len(corpus1) == 20
    assert corpus1 == corpus2  # Deterministic with seed
    assert all(line.endswith("\n") for line in corpus1)
