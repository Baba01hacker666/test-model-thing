"""Unit tests for tmt.benchmark."""

from __future__ import annotations

import numpy as np

from tmt.benchmark import ClassificationHead, mcc, rollout, run_benchmark
from tmt.model import Model


def test_mcc_metric_cases():
    # Perfect score: TP and TN high, no errors
    assert mcc(10, 10, 0, 0) == 100.0

    # Perfect inverse: FP and FN only
    assert mcc(0, 0, 10, 10) == -100.0

    # Zero denominator edge case
    assert mcc(0, 0, 0, 0) == 0.0

    # Balanced mixed predictions
    score = mcc(10, 10, 5, 5)
    assert -100.0 <= score <= 100.0
    assert score > 0.0


def test_classification_head():
    dim = 32
    head = ClassificationHead(dim=dim, lr=1e-2, seed=42)
    x = np.random.randn(dim).astype(np.float32)

    logits = head(x)
    assert logits.shape == (2,)

    # Train step reduces loss on repeated optimization
    loss1, _ = head.train_step(x, target=1)
    for _ in range(5):
        loss_final, pred = head.train_step(x, target=1)
    assert loss_final < loss1
    assert pred in (0, 1)


def test_rollout(small_model: Model):
    b_data = b"Hello, world!"
    final_state = rollout(small_model, b_data)

    assert final_state is not None
    assert final_state.shape == (small_model.dim,)
    assert not np.isnan(final_state).any()


def test_rollout_empty_bytes(small_model: Model):
    final_state = rollout(small_model, b"")
    assert final_state is None


def test_run_benchmark_end_to_end(small_model: Model, sample_cola_file: str, temp_dir):
    ckpt_path = str(temp_dir / "backbone.safetensors")
    small_model.save(ckpt_path)

    results = run_benchmark(
        checkpoint_path=ckpt_path,
        data_path=sample_cola_file,
        epochs=2,
        dev=0.2,
        lr=1e-2,
        dim=small_model.dim,
        layers=small_model.layercount,
        verbose=False,
    )

    assert "final_train_mcc" in results
    assert "final_dev_mcc" in results
    assert len(results["history"]) == 2
    assert results["epochs"] == 2
