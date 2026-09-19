"""Benchmark suite and probe evaluation for Test-Model-Thing."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from tmt.data import create_train_dev_split, get_sample_cola_path, load_cola
from tmt.model import AdamW, Model


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Calculate Matthews Correlation Coefficient (MCC) scaled to [-100.0, 100.0]."""
    numerator = float(tp * tn - fp * fn)
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0


class ClassificationHead:
    """Linear classification probe on top of frozen backbone representations."""

    def __init__(self, dim: int, lr: float = 1e-3, seed: int | None = 42):
        self.dim = dim
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(dim)
        self.weight = (rng.standard_normal((2, dim)) * scale).astype(np.float32)
        self.bias = np.zeros(2, dtype=np.float32)
        self.optimizer = AdamW(learning_rate=lr)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.weight.T + self.bias).astype(np.float32)

    def train_step(self, x: np.ndarray, target: int) -> tuple[float, int]:
        logits = self(x)
        shift = logits - np.max(logits)
        exps = np.exp(shift)
        probs = exps / np.sum(exps)

        loss = -float(np.log(max(probs[target], 1e-12)))

        dlogits = probs.copy()
        dlogits[target] -= 1.0

        grads = {
            "weight": np.outer(dlogits, x),
            "bias": dlogits,
        }
        params = {
            "weight": self.weight,
            "bias": self.bias,
        }
        self.optimizer.update(params, grads)
        pred = int(np.argmax(logits))
        return loss, pred


def rollout(model: Model, b_s: bytes, dummies: list[np.ndarray] | None = None) -> np.ndarray | None:
    """Run model state forward on byte sequence and return the final hidden state."""
    model.reset()
    final: np.ndarray | None = None

    for b in b_s:
        enc = model.embed[b]
        x = enc.copy()

        for j in range(model.layercount):
            dummy = dummies[j] if dummies is not None else np.zeros(model.dim, dtype=np.float32)
            decay = model.layer_decay[j]
            sig_decay = (1.0 / (1.0 + np.exp(-decay))).astype(np.float32)
            state = (sig_decay * model.layers[j].states) + enc + dummy

            mean = np.mean(state)
            var = np.var(state)
            norm_s = (state - mean) / np.sqrt(var + 1e-5) * model.layer_norm_w[j] + model.layer_norm_b[j]

            lin = norm_s @ model.layer_w[j].T
            sig_lin = 1.0 / (1.0 + np.exp(-lin))
            x = x + (lin * sig_lin)

            model.layers[j].states = state.astype(np.float32)

        final = model.layers[-1].states

    return final


def run_benchmark(
    checkpoint_path: str,
    data_path: str | None = None,
    epochs: int = 3,
    dev: float = 0.1,
    lr: float = 1e-3,
    dim: int = 512,
    layers: int = 16,
    log_interval: int = 500,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train linear classification probe on frozen backbone representation."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path!r}. Train a model first "
            f"(tmt train) or pass --path to an existing checkpoint."
        )

    model = Model(dim=dim, layers=layers)
    model.load(checkpoint_path)
    model.freeze()

    resolved_data = data_path or get_sample_cola_path()
    rows = load_cola(resolved_data)
    if not rows:
        raise ValueError(
            f"Invalid or empty CoLA dataset at {resolved_data!r}. "
            "Pass a valid CoLA TSV path or omit --data to use bundled sample CoLA data."
        )

    train_rows, heldout_rows = create_train_dev_split(rows, dev_fraction=dev)

    head = ClassificationHead(dim=model.dim, lr=lr)
    dummies = [np.zeros(model.dim, dtype=np.float32) for _ in range(model.layercount)]

    results: list[dict[str, Any]] = []

    for epoch in range(epochs):
        tp, tn, fp, fn = 0, 0, 0, 0
        score = 0.0

        for i, (b_s, label) in enumerate(train_rows):
            if not b_s:
                continue
            final_state = rollout(model, b_s, dummies)
            if final_state is None:
                continue

            _, pred = head.train_step(final_state, label)
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 0 and label == 0:
                tn += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1

            score = mcc(tp, tn, fp, fn)
            if verbose and i > 0 and i % log_interval == 0:
                print(f"[{i}/{len(train_rows)}] T+ {tp}, T- {tn}, F+ {fp}, F- {fn} (MCC: {score:.2f})")

        train_mcc = mcc(tp, tn, fp, fn)
        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} Train ({len(train_rows)}): T+ {tp}, T- {tn}, F+ {fp}, F- {fn} (MCC: {train_mcc:.2f})"
            )

        # Dev evaluation (frozen head)
        dtp, dtn, dfp, dfn = 0, 0, 0, 0
        for b_s, label in heldout_rows:
            if not b_s:
                continue
            final_state = rollout(model, b_s, dummies)
            if final_state is None:
                continue
            logits = head(final_state)
            pred = int(np.argmax(logits))
            if pred == 1 and label == 1:
                dtp += 1
            elif pred == 0 and label == 0:
                dtn += 1
            elif pred == 1 and label == 0:
                dfp += 1
            elif pred == 0 and label == 1:
                dfn += 1

        dev_mcc = mcc(dtp, dtn, dfp, dfn)
        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} Dev   ({len(heldout_rows)}): T+ {dtp}, T- {dtn}, F+ {dfp}, F- {dfn} (MCC: {dev_mcc:.2f})"
            )

        results.append(
            {
                "epoch": epoch + 1,
                "train_mcc": train_mcc,
                "dev_mcc": dev_mcc,
                "train_counts": (tp, tn, fp, fn),
                "dev_counts": (dtp, dtn, dfp, dfn),
            }
        )

    return {
        "checkpoint": checkpoint_path,
        "data": resolved_data,
        "epochs": epochs,
        "final_train_mcc": results[-1]["train_mcc"] if results else 0.0,
        "final_dev_mcc": results[-1]["dev_mcc"] if results else 0.0,
        "history": results,
    }
