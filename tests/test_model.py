"""Unit tests for tmt.model."""

from __future__ import annotations

import numpy as np

from tmt.model import AdamW, Model, count_params


def test_count_params():
    # Canonical 4.5M configuration: dim=512, layers=16
    assert count_params(512, 16) == 4481793
    # Small configurations
    assert count_params(64, 2) == 41665
    assert count_params(32, 2) == 18913


def test_model_initialization(small_model: Model):
    dim = small_model.dim
    layers = small_model.layercount

    assert small_model.embed.shape == (256, dim)
    assert small_model.dec_w.shape == (256, dim)
    assert small_model.dec_b.shape == (256,)
    assert small_model.stop_w.shape == (1, dim)
    assert small_model.stop_b.shape == (1,)
    assert len(small_model.layers) == layers

    for layer in small_model.layers:
        assert layer.states.shape == (dim,)
        assert layer.decaytrace.shape == (dim,)
        assert layer.embedtrace.shape == (256, dim)


def test_model_reset(small_model: Model):
    # Set dummy state
    small_model.layers[0].states.fill(1.5)
    small_model.layers[0].decaytrace.fill(0.7)
    small_model.layers[0].embedtrace.fill(0.3)

    small_model.reset()

    for layer in small_model.layers:
        assert np.all(layer.states == 0.0)
        assert np.all(layer.decaytrace == 0.0)
        assert np.all(layer.embedtrace == 0.0)


def test_model_step(small_model: Model):
    (x, states, decays), (logits, stop_val) = small_model.step(65)

    assert x.shape == (small_model.dim,)
    assert len(states) == small_model.layercount
    assert len(decays) == small_model.layercount
    assert logits.shape == (256,)
    assert 0.0 <= stop_val <= 1.0


def test_model_sample(small_model: Model):
    logits = np.random.randn(256).astype(np.float32)
    b = small_model.sample(logits)
    assert isinstance(b, int)
    assert 0 <= b < 256


def test_model_training_step(small_model: Model):
    orig_embed = small_model.embed.copy()
    b_out, stop_prob = small_model(65, 66, end=False)

    assert 0 <= b_out < 256
    assert 0.0 <= stop_prob <= 1.0
    # Weights should have updated via AdamW
    assert not np.allclose(small_model.embed, orig_embed)
    assert small_model.optimizer.step == 1


def test_model_frozen_mode(small_model: Model):
    orig_embed = small_model.embed.copy()
    step_before = small_model.optimizer.step

    b_out, stop_prob = small_model(65, 66, end=False, frozen=True)

    # Weights and optimizer step should NOT change
    assert np.allclose(small_model.embed, orig_embed)
    assert small_model.optimizer.step == step_before
    # But layer states SHOULD be updated
    assert not np.all(small_model.layers[0].states == 0.0)


def test_model_notrace_mode(small_model: Model):
    small_model.reset()
    b_out, stop_prob = small_model(65, None, end=False, notrace=True)

    assert 0 <= b_out < 256
    assert 0.0 <= stop_prob <= 1.0
    # States should remain zero in notrace mode
    for layer in small_model.layers:
        assert np.all(layer.states == 0.0)


def test_model_save_load_safetensors(small_model: Model, temp_dir):
    ckpt_path = str(temp_dir / "model.safetensors")
    small_model(65, 66, False)  # advance state & train step
    small_model.save(ckpt_path)

    new_model = Model(dim=small_model.dim, layers=small_model.layercount)
    assert new_model.load(ckpt_path)

    assert np.allclose(new_model.embed, small_model.embed)
    assert np.allclose(new_model.dec_w, small_model.dec_w)
    assert np.allclose(new_model.layers[0].states, small_model.layers[0].states)
    assert new_model.optimizer.step == small_model.optimizer.step


def test_model_save_load_npz(small_model: Model, temp_dir):
    ckpt_path = str(temp_dir / "model.npz")
    small_model(65, 66, False)
    small_model.save(ckpt_path)

    new_model = Model(dim=small_model.dim, layers=small_model.layercount)
    assert new_model.load(ckpt_path)

    assert np.allclose(new_model.embed, small_model.embed)
    assert np.allclose(new_model.layers[0].states, small_model.layers[0].states)
    assert new_model.optimizer.step == small_model.optimizer.step


def test_adamw_optimizer():
    opt = AdamW(learning_rate=1e-2, weight_decay=0.01)
    params = {"w": np.ones((4, 4), dtype=np.float32)}
    grads = {"w": np.full((4, 4), 0.5, dtype=np.float32)}

    opt.update(params, grads)
    assert opt.step == 1
    assert not np.allclose(params["w"], 1.0)
    assert params["w"].mean() < 1.0
