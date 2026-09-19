"""Tests for optional PyTorch integration and fallback behavior."""

from __future__ import annotations

import pytest

from tmt.cli import main
from tmt.torch_model import TorchModel, is_torch_available


def test_is_torch_available():
    res = is_torch_available()
    assert isinstance(res, bool)


def test_torch_model_fallback():
    if not is_torch_available():
        with pytest.raises(ImportError, match="PyTorch is not installed"):
            TorchModel()
    else:
        model = TorchModel(dim=32, layers=2)
        assert model is not None


def test_cli_info_shows_torch_status(capsys):
    main(["info"])
    captured = capsys.readouterr()
    assert "Optional PyTorch" in captured.out
