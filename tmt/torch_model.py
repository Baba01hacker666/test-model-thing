"""Optional PyTorch implementation and interoperability for Test-Model-Thing (TMT).

PyTorch is completely optional. If torch is not installed, this module provides
graceful fallbacks and informative errors without affecting the core pure-NumPy engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False


def is_torch_available() -> bool:
    """Return True if PyTorch is installed and importable."""
    return HAS_TORCH


if HAS_TORCH:

    class TorchEncoder(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.embed = nn.Embedding(256, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.embed(x)

    class TorchDecoder(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.decode = nn.Linear(dim, 256)
            self.stop = nn.Linear(dim, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.decode(x), torch.sigmoid(self.stop(x))

    class TorchLayer(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.decay = nn.Parameter(torch.zeros(dim))
            self.norm = nn.LayerNorm(dim)
            self.weights = nn.Linear(dim, dim, bias=False)
            self.silu = nn.SiLU()

            # Persistent recurrent states (buffers, not parameters)
            self.register_buffer("states", torch.zeros(dim))
            self.register_buffer("decaytrace", torch.zeros(dim))
            self.register_buffer("embedtrace", torch.zeros(256, dim))

        def forward(
            self, enc: torch.Tensor, x: torch.Tensor, dummy: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            decay = torch.sigmoid(self.decay)
            d = dummy if dummy is not None else torch.zeros(self.dim, device=x.device)
            state = (decay * self.states) + enc + d
            act = self.silu(self.weights(self.norm(state)))
            return x + act, state, decay

    class TorchModel(nn.Module):
        """PyTorch equivalent of Test-Model-Thing architecture."""

        def __init__(self, dim: int = 512, layers: int = 16, temp: float = 0.75, lr: float = 5e-4):
            super().__init__()
            self.dim = dim
            self.layercount = layers
            self.temp = temp
            self.lr = lr

            self.encoder = TorchEncoder(dim)
            self.decoder = TorchDecoder(dim)
            self.layers = nn.ModuleList([TorchLayer(dim) for _ in range(layers)])

        def reset(self) -> None:
            for layer in self.layers:
                layer.states.zero_()
                layer.decaytrace.zero_()
                layer.embedtrace.zero_()

        def forward_byte(self, currb: int) -> tuple[torch.Tensor, torch.Tensor]:
            c = torch.tensor(currb, dtype=torch.long, device=next(self.parameters()).device)
            enc = self.encoder(c)
            x = enc.clone()
            for layer in self.layers:
                x, state, _ = layer(enc, x)
                layer.states.copy_(state.detach())
            logits, stop = self.decoder(x)
            return logits, stop

        def to_numpy_state_dict(self) -> dict[str, np.ndarray]:
            """Convert PyTorch weights into TMT NumPy state dict."""
            res: dict[str, np.ndarray] = {
                "m.encoder.embed.weight": self.encoder.embed.weight.detach().cpu().numpy(),
                "m.decoder.decode.weight": self.decoder.decode.weight.detach().cpu().numpy(),
                "m.decoder.decode.bias": self.decoder.decode.bias.detach().cpu().numpy(),
                "m.decoder.stop.weight": self.decoder.stop.weight.detach().cpu().numpy(),
                "m.decoder.stop.bias": self.decoder.stop.bias.detach().cpu().numpy(),
            }
            for i, layer in enumerate(self.layers):
                res[f"m.layers.{i}.decay"] = layer.decay.detach().cpu().numpy()
                res[f"m.layers.{i}.norm.weight"] = layer.norm.weight.detach().cpu().numpy()
                res[f"m.layers.{i}.norm.bias"] = layer.norm.bias.detach().cpu().numpy()
                res[f"m.layers.{i}.weights.weight"] = layer.weights.weight.detach().cpu().numpy()
                res[f"state.{i}"] = layer.states.detach().cpu().numpy()
                res[f"decaytrace.{i}"] = layer.decaytrace.detach().cpu().numpy()
                res[f"embedtrace.{i}"] = layer.embedtrace.detach().cpu().numpy()
            return res

        def load_numpy_state_dict(self, state: dict[str, np.ndarray]) -> None:
            """Load weights from a TMT NumPy state dict."""
            with torch.no_grad():
                if "m.encoder.embed.weight" in state:
                    self.encoder.embed.weight.copy_(torch.from_numpy(state["m.encoder.embed.weight"]))
                if "m.decoder.decode.weight" in state:
                    self.decoder.decode.weight.copy_(torch.from_numpy(state["m.decoder.decode.weight"]))
                if "m.decoder.decode.bias" in state:
                    self.decoder.decode.bias.copy_(torch.from_numpy(state["m.decoder.decode.bias"]))
                if "m.decoder.stop.weight" in state:
                    self.decoder.stop.weight.copy_(torch.from_numpy(state["m.decoder.stop.weight"]))
                if "m.decoder.stop.bias" in state:
                    self.decoder.stop.bias.copy_(torch.from_numpy(state["m.decoder.stop.bias"]))
                for i, layer in enumerate(self.layers):
                    if f"m.layers.{i}.decay" in state:
                        layer.decay.copy_(torch.from_numpy(state[f"m.layers.{i}.decay"]))
                    if f"m.layers.{i}.norm.weight" in state:
                        layer.norm.weight.copy_(torch.from_numpy(state[f"m.layers.{i}.norm.weight"]))
                    if f"m.layers.{i}.norm.bias" in state:
                        layer.norm.bias.copy_(torch.from_numpy(state[f"m.layers.{i}.norm.bias"]))
                    if f"m.layers.{i}.weights.weight" in state:
                        layer.weights.weight.copy_(torch.from_numpy(state[f"m.layers.{i}.weights.weight"]))
                    if f"state.{i}" in state:
                        layer.states.copy_(torch.from_numpy(state[f"state.{i}"]))

else:

    class TorchModel:  # type: ignore[no-redef]
        """Stub when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is not installed. To use TorchModel, install PyTorch: "
                "pip install 'test-model-thing[torch]' or pip install torch"
            )
