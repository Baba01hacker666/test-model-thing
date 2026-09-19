# MLX to Pure NumPy Migration Notes

## Why Pure NumPy?

The initial proof-of-concept for Test-Model-Thing was developed with Apple MLX (`mlx.core`, `mlx.nn`). While MLX is highly optimized for Apple Silicon Unified Memory, it introduced substantial deployment constraints:
- **Platform Lock-in**: MLX primarily targets macOS / Apple Silicon. Linux binary wheels often lack the compiled C++ backend (`libmlx.so`), preventing execution on Linux servers, cloud instances, and Termux/Android environments.
- **Dependencies**: Requiring specialized hardware frameworks creates high friction for contributors and testing suites.

By porting the core model, forward/backward passes, and AdamW optimizer to pure NumPy:
- **Universal Portability**: Runs out-of-the-box on Linux (x86_64, aarch64), macOS (Intel and Apple Silicon), Windows, and Android.
- **Zero Heavy C++ Dependencies**: Only standard `numpy` and `safetensors`.
- **Fast, Clean Unit Testing**: Tests run anywhere in continuous integration without GPU or specialized hardware.

---

## Architectural Mapping

| MLX Component | Pure NumPy Implementation |
|---|---|
| `nn.Embedding(256, dim)` | Array index lookup `embed[c]` with outer product trace updates |
| `nn.Linear(dim, dim)` | Matrix multiplication `norm_s @ weights.T` |
| `nn.LayerNorm(dim)` | Direct mean, variance normalization with analytical backprop |
| `nn.SiLU()` | Numerically clipped $x \cdot \sigma(x)$ with analytical derivative |
| `opt.AdamW` | Standard AdamW moment tracking with decoupled weight decay |
| `mx.save_safetensors` | `safetensors.numpy.save_file` (with `.npz` fallback) |

---

## Weight Format Compatibility

Checkpoints saved by the NumPy implementation use identical keys (`m.*`, `o.*`, `state.*`, `decaytrace.*`, `embedtrace.*`), enabling seamless loading with both `.safetensors` and `.npz`.
