"""Core byte-level recurrent model implementation in pure NumPy."""

from __future__ import annotations

import os

import numpy as np

try:
    import safetensors.numpy as stn

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def count_params(dim: int, layers: int) -> int:
    """Calculate total trainable parameter count for the given dim and layer count."""
    per_layer = (
        dim * dim + 3 * dim
    )  # weights (dim*dim) + norm_weight (dim) + norm_bias (dim) + decay (dim) - decay is dim, no linear bias
    # Breakdown:
    # encoder: embed (256 * dim)
    # layers: layers * (weights (dim*dim) + norm_w (dim) + norm_b (dim) + decay (dim))
    # decoder: decode.weight (256 * dim) + decode.bias (256) + stop.weight (1 * dim) + stop.bias (1)
    return 256 * dim + layers * per_layer + 256 * dim + 256 + dim + 1


def _softmax(x: np.ndarray) -> np.ndarray:
    shift = x - np.max(x)
    exps = np.exp(shift)
    sum_exps = np.sum(exps)
    if sum_exps == 0:
        return np.ones_like(x) / len(x)
    return exps / sum_exps


def _logsumexp(x: np.ndarray) -> float:
    max_x = np.max(x)
    return float(max_x + np.log(np.sum(np.exp(x - max_x))))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    # Numerically stable sigmoid
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _dsilu(x: np.ndarray, sig_x: np.ndarray) -> np.ndarray:
    """Derivative of SiLU with respect to x: sig(x) * (1 + x * (1 - sig(x)))."""
    return sig_x * (1.0 + x * (1.0 - sig_x))


class AdamW:
    """NumPy implementation of the AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        learning_rate: float = 5e-4,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.lr = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.step = 0

    def update(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.step += 1
        lr = self.lr
        bias_corr1 = 1.0 - (self.b1**self.step)
        bias_corr2 = 1.0 - (self.b2**self.step)

        for name, p in params.items():
            if name not in grads:
                continue
            g = grads[name]
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)

            # Weight decay update
            if self.weight_decay != 0:
                p -= lr * self.weight_decay * p

            # Momentum updates
            m = self.m[name]
            v = self.v[name]
            m[:] = self.b1 * m + (1.0 - self.b1) * g
            v[:] = self.b2 * v + (1.0 - self.b2) * (g * g)

            m_hat = m / bias_corr1
            v_hat = v / bias_corr2
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict[str, np.ndarray]:
        res: dict[str, np.ndarray] = {"step": np.array([self.step], dtype=np.int64)}
        for k, arr in self.m.items():
            res[f"m.{k}"] = arr
        for k, arr in self.v.items():
            res[f"v.{k}"] = arr
        return res

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        if "step" in state:
            self.step = int(state["step"][0])
        for k, arr in state.items():
            if k.startswith("m."):
                self.m[k[2:]] = arr.copy()
            elif k.startswith("v."):
                self.v[k[2:]] = arr.copy()


class LayerState:
    """Internal recurrent state and traces for a single layer."""

    def __init__(self, dim: int):
        self.dim = dim
        self.states = np.zeros(dim, dtype=np.float32)
        self.decaytrace = np.zeros(dim, dtype=np.float32)
        self.embedtrace = np.zeros((256, dim), dtype=np.float32)

    def reset(self) -> None:
        self.states.fill(0.0)
        self.decaytrace.fill(0.0)
        self.embedtrace.fill(0.0)


class Model:
    """Test-Model-Thing (TMT) byte-level recurrent model."""

    def __init__(
        self,
        dim: int = 512,
        layers: int = 16,
        temp: float = 0.75,
        lr: float = 5e-4,
        seed: int | None = None,
    ):
        self.dim = dim
        self.layercount = layers
        self.temp = temp
        self.lr = lr
        self._frozen = False

        rng = np.random.default_rng(seed)

        # Encoder: 256 -> dim
        scale = 1.0 / np.sqrt(dim)
        self.embed = (rng.standard_normal((256, dim)) * scale).astype(np.float32)

        # Decoder: dim -> 256, and stop: dim -> 1
        self.dec_w = (rng.standard_normal((256, dim)) * scale).astype(np.float32)
        self.dec_b = np.zeros(256, dtype=np.float32)
        self.stop_w = (rng.standard_normal((1, dim)) * scale).astype(np.float32)
        self.stop_b = np.zeros(1, dtype=np.float32)

        # Layers: decay logit, norm_w, norm_b, linear weights
        self.layer_decay: list[np.ndarray] = [np.zeros(dim, dtype=np.float32) for _ in range(layers)]
        self.layer_norm_w: list[np.ndarray] = [np.ones(dim, dtype=np.float32) for _ in range(layers)]
        self.layer_norm_b: list[np.ndarray] = [np.zeros(dim, dtype=np.float32) for _ in range(layers)]
        self.layer_w: list[np.ndarray] = [
            (rng.standard_normal((dim, dim)) * scale).astype(np.float32) for _ in range(layers)
        ]

        # Recurrent state per layer
        self.layers = [LayerState(dim) for _ in range(layers)]
        self.optimizer = AdamW(learning_rate=lr)

    def freeze(self) -> None:
        """Freeze model parameters so no gradient updates occur."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze model parameters."""
        self._frozen = False

    def reset(self) -> None:
        """Reset internal recurrent state and traces across all layers."""
        for layer in self.layers:
            layer.reset()

    def sample(self, logits: np.ndarray) -> int:
        """Categorical sampling with entropy-scaled temperature."""
        probs = _softmax(logits)
        entropy = -float(np.sum(probs * np.log(probs + 1e-8))) / float(np.log(256.0))
        temp = max(0.1, float(self.temp * (1.0 - self.temp * entropy)))
        scaled = logits / temp
        probs_adj = _softmax(scaled)
        return int(np.random.choice(256, p=probs_adj))

    def step(
        self, c: int, dummies: list[np.ndarray] | None = None
    ) -> tuple[tuple[np.ndarray, list[np.ndarray], list[np.ndarray]], tuple[np.ndarray, float]]:
        """Run single step forward for byte c."""
        enc = self.embed[c]
        x = enc.copy()
        states, decays = [], []

        for i in range(self.layercount):
            dummy = dummies[i] if dummies is not None else np.zeros(self.dim, dtype=np.float32)
            decay = _sigmoid(self.layer_decay[i]).astype(np.float32)
            state = (decay * self.layers[i].states) + enc + dummy

            # LayerNorm
            mean = np.mean(state)
            var = np.var(state)
            norm_s = (state - mean) / np.sqrt(var + 1e-5) * self.layer_norm_w[i] + self.layer_norm_b[i]

            # Linear + SiLU
            lin = norm_s @ self.layer_w[i].T
            act = _silu(lin)
            x = x + act

            states.append(state)
            decays.append(decay)

        logits = (x @ self.dec_w.T + self.dec_b).astype(np.float32)
        stop_val = float(_sigmoid(x @ self.stop_w.T + self.stop_b)[0])

        return (x, states, decays), (logits, stop_val)

    def __call__(
        self,
        currb: int,
        nextb: int | None,
        end: bool,
        notrace: bool = False,
        frozen: bool = False,
    ) -> tuple[int, float]:
        """Process byte currb, optionally train on nextb/end, and return (sampled_byte, stop_prob)."""
        is_frozen = self._frozen or frozen

        if notrace:
            # Memoryless pass: zero states and dummies
            enc = self.embed[currb]
            x = enc.copy()
            for i in range(self.layercount):
                decay = _sigmoid(self.layer_decay[i])
                state = enc.copy()  # no accumulated state
                mean = np.mean(state)
                var = np.var(state)
                norm_s = (state - mean) / np.sqrt(var + 1e-5) * self.layer_norm_w[i] + self.layer_norm_b[i]
                lin = norm_s @ self.layer_w[i].T
                x = x + _silu(lin)
            logits = x @ self.dec_w.T + self.dec_b
            stop_val = float(_sigmoid(x @ self.stop_w.T + self.stop_b)[0])
            return self.sample(logits), stop_val

        if is_frozen:
            # Stateful inference without parameter updates
            (x, states, _), (logits, stop_val) = self.step(currb)
            for i, st in enumerate(states):
                self.layers[i].states = st.astype(np.float32)
            return self.sample(logits), stop_val

        # --- Training Forward Pass ---
        enc = self.embed[currb]
        x = enc.copy()
        states: list[np.ndarray] = []
        decays: list[np.ndarray] = []
        norm_states: list[np.ndarray] = []
        lins: list[np.ndarray] = []
        sig_lins: list[np.ndarray] = []
        stds: list[float] = []
        norm_s_raws: list[np.ndarray] = []

        for i in range(self.layercount):
            decay = _sigmoid(self.layer_decay[i]).astype(np.float32)
            state = (decay * self.layers[i].states) + enc

            mean = np.mean(state)
            var = np.var(state)
            std = float(np.sqrt(var + 1e-5))
            norm_s_raw = (state - mean) / std
            norm_s = norm_s_raw * self.layer_norm_w[i] + self.layer_norm_b[i]

            lin = norm_s @ self.layer_w[i].T
            sig_lin = _sigmoid(lin).astype(np.float32)
            act = lin * sig_lin

            x = x + act

            states.append(state)
            decays.append(decay)
            norm_states.append(norm_s)
            lins.append(lin)
            sig_lins.append(sig_lin)
            stds.append(std)
            norm_s_raws.append(norm_s_raw)

        logits = (x @ self.dec_w.T + self.dec_b).astype(np.float32)
        stop_val = float(_sigmoid(x @ self.stop_w.T + self.stop_b)[0])

        # --- Loss and Gradients ---
        grads: dict[str, np.ndarray] = {}

        # 1. Variance regularization loss gradient on final latent x
        dim = self.dim
        var_x = np.var(x)
        std_x = np.sqrt(var_x + 1e-4)
        if 1.0 - std_x > 0:
            dx_total = -((x - np.mean(x)) / (dim * std_x))
        else:
            dx_total = np.zeros_like(x)

        dloss_denc = np.zeros(dim, dtype=np.float32)

        if nextb is not None:
            # 2. Prediction MSE loss: mean((x - tgt)**2)
            tgt = self.embed[nextb]
            dx_total += (2.0 / dim) * (x - tgt)

            # 3. Cross-entropy loss on logits
            probs = _softmax(logits)
            dlogits = probs.copy()
            dlogits[nextb] -= 1.0

            grads["decoder.decode.weight"] = np.outer(dlogits, x)
            grads["decoder.decode.bias"] = dlogits
            dx_total += dlogits @ self.dec_w

            # 4. Stop prediction MSE loss: (stop - target)**2
            target_stop = 1.0 if end else 0.0
            dstop = 2.0 * (stop_val - target_stop)
            dstop_lin = dstop * stop_val * (1.0 - stop_val)
            grads["decoder.stop.weight"] = np.outer(np.array([dstop_lin]), x)
            grads["decoder.stop.bias"] = np.array([dstop_lin], dtype=np.float32)
            dx_total += dstop_lin * self.stop_w[0]

        # Backprop through layers
        dlds_s: list[np.ndarray] = []
        dx = dx_total

        for i in reversed(range(self.layercount)):
            # Residual connection: dx passes straight through to layer input
            dact = dx.copy()
            dlin = dact * _dsilu(lins[i], sig_lins[i])

            grads[f"layers.{i}.weights.weight"] = np.outer(dlin, norm_states[i])
            dnorm_s = dlin @ self.layer_w[i]

            grads[f"layers.{i}.norm.weight"] = dnorm_s * norm_s_raws[i]
            grads[f"layers.{i}.norm.bias"] = dnorm_s

            # LayerNorm backprop w.r.t input state
            ds_norm = dnorm_s * self.layer_norm_w[i]
            x_hat = norm_s_raws[i]
            std = stds[i]
            dstate = (1.0 / (dim * std)) * (dim * ds_norm - np.sum(ds_norm) - x_hat * np.sum(ds_norm * x_hat))

            dlds_s.insert(0, dstate)
            dloss_denc += dstate
            # In addition, residual x carries through to earlier layer: dx remains dx

        # Encoder receives dx from layer 0 residual input plus state branches
        dloss_denc += dx
        grads["encoder.embed.weight"] = np.zeros_like(self.embed)
        grads["encoder.embed.weight"][currb] += dloss_denc

        # --- Recurrent Trace Unit (RTU) updates ---
        c_onehot = np.zeros(256, dtype=np.float32)
        c_onehot[currb] = 1.0

        for i in range(self.layercount):
            dlds = dlds_s[i]
            decay = decays[i]
            layer = self.layers[i]

            embedtrace = (layer.embedtrace * decay) + c_onehot[:, None]
            grads["encoder.embed.weight"] += np.outer(c_onehot, dlds * (layer.embedtrace * decay).sum(axis=0))

            decaytrace = (decay * layer.decaytrace) + (decay * (1.0 - decay) * layer.states)
            grads[f"layers.{i}.decay"] = dlds * decaytrace

            layer.states = states[i].astype(np.float32)
            layer.decaytrace = decaytrace.astype(np.float32)
            layer.embedtrace = embedtrace.astype(np.float32)

        # Apply gradients with AdamW
        params = self._parameter_map()
        self.optimizer.update(params, grads)

        return self.sample(logits), stop_val

    def _parameter_map(self) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {
            "encoder.embed.weight": self.embed,
            "decoder.decode.weight": self.dec_w,
            "decoder.decode.bias": self.dec_b,
            "decoder.stop.weight": self.stop_w,
            "decoder.stop.bias": self.stop_b,
        }
        for i in range(self.layercount):
            params[f"layers.{i}.decay"] = self.layer_decay[i]
            params[f"layers.{i}.norm.weight"] = self.layer_norm_w[i]
            params[f"layers.{i}.norm.bias"] = self.layer_norm_b[i]
            params[f"layers.{i}.weights.weight"] = self.layer_w[i]
        return params

    def save(self, path: str) -> None:
        """Save weights, optimizer state, and recurrent traces to path (.safetensors or .npz)."""
        data: dict[str, np.ndarray] = {}

        # Model parameters
        for k, v in self._parameter_map().items():
            data[f"m.{k}"] = v

        # Optimizer state
        for k, v in self.optimizer.state_dict().items():
            data[f"o.{k}"] = v

        # Recurrent state and traces
        for i, layer in enumerate(self.layers):
            data[f"state.{i}"] = layer.states
            data[f"decaytrace.{i}"] = layer.decaytrace
            data[f"embedtrace.{i}"] = layer.embedtrace

        tmp = os.path.join(os.path.dirname(path) or ".", f"temporary-{os.path.basename(path)}")

        if path.endswith(".npz") or not HAS_SAFETENSORS:
            np.savez(tmp, **data)
        else:
            stn.save_file(data, tmp)

        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        """Load weights, optimizer state, and recurrent traces from path."""
        if not os.path.exists(path):
            return False

        if path.endswith(".npz") or not HAS_SAFETENSORS:
            npz = np.load(path)
            data = {k: npz[k] for k in npz.files}
        else:
            data = stn.load_file(path)

        params = self._parameter_map()
        opt_state: dict[str, np.ndarray] = {}

        for k, v in data.items():
            if k.startswith("m."):
                pname = k[2:]
                if pname in params and params[pname].shape == v.shape:
                    params[pname][:] = v
            elif k.startswith("o."):
                opt_state[k[2:]] = v
            elif k.startswith("state."):
                idx = int(k.split(".")[1])
                if idx < self.layercount and self.layers[idx].states.shape == v.shape:
                    self.layers[idx].states[:] = v
            elif k.startswith("decaytrace."):
                idx = int(k.split(".")[1])
                if idx < self.layercount and self.layers[idx].decaytrace.shape == v.shape:
                    self.layers[idx].decaytrace[:] = v
            elif k.startswith("embedtrace."):
                idx = int(k.split(".")[1])
                if idx < self.layercount and self.layers[idx].embedtrace.shape == v.shape:
                    self.layers[idx].embedtrace[:] = v

        if opt_state:
            self.optimizer.load_state_dict(opt_state)

        return True
