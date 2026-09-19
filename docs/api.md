# Python API Reference

The `tmt` package provides programmatic access to the model, runtime, benchmark suite, and data utilities.

---

## 1. `tmt.model`

### `Model`
Core byte-level recurrent model.

```python
from tmt.model import Model

# Initialize model
model = Model(dim=512, layers=16, temp=0.75, lr=5e-4, seed=42)

# Run single step forward (inference)
(x, states, decays), (logits, stop_val) = model.step(currb=65)

# Train on a transition: byte 'A' (65) -> 'B' (66)
sampled_byte, stop_prob = model(currb=65, nextb=66, end=False)

# Frozen inference (updates recurrent memory without gradient updates)
sampled_byte, stop_prob = model(currb=65, nextb=None, end=False, frozen=True)

# Memoryless inference
sampled_byte, stop_prob = model(currb=65, nextb=None, end=False, notrace=True)

# Save / load checkpoints (safetensors or npz)
model.save("my_model.safetensors")
model.load("my_model.safetensors")

# Reset recurrent memory states
model.reset()
```

### `count_params(dim: int, layers: int) -> int`
Computes the exact number of trainable parameters for a given dimension and layer count.

```python
from tmt.model import count_params

params = count_params(512, 16)
print(f"Parameters: {params:,}")  # 4,481,793
```

---

## 2. `tmt.runtime`

### `Runtime`
High-level orchestrator for training, streaming inference, and interactive chatting.

```python
from tmt.runtime import Runtime

runtime = Runtime(path="my_model.safetensors", dim=512, layers=16)

# Generate text from a prompt
response = runtime.generate("Hello world", max_bytes=100)

# Train on a text dataset or glob pattern
steps = runtime.train_on_data("data/**/*.txt", max_epochs=1)

# Launch interactive chat loop
runtime.chat(readonly=True)
```

---

## 3. `tmt.benchmark`

### `run_benchmark`
Runs linear probe evaluation on CoLA benchmark data.

```python
from tmt.benchmark import run_benchmark

results = run_benchmark(
    checkpoint_path="experimental-4.5m.safetensors",
    data_path="data/samples/sample_cola.tsv",
    epochs=3,
    dev=0.1,
    lr=1e-3,
)

print(f"Final Dev MCC: {results['final_dev_mcc']:.2f}")
```

### `mcc(tp: int, tn: int, fp: int, fn: int) -> float`
Calculates Matthews Correlation Coefficient scaled to $[-100.0, 100.0]$.

```python
from tmt.benchmark import mcc

score = mcc(tp=10, tn=10, fp=2, fn=1)
```

---

## 4. `tmt.data`

### `load_text_lines(path_or_pattern: str, min_bytes: int = 2) -> Iterator[str]`
Streams non-empty lines from files matching a path or glob pattern.

### `load_cola(filepath: str) -> list[tuple[bytes, int]]`
Parses a standard 4-column CoLA TSV file into `(sentence_bytes, label_int)`.

### `create_train_dev_split(items: Sequence[T], dev_fraction: float = 0.1) -> tuple[list[T], list[T]]`
Splits a sequence into train and dev heldout slices.
