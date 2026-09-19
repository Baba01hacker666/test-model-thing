# Command-Line Interface (CLI) Guide

Test-Model-Thing provides a modern, modular CLI tool `tmt` (also runnable via `python -m tmt.cli` or legacy `python main.py`).

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Baba01hacker666/test-model-thing.git
cd test-model-thing

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable CLI package
pip install -e .
```

Verify installation:
```bash
tmt --help
```

---

## Subcommands

### 1. `tmt train`
Train the model on custom text files, glob patterns, or the bundled testing dataset.

```bash
# Train on bundled sample data (quick smoke test)
tmt train --sample --steps 500

# Train on custom text files with glob matching
tmt train --data "wikipedia_clean/**/wiki_*" --path my_model.safetensors

# Customize architecture and hyperparameters
tmt train --data "data/corpus.txt" --dim 256 --layers 8 --epochs 3 --lr 5e-4 --save-every 1000
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--data` | `-d` | `None` | Path or glob pattern to text files |
| `--sample` | | `False` | Use bundled sample text dataset (`data/samples/sample_text.txt`) |
| `--path` | `-p` | `experimental-4.5m.safetensors` | Output/input checkpoint file path |
| `--epochs` | `-e` | `1` | Number of training epochs |
| `--steps` | `-s` | `None` | Maximum byte steps (unbounded if omitted) |
| `--lr` | | `5e-4` | AdamW learning rate |
| `--dim` | | `512` | Latent state dimension |
| `--layers` | | `16` | Number of recurrent layers |
| `--temp` | | `0.75` | Sampling temperature |
| `--threshold` | | `0.35` | Stop probability threshold |
| `--save-every` | | `500` | Periodic checkpoint saving interval |
| `--quiet` | `-q` | `False` | Suppress streaming byte output to stdout |

---

### 2. `tmt chat`
Chat with the model interactively or generate output from a single prompt.

```bash
# Interactive chat with in-memory learning (skips saving to disk)
tmt chat --readonly

# Interactive chat with completely frozen weights (memory still advances)
tmt chat --frozen

# Interactive chat with memoryless notrace mode
tmt chat --notrace

# One-shot non-interactive generation from a prompt string
tmt chat --prompt "Once upon a time" --max-bytes 128
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--prompt` | `-p` | `None` | Starting prompt string for one-shot generation |
| `--max-bytes` | `-n` | `256` | Maximum bytes to generate |
| `--path` | | `experimental-4.5m.safetensors` | Model checkpoint path |
| `--readonly` | | `False` | Continual learning in memory without saving weights to disk |
| `--frozen` | | `False` | Freeze all weights; recurrent states still advance |
| `--notrace` | | `False` | Memoryless mode (zero recurrent state) |
| `--temp` | | `0.75` | Sampling temperature |
| `--threshold` | | `0.35` | Stop probability threshold |
| `--quiet` | `-q` | `False` | Only print the final output string |

---

### 3. `tmt benchmark`
Evaluate backbone representation quality using a linear probe on the Corpus of Linguistic Acceptability (CoLA).

```bash
# Benchmark on bundled sample CoLA dataset
tmt benchmark --sample

# Benchmark on full official CoLA dataset
tmt benchmark --data CoLA/original/raw/in_domain_train.tsv --path experimental-4.5m.safetensors --epochs 5 --dev 0.1
```

| Flag | Short | Default | Description |
|---|---|---|---|
| `--path` | `-p` | `experimental-4.5m.safetensors` | Backbone checkpoint path |
| `--data` | `-d` | `None` | CoLA TSV file path |
| `--sample` | | `False` | Use bundled sample CoLA dataset |
| `--epochs` | `-e` | `3` | Number of probe training epochs |
| `--dev` | | `0.1` | Heldout validation split fraction |
| `--lr` | | `1e-3` | Probe classifier learning rate |

---

### 4. `tmt info`
Inspect architecture parameters, parameter count breakdown, and checkpoint status.

```bash
tmt info
tmt info --dim 256 --layers 8 --path custom_model.safetensors
```

---

### 5. `tmt status`
Inspect local workspace status, datasets, and existing checkpoint files.

```bash
tmt status
```

---

### 6. `tmt sample`
Preview or export bundled testing datasets, or generate synthetic corpora.

```bash
# Preview bundled sample text
tmt sample --type text

# Export bundled sample CoLA to a custom path
tmt sample --type cola --out my_test.tsv

# Generate 100 synthetic text lines
tmt sample --type synthetic --lines 100 --out synthetic_corpus.txt
```

---

### 7. `tmt test`
Run the automated test suite directly through the CLI:

```bash
tmt test
tmt test -v
tmt test -k "benchmark"
```

---

## Legacy Backwards Compatibility

All previous commands from the original repository continue to work without modification:

```bash
python main.py --mode train --pattern 'wikipedia_clean/**/wiki_*'
python main.py --mode chat
python main.py --mode chatreadonly
python main.py --mode chat --frozen
python benchmark.py --path experimental-4.5m.safetensors
```
