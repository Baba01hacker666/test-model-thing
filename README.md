# Test-Model-Thing (TMT)

[YouTube Video](https://youtu.be/9UERVVwpNew)

This is a small proof-of-concept language model (not an LLM) that incorporates the following (and some smaller features as well):
* Latent-space prediction (JEPA style)
* Internal state + recurrent trace units (RTUs)
* Byte input/output ($V = 256$)
* Continuous data streaming
* Test-time training & continual learning
* Modular CLI tool (`tmt`) in pure NumPy with automated tests and bundled sample datasets

The model engine is built with pure **NumPy**, running natively across all platforms (Linux, macOS, Windows, and Android/Termux) without requiring Apple Silicon or heavy framework dependencies.

Being a proof of concept I have only trained a 4.5-million parameter model (keep in mind, GPT-1 was ~117m) for about 12 hours, but there are very promising results. The model tends to misspell characters (since it outputs byte-by-byte, rather than token-by-token) but it is able to close quotes/brackets and such. Given further training and scaling up the hyperparameters this could become much more powerful. My dataset is also tiny (only a few hundred MB), so there's a lot more world knowledge that can be fed into the model.

This model architecture was designed in about a month by me (a solo high school dev) and some Gemini (only pair programming, no agents). I wrote about a dozen prototypes before creating this architecture. I write READMEs myself without AI.

Feel free to fork the training and benchmark code (everything is under MIT). I really encourage you to try things out, submit issues, and fork the repo.

<img width="499" height="497" alt="3f7f1530-c0c7-43c4-9981-30e9023a19fb" src="https://github.com/user-attachments/assets/eb7e5a97-09b5-4a7b-9484-eb898042e9dc" />

---

## Quick Start

```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies & CLI
pip install -e .

# 3. Verify CLI and view model info
tmt info
```

---

## Modular CLI (`tmt`)

TMT provides a unified command line interface:

```bash
# Smoke test: train for 200 steps on bundled sample data
tmt train --sample --steps 200

# Train on custom text dataset or glob
tmt train --data 'wikipedia_clean/**/wiki_*' --path experimental-4.5m.safetensors

# Interactive chat with in-memory learning
tmt chat --readonly

# Interactive chat with frozen weights (memory still advances)
tmt chat --frozen

# One-shot prompt generation
tmt chat --prompt "Hello world" --max-bytes 64

# Run CoLA probe benchmark on bundled sample data
tmt benchmark --sample

# Inspect workspace status and checkpoints
tmt status

# Run the automated test suite
tmt test -v
```

### Backwards Compatibility

All original commands continue to work seamlessly:

```bash
python main.py --mode train --pattern 'wikipedia_clean/**/wiki_*'
python main.py --mode chat
python main.py --mode chatreadonly   # still trains in memory, skips disk save
python main.py --mode chat --frozen  # no in-memory training either
python benchmark.py --path experimental-4.5m.safetensors
```

---

## Testing & Sample Datasets

Lightweight testing datasets are included directly in `data/samples/` so you can train and evaluate instantly:
- `data/samples/sample_text.txt`: Sample text for immediate training tests (`tmt train --sample`).
- `data/samples/sample_cola.tsv`: Sample CoLA TSV for probe evaluation tests (`tmt benchmark --sample`).

To run the complete automated test suite:
```bash
pytest -v
# or via the CLI:
tmt test
```

---

## Documentation

Full documentation is available in the [`docs/`](docs/) directory:
- [Architecture Overview](docs/overview.md)
- [CLI Reference Guide](docs/cli.md)
- [Datasets & Reproduction Guide](docs/datasets.md)
- [Python API Reference](docs/api.md)
- [NumPy Migration Notes](docs/migration_numpy.md)

---

## How it works

In detail, here are some of the main capabilities of the model that differ from LLMs:
* JEPA-style latent space prediction, as the decoder can be removed/disabled and the model still rolls out forward as is. The model is not trained explicitly on predicting the next byte, but rather on two separate goals (predicting the next 'thing' in latent space, and translating the current latent space vector to a byte).
* Theoretically infinite memory, as it does not have a context window and instead relies on RTUs to store internal state/memory. However it does decay old memories over time. Also I think this should be O(1) memory based on my implementation but I'm not 100% sure.
* Built-in multimodality, as the model outputs bytes (and thus should theoretically be capable of handling any binary data).
* Streaming data live, since the model only processes one byte at once at rapid pace. In fact it is completely 'blind' to everything that came before the current byte, only relying on the current processed byte and its internal memory to decide the next byte. This confirms the model is definitely learning to remember things.
* Continual training, as it keeps training on user input, training data, and its own output to improve its predictions automatically. Keep in mind the model can only output a byte (0-255) each pass anyways (in addition to updating its own state).

The two important hyperparameters are the size of the latent vector (dim) and the amount of individual state layers the latent passes through before decoding (layers). For my 4.5m test these are ```dim = 512``` and ```layers = 16```. There are some other configurations you can change but I think they are less important.

For reproduction purposes the dataset I trained my model on is ```simplewiki-20260801-pages-articles.xml.bz2```, from the Wikipedia dumps.

## Benchmark (CoLA)

```tmt benchmark``` freezes the backbone and trains a small linear probe on CoLA (Matthews correlation). Download CoLA from https://nyu-mll.github.io/CoLA/ so that ```CoLA/original/raw/in_domain_train.tsv``` exists, then run:

```bash
tmt benchmark --data CoLA/original/raw/in_domain_train.tsv
```

A dev slice is held out automatically so the reported score reflects generalization.

Below is an approximate flow chart of the model architecture, made in Apple's Freeform app (excluding the wrapper for dataset cleaning and input/output handling) for reference. Note that the arrow connecting the target latent to the CE loss should instead be the target byte to the CE loss.

<img width="1653" height="1161" alt="JEPA thing" src="https://github.com/user-attachments/assets/2d3a34ff-ba6a-44b8-b361-6c73da9216c0" />
