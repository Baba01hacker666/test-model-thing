# Datasets & Testing Guide

Test-Model-Thing (TMT) supports raw UTF-8 text files, glob patterns, bundled sample datasets, and benchmark TSV files.

---

## Bundled Sample Datasets

To enable immediate testing and local development without downloading multi-gigabyte files, TMT bundles lightweight testing datasets in `data/samples/`:

### 1. `data/samples/sample_text.txt`
- A sample text corpus containing natural language with varied punctuation, numbers, brackets, and sentences.
- Used automatically when passing `--sample` to `tmt train`, or when omitting `--data`.

```bash
tmt train --sample --steps 200
```

### 2. `data/samples/sample_cola.tsv`
- A 20-row sample of the Corpus of Linguistic Acceptability (CoLA) formatted in standard 4-column TSV.
- Used automatically when passing `--sample` to `tmt benchmark`.

```bash
tmt benchmark --sample
```

### 3. Synthetic Corpus Generator
You can generate synthetic text on the fly using `tmt sample`:
```bash
tmt sample --type synthetic --lines 500 --out synthetic_data.txt
tmt train --data synthetic_data.txt
```

---

## Using Custom Training Data

Any plain UTF-8 text file or directory of text files can be used:

```bash
# Single file
tmt train --data my_data.txt

# Recursive glob pattern
tmt train --data "articles/**/*.txt"
```

The dataset loader automatically filters out lines shorter than 2 bytes and iterates continuously line-by-line.

---

## Wikipedia Dumps (Reproduction Dataset)

The original proof-of-concept was trained on Simple English Wikipedia. To reproduce:

1. Download the latest Simple Wikipedia dump:
   ```bash
   wget https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
   ```

2. Extract and clean the XML dump into clean text using `wikiextractor`:
   ```bash
   pip install wikiextractor
   python -m wikiextractor.WikiExtractor simplewiki-latest-pages-articles.xml.bz2 -o wikipedia_clean/
   ```

3. Train using the glob pattern:
   ```bash
   tmt train --data "wikipedia_clean/**/wiki_*" --path experimental-4.5m.safetensors
   ```

---

## CoLA Benchmark Dataset

The Corpus of Linguistic Acceptability (CoLA) evaluates whether a language representation has learned grammatical rules.

1. Download the official CoLA dataset:
   ```bash
   wget https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
   unzip cola_public_1.1.zip -d CoLA/
   ```

2. Run the probe benchmark:
   ```bash
   tmt benchmark --data CoLA/cola_public/raw/in_domain_train.tsv --epochs 5 --dev 0.1
   ```
