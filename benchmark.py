"""Benchmark entrypoint for Test-Model-Thing (TMT).

Preserves full backwards compatibility with:
  python benchmark.py --path experimental-4.5m.safetensors --data CoLA/...

For modern modular usage with subcommands, use:
  tmt benchmark --sample
"""

from tmt.benchmark import ClassificationHead as Classification
from tmt.benchmark import mcc, rollout, run_benchmark
from tmt.data import load_cola as cola
from tmt.model import Model


def run(path: str, data: str = "data/samples/sample_cola.tsv", epochs: int = 3, dev: float = 0.1) -> None:
    """Run benchmark evaluation with backwards-compatible argument signature."""
    run_benchmark(checkpoint_path=path, data_path=data, epochs=epochs, dev=dev)


__all__ = ["Classification", "cola", "mcc", "rollout", "run", "Model"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoLA probe for a frozen TMT backbone.")
    parser.add_argument("--path", default="experimental-4.5m.safetensors")
    parser.add_argument("--data", default="data/samples/sample_cola.tsv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dev", type=float, default=0.1)
    args = parser.parse_args()

    run(args.path, args.data, args.epochs, args.dev)
