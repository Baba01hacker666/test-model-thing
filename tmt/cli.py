"""Command Line Interface (CLI) for Test-Model-Thing (TMT).

Modular, pure-NumPy CLI with subcommands:
  tmt train      - Train backbone on text files, globs, or sample datasets
  tmt chat       - Interactive conversation or prompt generation
  tmt benchmark  - CoLA probe evaluation (MCC) with dev heldout split
  tmt info       - Model architecture, parameter counts, and checkpoint inspector
  tmt status     - Workspace status and dataset checks
  tmt sample     - Inspect or export bundled sample datasets
  tmt test       - Run automated test suite via pytest
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tmt import __version__
from tmt.benchmark import run_benchmark
from tmt.data import (
    DEFAULT_SAMPLE_COLA,
    DEFAULT_SAMPLE_TEXT,
    generate_synthetic_corpus,
    get_sample_cola_path,
    get_sample_text_path,
)
from tmt.model import count_params
from tmt.runtime import Runtime
from tmt.torch_model import is_torch_available


def cmd_train(args: argparse.Namespace) -> None:
    """Execute model training."""
    print("═" * 60)
    print(f"🚀 TMT Training: {args.dim} dim | {args.layers} layers (~{count_params(args.dim, args.layers):,} params)")
    print("═" * 60)

    runtime = Runtime(
        path=args.path,
        threshold=args.threshold,
        save_every=args.save_every,
        dim=args.dim,
        layers=args.layers,
        temp=args.temp,
        lr=args.lr,
    )

    if os.path.exists(args.path):
        print(f"📦 Resuming from existing checkpoint: {args.path}")
        runtime.model.load(args.path)
    else:
        print(f"✨ Initializing fresh model weights -> {args.path}")

    data_source = args.data
    if args.sample or not data_source:
        data_source = get_sample_text_path()
        print(f"📂 Using bundled sample dataset: {data_source}")
    else:
        print(f"📂 Training data: {data_source}")

    print(f"⚙️  Epochs: {args.epochs} | Max steps: {args.steps or 'unbounded'} | LR: {args.lr}")
    print("─" * 60)

    try:
        total_steps = runtime.train_on_data(
            path_or_pattern=data_source,
            max_steps=args.steps,
            max_epochs=args.epochs,
            stream_bytes_to_stdout=not args.quiet,
        )
        print(f"\n\n✅ Training finished successfully ({total_steps:,} byte-steps).")
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user. Saving checkpoint...")
    finally:
        runtime.model.save(args.path)
        print(f"💾 Checkpoint saved to: {args.path}")


def cmd_chat(args: argparse.Namespace) -> None:
    """Execute conversational chat or one-shot prompt generation."""
    runtime = Runtime(
        path=args.path,
        threshold=args.threshold,
        dim=args.dim,
        layers=args.layers,
        temp=args.temp,
        lr=args.lr,
    )

    if os.path.exists(args.path):
        runtime.model.load(args.path)
    elif not args.prompt:
        print(f"ℹ️ Note: No checkpoint found at {args.path!r}. Using randomly initialized weights.")

    if args.prompt:
        # Non-interactive generation
        out = runtime.generate(
            prompt=args.prompt,
            max_bytes=args.max_bytes,
            readonly=args.readonly,
            notrace=args.notrace,
            frozen=args.frozen,
            callback=runtime.write if not args.quiet else None,
        )
        if args.quiet:
            print(out)
        else:
            print()
    else:
        # Interactive loop
        runtime.chat(
            readonly=args.readonly,
            notrace=args.notrace,
            frozen=args.frozen,
        )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Execute CoLA benchmark probe evaluation."""
    print("═" * 60)
    print("📊 TMT Benchmark: CoLA Linguistic Acceptability Probe")
    print("═" * 60)

    data_path = args.data
    if args.sample or not data_path:
        data_path = get_sample_cola_path()
        print(f"📂 Using sample CoLA dataset: {data_path}")
    else:
        print(f"📂 Dataset: {data_path}")

    print(f"⚙️  Backbone checkpoint: {args.path}")
    print(f"⚙️  Probe epochs: {args.epochs} | Dev fraction: {args.dev:.2f} | Probe LR: {args.lr}")
    print("─" * 60)

    try:
        results = run_benchmark(
            checkpoint_path=args.path,
            data_path=data_path,
            epochs=args.epochs,
            dev=args.dev,
            lr=args.lr,
            dim=args.dim,
            layers=args.layers,
            verbose=True,
        )
        print("═" * 60)
        print(f"🏆 Final Train MCC: {results['final_train_mcc']:.2f}")
        print(f"🏆 Final Dev   MCC: {results['final_dev_mcc']:.2f}")
        print("═" * 60)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_info(args: argparse.Namespace) -> None:
    """Inspect model configuration and checkpoint metadata."""
    params = count_params(args.dim, args.layers)
    print("═" * 60)
    print(f"🧠 Test-Model-Thing (TMT) Architecture Information (v{__version__})")
    print("═" * 60)
    print(f"  • Hidden Dimension:     {args.dim}")
    print(f"  • Layer Count:          {args.layers}")
    print(f"  • Total Parameters:     {params:,} (~{params / 1e6:.2f}M)")
    print("  • Vocab Size:           256 (Raw byte processing)")
    print("  • Engine Backend:       Pure NumPy + SafeTensors")
    print(f"  • Optional PyTorch:     {'Installed' if is_torch_available() else 'Not installed (optional)'}")
    print("  • Memory Mechanism:     Recurrent Trace Units (RTUs) with learned decay")
    print("  • Training Paradigm:    Latent-space JEPA prediction + continual learning")
    print("─" * 60)

    chkpt = Path(args.path)
    if chkpt.exists():
        size_mb = chkpt.stat().st_size / (1024 * 1024)
        print(f"📦 Checkpoint Status:    FOUND ({chkpt})")
        print(f"  • File Size:            {size_mb:.2f} MB")
        print(f"  • Modified Time:        {chkpt.stat().st_mtime}")
    else:
        print(f"📦 Checkpoint Status:    NOT FOUND ({chkpt})")
        print("  • Run 'tmt train' to train or create one.")
    print("═" * 60)


def cmd_status(args: argparse.Namespace) -> None:
    """Display workspace status, dataset availability, and checkpoints."""
    print("═" * 60)
    print("📋 TMT Workspace Status")
    print("═" * 60)
    print(f"  • Package Version:     {__version__}")
    print(f"  • Current Directory:   {os.getcwd()}")
    print(f"  • Sample Text Dataset: {'FOUND' if DEFAULT_SAMPLE_TEXT.exists() else 'MISSING'} ({DEFAULT_SAMPLE_TEXT})")
    print(f"  • Sample CoLA Dataset: {'FOUND' if DEFAULT_SAMPLE_COLA.exists() else 'MISSING'} ({DEFAULT_SAMPLE_COLA})")

    checkpoints = list(Path(".").glob("*.safetensors")) + list(Path(".").glob("*.npz"))
    print(f"  • Local Checkpoints:   {len(checkpoints)}")
    for ck in checkpoints:
        print(f"      - {ck.name} ({ck.stat().st_size / (1024 * 1024):.2f} MB)")
    print("═" * 60)


def cmd_sample(args: argparse.Namespace) -> None:
    """Inspect or export sample datasets."""
    if args.type == "text":
        path = get_sample_text_path()
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if args.out:
            Path(args.out).write_text(content, encoding="utf-8")
            print(f"✅ Sample text dataset exported to: {args.out}")
        else:
            print("📄 Sample Text Dataset Preview:")
            print("─" * 40)
            print(content)
    elif args.type == "cola":
        path = get_sample_cola_path()
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if args.out:
            Path(args.out).write_text(content, encoding="utf-8")
            print(f"✅ Sample CoLA dataset exported to: {args.out}")
        else:
            print("📄 Sample CoLA Dataset Preview:")
            print("─" * 40)
            print(content)
    elif args.type == "synthetic":
        lines = generate_synthetic_corpus(num_lines=args.lines)
        text = "".join(lines)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"✅ Generated {len(lines)} synthetic lines to: {args.out}")
        else:
            print(f"📄 Synthetic Corpus Preview ({min(10, len(lines))} lines):")
            print("─" * 40)
            print("".join(lines[:10]))


def cmd_test(args: argparse.Namespace) -> None:
    """Run automated unit and integration tests using pytest."""
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Run: pip install pytest", file=sys.stderr)
        sys.exit(1)

    test_args = ["tests"]
    if args.verbose:
        test_args.append("-v")
    if args.filter:
        test_args.extend(["-k", args.filter])

    print(f"🧪 Running test suite: pytest {' '.join(test_args)}")
    ret = pytest.main(test_args)
    sys.exit(ret)


def build_parser() -> argparse.ArgumentParser:
    """Construct full modular CLI argument parser with subcommands and legacy compatibility."""
    parser = argparse.ArgumentParser(
        prog="tmt",
        description="Test-Model-Thing: Modular byte-level recurrent LM in pure NumPy.",
    )
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")

    # Backwards-compatibility top-level flags (e.g. `python main.py --mode train`)
    parser.add_argument(
        "--mode",
        choices=["train", "chat", "chatreadonly", "chatnotrace"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pattern", default=None, help=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest="subcommand", title="Commands")

    # tmt train
    p_train = subparsers.add_parser("train", help="Train model on text files, glob, or sample data")
    p_train.add_argument("--data", "-d", default=None, help="File path or glob pattern (e.g. 'data/**/*.txt')")
    p_train.add_argument("--sample", action="store_true", help="Use bundled sample text dataset")
    p_train.add_argument("--path", "-p", default="experimental-4.5m.safetensors", help="Checkpoint file path")
    p_train.add_argument("--epochs", "-e", type=int, default=1, help="Number of training epochs (default: 1)")
    p_train.add_argument("--steps", "-s", type=int, default=None, help="Max training steps (default: full dataset)")
    p_train.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    p_train.add_argument("--dim", type=int, default=512, help="Latent dimension (default: 512)")
    p_train.add_argument("--layers", type=int, default=16, help="Layer count (default: 16)")
    p_train.add_argument("--temp", type=float, default=0.75, help="Sampling temperature (default: 0.75)")
    p_train.add_argument("--threshold", type=float, default=0.35, help="Stop threshold (default: 0.35)")
    p_train.add_argument("--save-every", type=int, default=500, help="Save frequency in steps (default: 500)")
    p_train.add_argument("--quiet", "-q", action="store_true", help="Suppress streaming stdout output")
    p_train.set_defaults(func=cmd_train)

    # tmt chat
    p_chat = subparsers.add_parser("chat", help="Chat interactively or generate from prompt")
    p_chat.add_argument("--prompt", "-p", default=None, help="Prompt string for one-shot non-interactive generation")
    p_chat.add_argument("--max-bytes", "-n", type=int, default=256, help="Max bytes to generate (default: 256)")
    p_chat.add_argument("--path", default="experimental-4.5m.safetensors", help="Checkpoint file path")
    p_chat.add_argument("--readonly", action="store_true", help="Learn in memory only, do not save to disk")
    p_chat.add_argument("--frozen", action="store_true", help="Freeze weights, only advance recurrent states")
    p_chat.add_argument("--notrace", action="store_true", help="Zero recurrent memory / memoryless mode")
    p_chat.add_argument("--dim", type=int, default=512, help="Latent dimension (default: 512)")
    p_chat.add_argument("--layers", type=int, default=16, help="Layer count (default: 16)")
    p_chat.add_argument("--temp", type=float, default=0.75, help="Sampling temperature (default: 0.75)")
    p_chat.add_argument("--threshold", type=float, default=0.35, help="Stop threshold (default: 0.35)")
    p_chat.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    p_chat.add_argument("--quiet", "-q", action="store_true", help="Do not stream bytes, only print final text")
    p_chat.set_defaults(func=cmd_chat)

    # tmt benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run CoLA probe benchmark evaluation")
    p_bench.add_argument("--path", "-p", default="experimental-4.5m.safetensors", help="Backbone checkpoint file")
    p_bench.add_argument("--data", "-d", default=None, help="CoLA TSV file path (omit or use --sample for bundled)")
    p_bench.add_argument("--sample", action="store_true", help="Use bundled sample CoLA TSV dataset")
    p_bench.add_argument("--epochs", "-e", type=int, default=3, help="Probe training epochs (default: 3)")
    p_bench.add_argument("--dev", type=float, default=0.1, help="Heldout dev fraction (default: 0.1)")
    p_bench.add_argument("--lr", type=float, default=1e-3, help="Probe learning rate (default: 1e-3)")
    p_bench.add_argument("--dim", type=int, default=512, help="Backbone dimension (default: 512)")
    p_bench.add_argument("--layers", type=int, default=16, help="Backbone layer count (default: 16)")
    p_bench.set_defaults(func=cmd_benchmark)

    # tmt info
    p_info = subparsers.add_parser("info", help="Inspect model parameters and checkpoint")
    p_info.add_argument("--dim", type=int, default=512, help="Latent dimension (default: 512)")
    p_info.add_argument("--layers", type=int, default=16, help="Layer count (default: 16)")
    p_info.add_argument("--path", "-p", default="experimental-4.5m.safetensors", help="Checkpoint to inspect")
    p_info.set_defaults(func=cmd_info)

    # tmt status
    p_stat = subparsers.add_parser("status", help="Display workspace status and datasets")
    p_stat.set_defaults(func=cmd_status)

    # tmt sample
    p_samp = subparsers.add_parser("sample", help="Inspect or export bundled testing datasets")
    p_samp.add_argument("--type", choices=["text", "cola", "synthetic"], default="text", help="Dataset type")
    p_samp.add_argument("--lines", type=int, default=50, help="Number of synthetic lines (for type=synthetic)")
    p_samp.add_argument("--out", "-o", default=None, help="Destination file path to export")
    p_samp.set_defaults(func=cmd_sample)

    # tmt test
    p_test = subparsers.add_parser("test", help="Run automated test suite via pytest")
    p_test.add_argument("--verbose", "-v", action="store_true", help="Run pytest with verbose output")
    p_test.add_argument("--filter", "-k", default=None, help="Pytest filter expression")
    p_test.set_defaults(func=cmd_test)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle legacy --mode invocation (e.g. `python main.py --mode train ...`)
    if args.mode is not None and args.subcommand is None:
        dim = getattr(args, "dim", 512)
        layers = getattr(args, "layers", 16)
        path = getattr(args, "path", "experimental-4.5m.safetensors")
        temp = getattr(args, "temp", 0.75)
        threshold = getattr(args, "threshold", 0.35)
        lr = getattr(args, "lr", 5e-4)

        runtime = Runtime(
            path=path,
            threshold=threshold,
            dim=dim,
            layers=layers,
            temp=temp,
            lr=lr,
        )

        if args.mode == "train":
            pattern = getattr(args, "pattern", None)
            if os.path.exists(path):
                runtime.model.load(path)
            try:
                runtime.train_on_data(path_or_pattern=pattern)
            finally:
                runtime.model.save(path)
        else:
            runtime(args.mode, frozen=getattr(args, "frozen", False))
        return

    if hasattr(args, "func"):
        args.func(args)
    else:
        # Show help with friendly banner
        print("═" * 60)
        print(f"🤖 Test-Model-Thing (TMT) CLI v{__version__}")
        print("═" * 60)
        print("A byte-level recurrent language model in pure NumPy.\n")
        parser.print_help()
        print("\n💡 Quick Start Examples:")
        print("  tmt train --sample                   # Train model on bundled sample data")
        print("  tmt chat --prompt 'Hello world'      # Generate from a prompt")
        print("  tmt chat --readonly                  # Chat with in-memory learning")
        print("  tmt benchmark --sample               # Run CoLA probe on sample data")
        print("  tmt info                             # Inspect model architecture and parameters")
        print("  tmt test                             # Run all tests with pytest")


if __name__ == "__main__":
    main()
