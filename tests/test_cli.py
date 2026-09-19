"""Unit and integration tests for tmt.cli."""

from __future__ import annotations

from tmt.cli import build_parser, main


def test_parser_subcommands():
    import argparse

    parser = build_parser()
    subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    subcommands = list(subparsers_action.choices.keys())
    assert "train" in subcommands
    assert "chat" in subcommands
    assert "benchmark" in subcommands
    assert "info" in subcommands
    assert "status" in subcommands
    assert "sample" in subcommands
    assert "test" in subcommands


def test_cli_info(capsys):
    main(["info", "--dim", "64", "--layers", "2"])
    captured = capsys.readouterr()
    assert "Architecture Information" in captured.out
    assert "41,665" in captured.out


def test_cli_status(capsys):
    main(["status"])
    captured = capsys.readouterr()
    assert "Workspace Status" in captured.out
    assert "Sample Text Dataset" in captured.out


def test_cli_sample_text(capsys):
    main(["sample", "--type", "text"])
    captured = capsys.readouterr()
    assert "Sample Text Dataset Preview" in captured.out
    assert "The quick brown fox" in captured.out


def test_cli_sample_cola(capsys):
    main(["sample", "--type", "cola"])
    captured = capsys.readouterr()
    assert "Sample CoLA Dataset Preview" in captured.out


def test_cli_train_and_chat(temp_dir, capsys):
    ckpt = str(temp_dir / "cli_model.safetensors")

    # Train for 20 steps on bundled sample data
    main(["train", "--sample", "--path", ckpt, "--dim", "32", "--layers", "2", "--steps", "20", "--quiet"])
    captured = capsys.readouterr()
    assert "Training finished successfully" in captured.out

    # Prompt generation
    main(["chat", "--prompt", "Hello", "--path", ckpt, "--dim", "32", "--layers", "2", "--max-bytes", "10", "--quiet"])
    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


def test_cli_benchmark_sample(temp_dir, capsys):
    ckpt = str(temp_dir / "bench_model.safetensors")

    # Create backbone checkpoint
    main(["train", "--sample", "--path", ckpt, "--dim", "32", "--layers", "2", "--steps", "10", "--quiet"])
    capsys.readouterr()

    # Benchmark on sample CoLA
    main(["benchmark", "--sample", "--path", ckpt, "--dim", "32", "--layers", "2", "--epochs", "1", "--dev", "0.2"])
    captured = capsys.readouterr()
    assert "Final Dev   MCC" in captured.out
