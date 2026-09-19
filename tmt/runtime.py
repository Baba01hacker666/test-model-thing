"""Runtime orchestrator for training, interactive chatting, and generation."""

from __future__ import annotations

import itertools
import sys
import time
from datetime import datetime
from typing import Callable

from tmt.data import get_sample_text_path, load_text_lines
from tmt.model import Model


class Runtime:
    """Runtime runner for Test-Model-Thing models."""

    def __init__(
        self,
        path: str = "experimental-4.5m.safetensors",
        threshold: float = 0.35,
        save_every: int = 500,
        dim: int = 512,
        layers: int = 16,
        temp: float = 0.75,
        lr: float = 5e-4,
        seed: int | None = None,
    ):
        self.path = path
        self.threshold = threshold
        self.save_every = save_every
        self.model = Model(dim=dim, layers=layers, temp=temp, lr=lr, seed=seed)

        self.step = 0
        self.prevtime: float | None = None

    def save(self) -> None:
        """Save model checkpoint if step interval reached."""
        self.step += 1
        if self.step % self.save_every == 0:
            self.model.save(self.path)

    def call(
        self,
        c: int,
        n: int | None,
        end: bool,
        readonly: bool = False,
        notrace: bool = False,
        frozen: bool = False,
    ) -> tuple[int, float]:
        """Forward byte through model and conditionally save."""
        if frozen:
            return self.model(c, n, end, frozen=True)
        outputs = self.model(c, n, end, notrace=notrace)
        if not readonly:
            self.save()
        return outputs

    def write(self, b: int) -> None:
        """Write single byte directly to stdout."""
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def now(self) -> str:
        """Current formatted timestamp."""
        return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    def generate(
        self,
        prompt: str,
        max_bytes: int = 256,
        readonly: bool = True,
        notrace: bool = False,
        frozen: bool = False,
        callback: Callable[[int], None] | None = None,
    ) -> str:
        """Generate text given an initial prompt string."""
        data = (prompt + "\n").encode("utf-8")
        for i, (c, n) in enumerate(itertools.pairwise(data)):
            self.call(c, n, i == len(data) - 2, readonly=readonly, notrace=notrace, frozen=frozen)

        b = data[-1]
        out_bytes = bytearray()

        for _ in range(max_bytes):
            b, stop = self.call(b, None, False, readonly=readonly, notrace=notrace, frozen=frozen)
            out_bytes.append(b)
            if callback:
                callback(b)
            if stop > self.threshold:
                break

        return out_bytes.decode("utf-8", errors="replace")

    def chat(
        self,
        readonly: bool = False,
        notrace: bool = False,
        frozen: bool = False,
    ) -> None:
        """Interactive loop in terminal."""
        print(f"[{self.now()}] TMT chat initialized. (readonly={readonly}, notrace={notrace}, frozen={frozen})")
        print("Press Ctrl+C or Ctrl+D to exit.")
        try:
            while True:
                elapsed = 0.0 if self.prevtime is None else time.time() - self.prevtime
                try:
                    text = input(f"\n[{self.now()} | {elapsed:.4f}s]\nUser >> ")
                except EOFError:
                    print()
                    break
                self.prevtime = time.time()

                data = (text + "\n").encode("utf-8")
                for i, (c, n) in enumerate(itertools.pairwise(data)):
                    self.call(c, n, i == len(data) - 2, readonly, notrace, frozen)

                print(f"\n[{self.now()}]\nModel >> ", end="", flush=True)

                b = data[-1]
                while True:
                    b, stop = self.call(b, None, False, readonly, notrace, frozen)
                    self.write(b)
                    if stop > self.threshold:
                        print()
                        break
        except KeyboardInterrupt:
            print("\nExiting chat.")

    def train_on_data(
        self,
        path_or_pattern: str | None = None,
        max_steps: int | None = None,
        max_epochs: int = 1,
        stream_bytes_to_stdout: bool = True,
    ) -> int:
        """Train model sequentially over text matching path or glob pattern."""
        resolved_pattern = path_or_pattern or get_sample_text_path()

        epoch = 0
        total_steps = 0

        while epoch < max_epochs:
            epoch += 1
            for line in load_text_lines(resolved_pattern):
                data = line.encode("utf-8")
                if len(data) < 2:
                    continue
                for i, (c, n) in enumerate(itertools.pairwise(data)):
                    b, _ = self.call(c, n, i == len(data) - 2)
                    if stream_bytes_to_stdout:
                        self.write(b)
                    total_steps += 1
                    if max_steps is not None and total_steps >= max_steps:
                        return total_steps

        return total_steps

    def __call__(self, mode: str, frozen: bool = False) -> None:
        """Execute legacy mode string for backwards compatibility."""
        modes = ["train", "chat", "chatreadonly", "chatnotrace"]

        if mode not in modes:
            print(f"\nInvalid mode {mode!r}. Choose from {modes}.")
            return

        self.model.load(self.path)
        print()

        try:
            match mode:
                case "train":
                    self.train_on_data()
                case "chat":
                    self.chat(frozen=frozen)
                case "chatreadonly":
                    self.chat(readonly=True, frozen=frozen)
                case "chatnotrace":
                    self.chat(readonly=True, notrace=True)
        finally:
            if mode in ("train", "chat") and not frozen:
                self.model.save(self.path)
