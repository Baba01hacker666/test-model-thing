"""Main entrypoint for Test-Model-Thing (TMT).

Preserves full backwards compatibility with:
  python main.py --mode train --pattern 'wikipedia_clean/**/wiki_*'
  python main.py --mode chat
  python main.py --mode chatreadonly
  python main.py --mode chat --frozen

For modern modular usage with subcommands, use:
  tmt --help
"""

from tmt.cli import main as cli_main
from tmt.model import Model, count_params
from tmt.runtime import Runtime

__all__ = ["Model", "Runtime", "count_params"]

if __name__ == "__main__":
    cli_main()
