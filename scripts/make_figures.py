"""Compatibility entry point for the typed study figures."""

import sys
from pathlib import Path

from llm_grpo_gains.figures import main

if __name__ == "__main__":
    raise SystemExit(
        main(["--results-dir", str(Path(__file__).parents[1] / "results"), *sys.argv[1:]])
    )
