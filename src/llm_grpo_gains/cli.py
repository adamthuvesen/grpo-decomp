"""The ``grpo-decomp`` console script, with the reference study registered."""

from __future__ import annotations

import sys

from grpo_decomp.eval import cli
from llm_grpo_gains.registration import register


def main(argv: list[str] | None = None) -> int:
    register()
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
