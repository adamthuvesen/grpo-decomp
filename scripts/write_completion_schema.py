"""Rewrite the committed CompletionSet JSON Schema from the Pydantic models."""

import json
from pathlib import Path

from grpo_decomp.eval.completions import completion_set_json_schema

SCHEMA_PATH = Path(__file__).parents[1] / "docs" / "schemas" / "completion-set.schema.json"

if __name__ == "__main__":
    SCHEMA_PATH.write_text(
        json.dumps(completion_set_json_schema(), indent=2) + "\n", encoding="utf-8"
    )
