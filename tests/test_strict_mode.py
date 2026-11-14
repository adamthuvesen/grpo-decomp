"""Pyright suppressions are inline only: each ignore names its rule and gives a reason."""

import io
import os
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def is_python_file(path: Path) -> bool:
    if path.suffix:
        return path.suffix == ".py"
    with path.open("rb") as handle:
        first_line = handle.readline()
    return first_line.startswith(b"#!") and b"python" in first_line


def checkout_python_files() -> list[Path]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = (ROOT / name for name in listing.split("\0") if name)
    return [path for path in paths if path.is_file() and is_python_file(path)]


PYRIGHT_COMMENT = re.compile(r"#\s*pyright:\s*(?P<directive>.*)")
IGNORE_WITH_REASON = re.compile(r"ignore\[[A-Za-z]+(?:,\s*[A-Za-z]+)*\]\s+#\s*\S")


def pyright_comment_problems(source: str) -> list[tuple[int, str]]:
    """Real comments only: a `# pyright:` inside a string is not a suppression."""
    problems: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        found = PYRIGHT_COMMENT.search(token.string) if token.type == tokenize.COMMENT else None
        if found is None:
            continue
        directive = found.group("directive")
        if not directive.startswith("ignore"):
            problems.append((token.start[0], "file-level `# pyright:` directive; fix the cause"))
        elif not IGNORE_WITH_REASON.match(directive):
            problems.append((token.start[0], "`# pyright: ignore[rule]  # reason` needs both"))
    return problems


def test_pyright_comments_are_inline_ignores_with_a_rule_and_a_reason() -> None:
    problems = [
        f"{path.relative_to(ROOT)}:{line}: {message}"
        for path in checkout_python_files()
        for line, message in pyright_comment_problems(path.read_text(errors="replace"))
    ]
    assert not problems, "\n".join(problems)


def test_pyright_comment_detector_finds_known_cases() -> None:
    def lines(source: str) -> list[int]:
        return [line for line, _ in pyright_comment_problems(source)]

    assert lines("# pyright: basic\n") == [1]
    assert lines("# pyright: reportMissingImports=false\n") == [1]
    assert lines("x = 1  # pyright: ignore\n") == [1]
    assert lines("x = 1  # pyright: ignore[reportAny]\n") == [1]
    assert lines("x = 1  # noqa: F401  # pyright: ignore[reportAny]\n") == [1]
    assert lines("x = 1  # pyright: ignore[reportAny]  # runtime proof\n") == []
    assert lines("x = 1  # pyright: ignore[reportAny, reportCallIssue]  # why\n") == []
    assert lines("x = '# pyright: basic'\n") == []
