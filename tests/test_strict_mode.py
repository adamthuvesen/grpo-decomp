"""Suppressions are narrow: pyright ignores name a rule and a reason, and no noqa lifts a ban."""

import ast
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
            problems.append(
                (
                    token.start[0],
                    "file-level `# pyright:` directive is not allowed; remove it and fix the type error, or use an inline `# pyright: ignore[rule]  # reason`",
                )
            )
        elif not IGNORE_WITH_REASON.match(directive):
            problems.append(
                (
                    token.start[0],
                    "pyright ignore needs a rule and a reason; write `# pyright: ignore[ruleName]  # why`",
                )
            )
    return problems


FILE_WIDE_NOQA = re.compile(r"#\s*ruff:\s*noqa", re.IGNORECASE)
NOQA_ON_BAN = re.compile(r"#\s*noqa\b.*\bTID251", re.IGNORECASE)


def noqa_problems(source: str) -> list[tuple[int, str]]:
    problems: list[tuple[int, str]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        if FILE_WIDE_NOQA.search(token.string):
            problems.append(
                (
                    token.start[0],
                    "file-wide `# ruff: noqa` is not allowed; fix each finding, or put `# noqa: CODE` on the one line",
                )
            )
        elif NOQA_ON_BAN.search(token.string):
            problems.append(
                (
                    token.start[0],
                    "`noqa` must not lift a TID251 module ban; move the call into an owner file, or add the file to OWNERS in tests/test_banned_modules.py and to the TID251 per-file-ignores",
                )
            )
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


def test_no_noqa_lifts_a_ban_or_a_whole_file() -> None:
    problems = [
        f"{path.relative_to(ROOT)}:{line}: {message}"
        for path in checkout_python_files()
        for line, message in noqa_problems(path.read_text(errors="replace"))
    ]
    assert not problems, "\n".join(problems)


def test_noqa_detector_finds_known_cases() -> None:
    def lines(source: str) -> list[int]:
        return [line for line, _ in noqa_problems(source)]

    assert lines("# ruff: noqa\n") == [1]
    assert lines("# ruff: noqa: F401\n") == [1]
    assert lines("import subprocess  # noqa: TID251\n") == [1]
    assert lines("import os  # noqa: F401\n") == []
    assert lines("x = '# noqa: TID251'\n") == []


SKIP_CALLS = {"pytest.skip", "unittest.skip"}
SKIP_MARKS = {"pytest.mark.skip", "pytest.mark.xfail"}
SKIP_NAMES = SKIP_CALLS | SKIP_MARKS


def dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = dotted_name(node.value)
        return None if owner is None else f"{owner}.{node.attr}"
    return None


def has_reason(call: ast.Call, name: str) -> bool:
    keyword_reason = next((kw.value for kw in call.keywords if kw.arg == "reason"), None)
    reason = keyword_reason if name in SKIP_MARKS or not call.args else call.args[0]
    return (
        isinstance(reason, ast.Constant)
        and isinstance(reason.value, str)
        and bool(reason.value.strip())
    )


def skip_problems(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    problems: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name in SKIP_NAMES and not has_reason(node, name):
                problems.append(
                    (
                        node.lineno,
                        f'`{name}` needs a reason; pass a non-empty string, e.g. reason="needs a GPU"',
                    )
                )
        elif (
            isinstance(node, ast.Attribute | ast.Name)
            and id(node) not in called
            and dotted_name(node) in SKIP_NAMES
        ):
            problems.append(
                (
                    node.lineno,
                    f'bare `{dotted_name(node)}` needs a reason; call it with reason="why"',
                )
            )
    return sorted(problems)


def test_every_skip_names_a_reason() -> None:
    problems = [
        f"{path.relative_to(ROOT)}:{line}: {message}"
        for path in checkout_python_files()
        for line, message in skip_problems(path.read_text(errors="replace"))
    ]
    assert not problems, "\n".join(problems)


def test_skip_detector_finds_known_cases() -> None:
    def lines(source: str) -> list[int]:
        return [line for line, _ in skip_problems(source)]

    assert lines("@pytest.mark.skip\ndef test_a() -> None: ...\n") == [1]
    assert lines("@pytest.mark.xfail()\ndef test_b() -> None: ...\n") == [1]
    assert lines("pytest.skip()\n") == [1]
    assert lines("unittest.skip\n") == [1]
    assert lines('@pytest.mark.skip(reason="needs a GPU")\ndef test_c() -> None: ...\n') == []
    assert lines('pytest.skip("needs modal")\n') == []
    assert lines("x = '@pytest.mark.skip'\n") == []
