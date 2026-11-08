"""Core modules must not import a UI, read the environment, write files or contain Spanish text."""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "glaurlex"
MODULES = sorted(SRC.rglob("*.py"))

FORBIDDEN_IMPORTS = {"streamlit", "shutil", "tempfile", "pickle", "sqlite3"}
FORBIDDEN_CALLS = {
    "to_parquet", "read_parquet", "to_csv", "read_csv", "to_excel", "read_excel", "to_json",
    "write_gml", "read_gml", "write_text", "write_bytes", "read_text", "read_bytes",
    "mkdir", "unlink", "rmdir", "getenv",
}  # fmt: skip
READ_ONLY_OPEN = {"io.py"}


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            yield node, func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_module_is_free_of_ui_environment_and_file_writes(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")]
        )
    }
    assert not imported & FORBIDDEN_IMPORTS
    assert "environ" not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    for call, name in _calls(tree):
        assert name not in FORBIDDEN_CALLS, f"{path.name}: {name}()"
        if name == "open":
            assert path.name in READ_ONLY_OPEN, f"{path.name}: open()"
            mode = call.args[1] if len(call.args) > 1 else None
            assert isinstance(mode, ast.Constant)
            assert mode.value == "rb"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_source_is_english(path):
    text = path.read_text(encoding="utf-8")
    for citation in ("López Chávez", "Frías"):
        text = text.replace(citation, "")
    assert not re.findall(r"[ñáéíóúü¿¡]", text, flags=re.IGNORECASE)
