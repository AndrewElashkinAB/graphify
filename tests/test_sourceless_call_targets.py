"""Regression tests for calls whose bare name finds only a source-less stub.

An external name must not gain ``calls`` edges through its stub (#3156), and a
call to a project class must not be lost because the calling file also names
that class as a type.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphify.extract import extract


def _nodes(result: dict) -> dict[str, dict]:
    return {node["id"]: node for node in result["nodes"]}


def _labelled_edges(result: dict, nodes: dict[str, dict], label: str) -> list[dict]:
    return [
        edge
        for edge in result["edges"]
        if nodes.get(edge.get("target"), {}).get("label") == label
    ]


def test_sourceless_external_call_target_is_not_emitted(tmp_path: Path):
    """A type-reference stub must not become a project-wide calls hub."""
    typed = tmp_path / "typed.py"
    typed.write_text(
        "from fastapi import HTTPException\n"
        "\n"
        "def typed_error() -> HTTPException:\n"
        "    return HTTPException(status_code=400)\n",
        encoding="utf-8",
    )
    caller = tmp_path / "caller.py"
    caller.write_text(
        "def bare_error():\n"
        "    return HTTPException(status_code=401)\n",
        encoding="utf-8",
    )

    result = extract([typed, caller], cache_root=tmp_path, root=tmp_path)
    nodes = _nodes(result)
    external = [node for node in nodes.values() if node.get("label") == "HTTPException"]

    assert len(external) == 1
    assert external[0].get("source_file") == ""
    references = _labelled_edges(result, nodes, "HTTPException")
    assert any(edge.get("context") == "return_type" for edge in references)
    assert not any(edge.get("relation") == "calls" for edge in references)


def test_source_backed_exception_call_is_preserved(tmp_path: Path):
    """A project-defined exception with the same name remains a real call target."""
    definitions = tmp_path / "exceptions.py"
    definitions.write_text(
        "class HTTPException(Exception):\n"
        "    pass\n",
        encoding="utf-8",
    )
    caller = tmp_path / "api.py"
    caller.write_text(
        "from exceptions import HTTPException\n"
        "\n"
        "def raise_error():\n"
        "    return HTTPException()\n",
        encoding="utf-8",
    )

    result = extract([definitions, caller], cache_root=tmp_path, root=tmp_path)
    nodes = _nodes(result)
    calls = _labelled_edges(result, nodes, "HTTPException")

    assert any(
        edge.get("relation") == "calls"
        and nodes[edge["target"]].get("source_file") == "exceptions.py"
        for edge in calls
    ), calls


# A class named as a type in the calling file (a parameter or return annotation,
# a generic argument, a base class) mints a source-less stub for that name, and
# a bare ``Widget("a")`` elsewhere in the file finds that stub in the file's own
# label map. From one file a project class and an external name look the same,
# so the call must go on to cross-file resolution exactly as it would with no
# type use in the file: that pass binds a project class, applies its own guards
# (language family, import evidence, the #3156 source-less target check), and
# sets the confidence. Dropping the call at the stub lost it altogether.
#
# Each case extracts the caller twice, without and with the type use, and
# requires both to produce the same ``calls`` edges out of ``build``.

_WIDGET_PY = (
    "class Widget:\n"
    "    def __init__(self, name):\n"
    "        self.name = name\n"
)


def _write(root: Path, files: dict[str, str]) -> list[Path]:
    paths = []
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    return paths


def _calls_out_of_build(root: Path, files: dict[str, str]) -> list[tuple[str, str]]:
    """``(target source_file, confidence)`` for each ``calls`` edge out of ``build``."""
    paths = _write(root, files)
    result = extract(paths, cache_root=root, root=root, parallel=False)
    nodes = _nodes(result)
    return sorted(
        (nodes.get(edge["target"], {}).get("source_file", ""), edge.get("confidence", ""))
        for edge in result["edges"]
        if edge.get("relation") == "calls"
        and nodes.get(edge["source"], {}).get("label", "").strip(".()").lower() == "build"
    )


def _assert_type_use_does_not_change_the_call(
    tmp_path: Path,
    definitions: dict[str, str],
    caller: tuple[str, str, str, str],
    expected: list[tuple[str, str]],
) -> None:
    path, header, type_use, body = caller
    plain = _calls_out_of_build(tmp_path / "plain", {**definitions, path: header + body})
    typed = _calls_out_of_build(
        tmp_path / "typed", {**definitions, path: header + type_use + body}
    )

    assert plain == expected
    assert typed == plain


_PY_PACKAGE = {"app/__init__.py": "", "app/models.py": _WIDGET_PY}
_PY_BUILD = "def build():\n    return Widget(\"a\")\n"
_PY_DESCRIBE = "def describe(widget: Widget) -> str:\n    return widget.name\n\n"


@pytest.mark.parametrize(
    ("definitions", "caller", "expected"),
    [
        pytest.param(
            _PY_PACKAGE,
            ("app/use.py", "from app.models import *\n\n", _PY_DESCRIBE, _PY_BUILD),
            [("app/models.py", "EXTRACTED")],
            id="python-parameter-annotation",
        ),
        pytest.param(
            _PY_PACKAGE,
            (
                "app/use.py",
                "from app.models import *\n\n",
                "class FancyWidget(Widget):\n    pass\n\n",
                _PY_BUILD,
            ),
            [("app/models.py", "EXTRACTED")],
            id="python-base-class",
        ),
        pytest.param(
            {"Widget.cs": "namespace App { public class Widget { public Widget(string n) {} } }\n"},
            (
                "Use.cs",
                "namespace App { public class Use {\n",
                "  string Describe(Widget w) { return \"x\"; }\n",
                "  object Build() { return new Widget(\"a\"); }\n} }\n",
            ),
            [("Widget.cs", "INFERRED")],
            id="csharp",
        ),
        pytest.param(
            {"Widget.kt": "package app\nclass Widget(val name: String)\n"},
            (
                "Use.kt",
                "package app\n",
                "fun describe(w: Widget): String = w.name\n",
                "fun build(): Any = Widget(\"a\")\n",
            ),
            [("Widget.kt", "INFERRED")],
            id="kotlin",
        ),
        pytest.param(
            {"Widget.swift": "class Widget { init(name: String) {} }\n"},
            (
                "Use.swift",
                "",
                "func describe(w: Widget) -> String { return \"x\" }\n",
                "func build() -> Any { return Widget(name: \"a\") }\n",
            ),
            [("Widget.swift", "INFERRED")],
            id="swift",
        ),
        pytest.param(
            {"Widget.php": "<?php\nclass Widget { public function __construct($n) {} }\n"},
            (
                "Use.php",
                "<?php\n",
                "function describe(Widget $w): string { return 'x'; }\n",
                "function build() { return new Widget('a'); }\n",
            ),
            [("Widget.php", "INFERRED")],
            id="php",
        ),
        # No declared constructor: with one, the constructor node shares the
        # class's name, and the cross-file pass binds neither of the two
        # same-file candidates, for a plain caller as well (a separate gap).
        pytest.param(
            {"app/Widget.java": "package app;\npublic class Widget {}\n"},
            (
                "app/Use.java",
                "package app;\npublic class Use {\n",
                "  String describe(Widget w) { return \"x\"; }\n",
                "  Object build() { return new Widget(); }\n}\n",
            ),
            [("app/Widget.java", "INFERRED")],
            id="java-without-declared-constructor",
        ),
    ],
)
def test_type_use_in_the_calling_file_does_not_drop_the_call(
    tmp_path: Path,
    definitions: dict[str, str],
    caller: tuple[str, str, str, str],
    expected: list[tuple[str, str]],
):
    """The call bound to the type use's stub and was dropped as a #3156 target,
    so it never reached cross-file resolution; the plain caller kept its edge."""
    _assert_type_use_does_not_change_the_call(tmp_path, definitions, caller, expected)


@pytest.mark.parametrize(
    ("definitions", "caller", "expected"),
    [
        pytest.param(
            {"web/widget.ts": "export class Widget { constructor(name: string) {} }\n"},
            ("use.py", "", _PY_DESCRIBE, _PY_BUILD),
            [],
            id="python-caller-typescript-class",
        ),
        pytest.param(
            {"app/Component.java": "package app;\npublic class Component { public Component() {} }\n"},
            (
                "web/Use.java",
                "package web;\n"
                "import org.springframework.stereotype.Component;\n"
                "public class Use {\n",
                "  String describe(Component c) { return \"x\"; }\n",
                "  Object build() { return new Component(); }\n}\n",
            ),
            [],
            id="java-external-import-sharing-a-project-class-name",
        ),
        pytest.param(
            _PY_PACKAGE,
            ("app/use.py", "from app.models import Widget\n\n", _PY_DESCRIBE, _PY_BUILD),
            [("app/models.py", "EXTRACTED")],
            id="python-import-resolved-by-symbol-resolution",
        ),
    ],
)
def test_type_use_in_the_calling_file_does_not_add_a_call(
    tmp_path: Path,
    definitions: dict[str, str],
    caller: tuple[str, str, str, str],
    expected: list[tuple[str, str]],
):
    """Guards: the stub must not bind the call where cross-file resolution would
    refuse it (another language family, an external import that only shares a
    project class's simple name), nor duplicate a call another pass emitted."""
    _assert_type_use_does_not_change_the_call(tmp_path, definitions, caller, expected)
