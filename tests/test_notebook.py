"""The walkthrough notebook runs top to bottom, offline, on CPU (the code path of every stage)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "walkthrough.ipynb"


def test_walkthrough_executes_without_errors(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    assert not [o for o in outputs if o.get("output_type") == "error"]
    assert any("test macro IoU" in o.get("text", "") for o in outputs)


def test_committed_notebook_is_generated_from_its_builder() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_walkthrough", NOTEBOOK.with_name("build_walkthrough.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells]
    assert committed == [src for _, src in module.CELLS]
