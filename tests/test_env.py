"""Provenance: the git state a run records (P1-M6.3)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from floodfusion.utils.env import code_changed_since, environment_record, git_state

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "table.md").write_text("v1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("doc\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_tree(repo) -> None:
    state = git_state(repo)
    assert len(state["git_sha"]) == 40 and state["git_dirty"] is False


def test_regenerated_reports_do_not_make_the_tree_dirty(repo) -> None:
    (repo / "reports" / "table.md").write_text("v2\n")
    assert git_state(repo)["git_dirty"] is False


def test_documentation_changes_do_not_either(repo) -> None:
    (repo / "README.md").write_text("more doc\n")
    assert git_state(repo)["git_dirty"] is False


def test_code_changes_do(repo) -> None:
    (repo / "src" / "code.py").write_text("x = 2\n")
    assert git_state(repo)["git_dirty"] is True


def test_a_docs_only_commit_keeps_old_runs_reproducible(repo) -> None:
    old = git_state(repo)["git_sha"]
    (repo / "README.md").write_text("results\n")
    _git(repo, "commit", "-qam", "docs")
    assert not code_changed_since(old, repo)
    (repo / "src" / "code.py").write_text("x = 3\n")
    _git(repo, "commit", "-qam", "code")
    assert code_changed_since(old, repo)


def test_outside_a_repository(tmp_path) -> None:
    state = git_state(tmp_path)
    assert state["git_sha"] is None and state["git_dirty"] is True


def test_environment_record_has_versions() -> None:
    record = environment_record()
    assert record["python"] and "git_sha" in record
