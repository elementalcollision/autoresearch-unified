# Ported from autoresearch-unified/tui/git_manager.py (MIT)
"""Git operations for the experiment orchestrator."""

import subprocess
from pathlib import Path


class GitManager:
    """Manages git operations for the experiment loop."""

    def __init__(self, repo_path: str = "."):
        self._repo = Path(repo_path).resolve()
        self._baseline_sha: str | None = None

    def _run(self, *args: str, check: bool = True) -> str:
        """Run a git command and return stdout."""
        result = subprocess.run(
            ["git"] + list(args),
            cwd=self._repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result.stdout.strip()

    def current_branch(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "HEAD")

    def current_commit(self) -> str:
        return self._run("rev-parse", "--short=7", "HEAD")

    def branch_exists(self, branch: str) -> bool:
        result = self._run("branch", "--list", branch, check=False)
        if result:
            return True
        result = self._run("branch", "-r", "--list", f"origin/{branch}", check=False)
        return bool(result)

    def create_branch(self, branch_name: str) -> str:
        self._run("checkout", "-b", branch_name)
        return branch_name

    def checkout(self, branch: str) -> None:
        self._run("checkout", branch)

    def commit_changes(self, message: str, files: list[str]) -> str:
        for f in files:
            self._run("add", f)
        self._run("commit", "-m", message)
        return self.current_commit()

    def revert_last_experiment(self) -> None:
        """Revert the most recent experiment commit."""
        import re as _re

        log_lines = self._run("log", "--oneline", "-10").strip().splitlines()
        exp_sha = None
        for line in log_lines:
            sha, _, msg = line.partition(" ")
            if _re.match(r"exp\d+:", msg):
                exp_sha = sha
                break

        if not exp_sha:
            return

        changed_files = self._run(
            "diff-tree", "--no-commit-id", "--name-only", "-r", exp_sha
        ).strip().splitlines()

        parent = exp_sha + "~1"
        for f in changed_files:
            try:
                self._run("show", f"{parent}:{f}")
                self._run("checkout", parent, "--", f)
            except RuntimeError:
                full_path = self._repo / f
                if full_path.exists():
                    full_path.unlink()

        for f in changed_files:
            try:
                self._run("add", f)
            except RuntimeError:
                pass

        try:
            self._run("commit", "--allow-empty", "-m", f"Revert {exp_sha[:7]} (discard/crash)")
        except RuntimeError:
            pass

    def read_file(self, path: str) -> str:
        return (self._repo / path).read_text()

    def write_file(self, path: str, content: str) -> None:
        (self._repo / path).write_text(content)

    def has_uncommitted_changes(self) -> bool:
        status = self._run("status", "--porcelain", check=False)
        for line in status.splitlines():
            if not line.startswith("??"):
                return True
        return False

    def record_baseline(self) -> str:
        self._baseline_sha = self._run("rev-parse", "HEAD")
        return self._baseline_sha

    @property
    def baseline_sha(self) -> str | None:
        return self._baseline_sha

    def restore_baseline_file(self, file_path: str) -> None:
        if not self._baseline_sha:
            raise RuntimeError("No baseline commit recorded.")
        abs_path = (self._repo / file_path).resolve()
        try:
            rel_path = abs_path.relative_to(self._repo)
        except ValueError:
            rel_path = Path(file_path)
        content = self._run("show", f"{self._baseline_sha}:{rel_path}")
        abs_path.write_text(content + "\n" if not content.endswith("\n") else content)
