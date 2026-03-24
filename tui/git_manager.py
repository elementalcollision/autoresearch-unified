"""Git operations for the experiment orchestrator."""

import subprocess
from pathlib import Path


class GitManager:
    """Manages git operations for the experiment loop.

    All operations run synchronously (designed to be called from the
    orchestrator's background thread, not the Textual event loop).
    """

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

    def ensure_auto_push_remote(self) -> None:
        """Configure git to auto-set upstream on push for new branches.

        Without this, new per-dataset experiment branches created by the
        orchestrator have no upstream tracking, causing the sync script's
        `git push` to silently fail. This was a recurring data loss issue
        across every RunPod deployment.
        """
        self._run("config", "push.autoSetupRemote", "true", check=False)

    def current_branch(self) -> str:
        """Get the current branch name."""
        return self._run("rev-parse", "--abbrev-ref", "HEAD")

    def current_commit(self) -> str:
        """Get the current short commit hash."""
        return self._run("rev-parse", "--short=7", "HEAD")

    def branch_exists(self, branch: str) -> bool:
        """Check if a branch already exists (local or remote)."""
        result = self._run("branch", "--list", branch, check=False)
        if result:
            return True
        result = self._run("branch", "-r", "--list", f"origin/{branch}", check=False)
        return bool(result)

    def create_branch(self, branch_name: str) -> str:
        """Create and checkout a new branch. Returns branch name."""
        self._run("checkout", "-b", branch_name)
        return branch_name

    def checkout(self, branch: str) -> None:
        """Checkout an existing branch."""
        self._run("checkout", branch)

    def commit_changes(self, message: str, files: list[str]) -> str:
        """Stage files and commit. Returns short commit hash."""
        for f in files:
            self._run("add", f)
        self._run("commit", "-m", message)
        return self.current_commit()

    def revert_last_experiment(self) -> None:
        """Revert the most recent experiment commit without touching other files.

        Finds the most recent commit matching the pattern 'expN: ...' and
        reverts only the files it changed. Skips over any intervening sync
        commits (from sync_results.sh) that may have landed between the
        experiment commit and the crash/discard evaluation.

        Previous approaches failed because:
        - `git reset --hard HEAD~1` wiped ALL files including results TSVs
        - Soft reset + restore from HEAD assumed HEAD was the experiment
          commit, but the sync script could push a commit in between,
          making HEAD a sync commit whose diff includes the results TSV
        """
        import re as _re

        # Walk back from HEAD to find the experiment commit
        # (format: "expN: description")
        log_lines = self._run(
            "log", "--oneline", "-10"
        ).strip().splitlines()

        exp_sha = None
        for line in log_lines:
            sha, _, msg = line.partition(" ")
            if _re.match(r"exp\d+:", msg):
                exp_sha = sha
                break

        if not exp_sha:
            # No experiment commit found — nothing to revert
            return

        # Get files changed in the experiment commit
        changed_files = self._run(
            "diff-tree", "--no-commit-id", "--name-only", "-r", exp_sha
        ).strip().splitlines()

        # Restore each file to its state BEFORE the experiment commit
        # (i.e., the parent of the experiment commit)
        parent = exp_sha + "~1"
        for f in changed_files:
            try:
                self._run("show", f"{parent}:{f}")  # verify it exists
                self._run("checkout", parent, "--", f)
            except RuntimeError:
                # File didn't exist before this commit — remove it
                full_path = self._repo / f
                if full_path.exists():
                    full_path.unlink()

        # Stage the restored files and commit the revert
        # (keeps git history linear and clean)
        for f in changed_files:
            try:
                self._run("add", f)
            except RuntimeError:
                pass

        try:
            self._run(
                "commit", "--allow-empty", "-m",
                f"Revert {exp_sha[:7]} (discard/crash)"
            )
        except RuntimeError:
            # Nothing to commit (files unchanged) — that's fine
            pass

    def read_file(self, path: str) -> str:
        """Read a file from the working tree."""
        full_path = self._repo / path
        return full_path.read_text()

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the working tree."""
        full_path = self._repo / path
        full_path.write_text(content)

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        status = self._run("status", "--porcelain", check=False)
        # Filter out untracked files like run.log
        for line in status.splitlines():
            if not line.startswith("??"):
                return True
        return False

    def reset_working_tree(self, file_path: str) -> None:
        """Restore a single file to its committed state."""
        self._run("checkout", "--", file_path)

    def head_commit_message(self) -> str:
        """Get the commit message of HEAD."""
        return self._run("log", "--format=%s", "-1")

    # ------------------------------------------------------------------
    # Baseline tracking
    # ------------------------------------------------------------------

    def record_baseline(self) -> str:
        """Record the current HEAD as the baseline commit (zero HP changes).

        Should be called AFTER the baseline training run completes,
        before any experiment modifications. Returns the full SHA.
        """
        self._baseline_sha = self._run("rev-parse", "HEAD")
        return self._baseline_sha

    @property
    def baseline_sha(self) -> str | None:
        """Return the recorded baseline commit SHA, or None if not set."""
        return self._baseline_sha

    def restore_baseline_file(self, file_path: str) -> None:
        """Restore a single file to its baseline state (zero HP modifications).

        Uses `git show <baseline_sha>:<path>` to read the original content
        and writes it to the working tree. This ensures every experiment
        starts from the same unmodified training script.
        """
        if not self._baseline_sha:
            raise RuntimeError(
                "No baseline commit recorded. Call record_baseline() "
                "after the baseline training run."
            )
        # Get the repo-relative path for git show
        abs_path = (self._repo / file_path).resolve()
        try:
            rel_path = abs_path.relative_to(self._repo)
        except ValueError:
            rel_path = Path(file_path)

        content = self._run("show", f"{self._baseline_sha}:{rel_path}")
        abs_path.write_text(content + "\n" if not content.endswith("\n") else content)
