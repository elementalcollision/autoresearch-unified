"""Tests for tui.git_manager — git operations and experiment revert logic."""

import subprocess
from pathlib import Path
from unittest.mock import call, patch, MagicMock

import pytest

from tui.git_manager import GitManager


# ---------------------------------------------------------------------------
# GitManager._run — low-level subprocess layer
# ---------------------------------------------------------------------------

class TestRun:
    def test_successful_command(self):
        gm = GitManager("/fake/repo")
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="main\n", stderr=""
        )
        with patch("tui.git_manager.subprocess.run", return_value=fake_result):
            assert gm._run("rev-parse", "--abbrev-ref", "HEAD") == "main"

    def test_failed_command_raises(self):
        gm = GitManager("/fake/repo")
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="fatal: not a git repo"
        )
        with patch("tui.git_manager.subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="not a git repo"):
                gm._run("status")

    def test_check_false_no_raise(self):
        gm = GitManager("/fake/repo")
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="output\n", stderr="err"
        )
        with patch("tui.git_manager.subprocess.run", return_value=fake_result):
            assert gm._run("status", check=False) == "output"

    def test_timeout_is_30s(self):
        gm = GitManager("/fake/repo")
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="\n", stderr=""
        )
        with patch("tui.git_manager.subprocess.run", return_value=fake_result) as mock_run:
            gm._run("status")
            assert mock_run.call_args[1]["timeout"] == 30


# ---------------------------------------------------------------------------
# Branch operations
# ---------------------------------------------------------------------------

class TestBranchOperations:
    def test_current_branch(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run", return_value="experiments/pubmed"):
            assert gm.current_branch() == "experiments/pubmed"

    def test_branch_exists_local(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run", return_value="  experiments/pubmed"):
            assert gm.branch_exists("experiments/pubmed") is True

    def test_branch_exists_remote(self):
        gm = GitManager("/fake/repo")
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # First call: local branch check returns empty
            if call_count[0] == 1:
                return ""
            # Second call: remote branch check returns match
            return "  origin/experiments/pubmed"

        with patch.object(gm, "_run", side_effect=side_effect):
            assert gm.branch_exists("experiments/pubmed") is True

    def test_branch_not_exists(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run", return_value=""):
            assert gm.branch_exists("nonexistent") is False

    def test_create_branch(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run") as mock_run:
            result = gm.create_branch("new-branch")
            mock_run.assert_called_once_with("checkout", "-b", "new-branch")
            assert result == "new-branch"


# ---------------------------------------------------------------------------
# commit_changes
# ---------------------------------------------------------------------------

class TestCommitChanges:
    def test_stages_and_commits(self):
        gm = GitManager("/fake/repo")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(args)
            return "abc1234"

        with patch.object(gm, "_run", side_effect=fake_run):
            gm.commit_changes("exp1: test", ["train.py", "config.py"])

        # Should add each file, then commit, then get current commit
        assert calls[0] == ("add", "train.py")
        assert calls[1] == ("add", "config.py")
        assert calls[2] == ("commit", "-m", "exp1: test")

    def test_returns_commit_hash(self):
        gm = GitManager("/fake/repo")
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            # current_commit() is the last call
            if args[0] == "rev-parse":
                return "abc1234"
            return ""

        with patch.object(gm, "_run", side_effect=fake_run):
            result = gm.commit_changes("exp1: test", ["train.py"])
            assert result == "abc1234"


# ---------------------------------------------------------------------------
# has_uncommitted_changes
# ---------------------------------------------------------------------------

class TestHasUncommittedChanges:
    def test_clean_status(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run", return_value=""):
            assert gm.has_uncommitted_changes() is False

    def test_modified_file(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run", return_value=" M train.py"):
            assert gm.has_uncommitted_changes() is True

    def test_untracked_ignored(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run", return_value="?? run.log"):
            assert gm.has_uncommitted_changes() is False

    def test_mixed_status(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run", return_value="?? run.log\n M train.py"):
            assert gm.has_uncommitted_changes() is True


# ---------------------------------------------------------------------------
# revert_last_experiment — tests the dacda45 bugfix
# ---------------------------------------------------------------------------

class TestRevertLastExperiment:
    def test_finds_experiment_commit(self):
        gm = GitManager("/fake/repo")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(args)
            if args[0] == "log":
                return "abc1234 exp3: increase LR"
            if args[0] == "diff-tree":
                return "train.py"
            return ""

        with patch.object(gm, "_run", side_effect=fake_run):
            gm.revert_last_experiment()

        # Should have looked up diff-tree for abc1234
        diff_call = [c for c in calls if c[0] == "diff-tree"]
        assert len(diff_call) == 1
        assert "abc1234" in diff_call[0]

    def test_skips_sync_commits(self):
        """Sync commits (from sync_results.sh) should be skipped — only exp commits targeted."""
        gm = GitManager("/fake/repo")
        found_sha = []

        def fake_run(*args, **kwargs):
            if args[0] == "log":
                # Sync commit on top, then experiment commit
                return "def5678 sync results\nabc1234 exp3: increase LR"
            if args[0] == "diff-tree":
                found_sha.append(args[-1])  # capture which SHA was queried
                return "train.py"
            return ""

        with patch.object(gm, "_run", side_effect=fake_run):
            gm.revert_last_experiment()

        # Should target abc1234 (exp commit), not def5678 (sync commit)
        assert found_sha[0] == "abc1234"

    def test_restores_changed_files(self):
        gm = GitManager("/fake/repo")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(args)
            if args[0] == "log":
                return "abc1234 exp3: increase LR"
            if args[0] == "diff-tree":
                return "train.py"
            if args[0] == "show" and "abc1234~1:train.py" in args[1]:
                return "original content"
            return ""

        with patch.object(gm, "_run", side_effect=fake_run):
            gm.revert_last_experiment()

        # Should checkout the file from parent commit
        checkout_calls = [c for c in calls if c[0] == "checkout" and "--" in c]
        assert len(checkout_calls) == 1
        assert "abc1234~1" in checkout_calls[0]
        assert "train.py" in checkout_calls[0]

    def test_deletes_new_files(self):
        """Files that didn't exist before the experiment should be removed."""
        gm = GitManager("/fake/repo")

        def fake_run(*args, **kwargs):
            if args[0] == "log":
                return "abc1234 exp3: add new file"
            if args[0] == "diff-tree":
                return "new_file.py"
            if args[0] == "show" and "abc1234~1:new_file.py" in args[1]:
                raise RuntimeError("path not found")
            if args[0] == "add":
                return ""
            if args[0] == "commit":
                return ""
            return ""

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        with patch.object(gm, "_run", side_effect=fake_run), \
             patch.object(Path, "__truediv__", return_value=mock_path):
            # Override _repo / f to return our mock_path
            gm._repo = MagicMock()
            gm._repo.__truediv__ = MagicMock(return_value=mock_path)
            gm.revert_last_experiment()

        mock_path.unlink.assert_called_once()

    def test_no_experiment_commit_noop(self):
        gm = GitManager("/fake/repo")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(args)
            if args[0] == "log":
                return "def5678 sync results\nghi9012 initial commit"
            return ""

        with patch.object(gm, "_run", side_effect=fake_run):
            gm.revert_last_experiment()

        # Should only have the log call — no diff-tree, checkout, etc.
        assert len(calls) == 1
        assert calls[0][0] == "log"

    def test_commits_revert(self):
        gm = GitManager("/fake/repo")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(args)
            if args[0] == "log":
                return "abc1234 exp3: increase LR"
            if args[0] == "diff-tree":
                return "train.py"
            return ""

        with patch.object(gm, "_run", side_effect=fake_run):
            gm.revert_last_experiment()

        commit_calls = [c for c in calls if c[0] == "commit"]
        assert len(commit_calls) == 1
        assert "--allow-empty" in commit_calls[0]
        assert "Revert abc1234" in commit_calls[0][-1]


# ---------------------------------------------------------------------------
# Baseline tracking
# ---------------------------------------------------------------------------

class TestBaselineRestore:
    def test_no_baseline_raises(self):
        gm = GitManager("/fake/repo")
        with pytest.raises(RuntimeError, match="No baseline commit"):
            gm.restore_baseline_file("train.py")

    def test_record_baseline(self):
        gm = GitManager("/fake/repo")
        with patch.object(gm, "_run", return_value="abc123def456"):
            sha = gm.record_baseline()
            assert sha == "abc123def456"
            assert gm.baseline_sha == "abc123def456"

    def test_restore_calls_git_show(self, tmp_path):
        gm = GitManager(str(tmp_path))
        gm._baseline_sha = "abc123"
        # Create the file so write_text works
        target = tmp_path / "train.py"
        target.touch()

        with patch.object(gm, "_run", return_value="original content") as mock_run:
            gm.restore_baseline_file("train.py")

        # Should have called git show with baseline SHA
        mock_run.assert_called_once()
        show_args = mock_run.call_args[0]
        assert show_args[0] == "show"
        assert "abc123:" in show_args[1]
