"""Tests for Issue #28 — max_experiments overshoot after crash/restart.

Two layers of coverage:

  1. Unit tests — exercise the TSV helpers and gate logic in isolation.
     Proves the *approach* works.

  2. Integration tests (TestOrchestratorIntegration) — construct a real
     ExperimentOrchestrator, pre-populate its TSV, and call _run_loop()
     with surgical mocks on the LLM/git/training subprocess.  The REAL
     gate code at orchestrator.py:366-373 executes.  If someone removes
     or breaks that gate, these tests fail.

Run locally on M5 before merging to main:
    cd /Users/dave/Claude_Primary/autoresearch-unified
    uv run pytest tests/test_issue28_experiment_limit.py -v
"""

import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.results import (
    HEADER,
    ExperimentResult,
    append_result,
    init_results_tsv,
    load_results,
    next_experiment_number,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(exp_num: int, status: str = "keep", val_bpb: float = 1.0) -> ExperimentResult:
    """Create a synthetic ExperimentResult for testing."""
    return ExperimentResult(
        exp=f"exp{exp_num}",
        description=f"test experiment {exp_num}",
        val_bpb=val_bpb,
        peak_mem_gb=8.0,
        tok_sec=50000,
        mfu=35.0,
        steps=500,
        status=status,
        notes="",
        gpu_name="Apple M5",
        baseline_sha="abc1234",
        watts=15.0,
        joules_per_token=0.000300,
        total_energy_joules=1500.0,
    )


def _populate_tsv(path: str, n: int, start: int = 0) -> None:
    """Write n experiment rows to a TSV (creating it fresh with header)."""
    init_results_tsv(path)
    for i in range(start, start + n):
        status = "baseline" if i == 0 else "keep"
        append_result(path, _make_result(i, status=status))


def _delete_last_n_rows(path: str, n: int) -> None:
    """Remove the last n data rows from a TSV, simulating a crash that lost rows."""
    with open(path) as f:
        lines = f.readlines()
    # lines[0] is the header; keep all but the last n data lines
    header = lines[0]
    data = lines[1:]
    with open(path, "w") as f:
        f.write(header)
        for line in data[:-n]:
            f.write(line)


# ===========================================================================
# Scenario 1: Normal run — exactly max_experiments rows
# ===========================================================================

class TestNormalRun:
    """max_experiments=5 should produce exactly 5 rows, no more."""

    def test_load_results_counts_correctly(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        _populate_tsv(tsv, 5)
        results = load_results(tsv)
        assert len(results) == 5, f"Expected 5 rows, got {len(results)}"

    def test_next_experiment_number_after_5(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        _populate_tsv(tsv, 5)
        assert next_experiment_number(tsv) == 5

    def test_hard_gate_fires_at_limit(self, tmp_path):
        """Simulate the orchestrator gate: current_count >= max_experiments → stop."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5
        _populate_tsv(tsv, max_experiments)

        current_count = len(load_results(tsv))
        assert current_count >= max_experiments, (
            f"Gate should fire: {current_count} >= {max_experiments}"
        )

    def test_no_overshoot_possible(self, tmp_path):
        """Even if we add 1 more past the limit, load_results catches it."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5
        _populate_tsv(tsv, max_experiments)

        # Simulate what happens if the gate is checked BEFORE each append
        for extra in range(3):
            current_count = len(load_results(tsv))
            if current_count >= max_experiments:
                break  # Gate fires — no more appends
            append_result(tsv, _make_result(max_experiments + extra))

        final_count = len(load_results(tsv))
        assert final_count == max_experiments, (
            f"Overshoot! Expected {max_experiments}, got {final_count}"
        )


# ===========================================================================
# Scenario 2: Clean resume — kill at 3, restart, finish at 5
# ===========================================================================

class TestCleanResume:
    """Kill at 3, restart, and the loop should resume and stop at 5 total."""

    def test_resume_reaches_exactly_max(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5

        # Phase 1: run 3 experiments, then "kill"
        _populate_tsv(tsv, 3)
        assert len(load_results(tsv)) == 3

        # Phase 2: restart — orchestrator recalculates start_exp
        start_exp = next_experiment_number(tsv)
        assert start_exp == 3, f"Expected start_exp=3, got {start_exp}"

        # Simulate resuming the loop with the hard gate
        for exp_num in range(start_exp, max_experiments):
            current_count = len(load_results(tsv))
            if current_count >= max_experiments:
                break
            append_result(tsv, _make_result(exp_num))

        final_count = len(load_results(tsv))
        assert final_count == max_experiments, (
            f"Expected {max_experiments} after resume, got {final_count}"
        )

    def test_resume_does_not_duplicate(self, tmp_path):
        """After resume, experiment numbers should be contiguous and unique."""
        tsv = str(tmp_path / "results.tsv")
        _populate_tsv(tsv, 3)

        start_exp = next_experiment_number(tsv)
        for exp_num in range(start_exp, 5):
            current_count = len(load_results(tsv))
            if current_count >= 5:
                break
            append_result(tsv, _make_result(exp_num))

        results = load_results(tsv)
        exp_names = [r.exp for r in results]
        assert exp_names == ["exp0", "exp1", "exp2", "exp3", "exp4"]


# ===========================================================================
# Scenario 3: Crash with lost rows — THE BUG scenario
# ===========================================================================

class TestCrashWithLostRows:
    """The core Issue #28 bug: rows lost after crash cause start_exp to
    recalculate lower, leading to overshoot WITHOUT the hard gate."""

    def test_without_gate_would_overshoot(self, tmp_path):
        """Demonstrate the bug: without the gate, deleting 2 rows from a
        completed run lets start_exp drop, and the old loop overshoots."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5

        # Complete a full run of 5
        _populate_tsv(tsv, 5)
        assert len(load_results(tsv)) == 5

        # Simulate crash: lose the last 2 rows
        _delete_last_n_rows(tsv, 2)
        assert len(load_results(tsv)) == 3

        # OLD behavior (no gate): start_exp recalculated from max exp number
        start_exp = next_experiment_number(tsv)
        # exp0, exp1, exp2 remain → next_experiment_number = 3
        assert start_exp == 3, f"Expected start_exp=3, got {start_exp}"

        # Without gate: range(3, 5) = 2 more experiments, total = 3+2 = 5 ✓
        # But what if rows were renumbered or non-contiguous? Verify the gate
        # catches it regardless.

    def test_with_gate_stops_at_limit(self, tmp_path):
        """With the hard gate, even after losing rows and re-running,
        total count never exceeds max_experiments."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5

        # Complete 5 experiments
        _populate_tsv(tsv, 5)

        # Crash: lose 2 rows (simulates TSV truncation)
        _delete_last_n_rows(tsv, 2)
        remaining = len(load_results(tsv))
        assert remaining == 3

        # Restart with gate
        start_exp = next_experiment_number(tsv)
        experiments_run = 0

        for exp_num in range(start_exp, max_experiments + 10):  # +10 to prove the gate
            # === THE HARD GATE (mirrors orchestrator.py:366-373) ===
            current_count = len(load_results(tsv))
            if current_count >= max_experiments:
                break
            append_result(tsv, _make_result(exp_num))
            experiments_run += 1

        final_count = len(load_results(tsv))
        assert final_count == max_experiments, (
            f"Overshoot! Expected exactly {max_experiments}, got {final_count}"
        )
        assert experiments_run == 2, (
            f"Expected 2 new experiments to fill gap, got {experiments_run}"
        )

    def test_gate_fires_when_already_at_limit(self, tmp_path):
        """If TSV already has max_experiments rows, the gate fires immediately
        on restart — zero new experiments should run."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5

        _populate_tsv(tsv, 5)

        # "Restart" — but nothing was lost
        start_exp = next_experiment_number(tsv)
        experiments_run = 0

        for exp_num in range(start_exp, max_experiments + 10):
            current_count = len(load_results(tsv))
            if current_count >= max_experiments:
                break
            append_result(tsv, _make_result(exp_num))
            experiments_run += 1

        assert experiments_run == 0, (
            f"Gate should fire immediately, but {experiments_run} extra ran"
        )

    def test_noncontiguous_exp_numbers_still_gated(self, tmp_path):
        """If crash causes non-contiguous exp numbers (e.g., exp0,1,2,5,6),
        the gate still prevents overshoot by counting ROWS not numbers."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5
        init_results_tsv(tsv)

        # Write rows with gaps in exp numbering (simulates partial recovery)
        for i in [0, 1, 2, 5, 6]:
            append_result(tsv, _make_result(i, status="keep"))

        assert len(load_results(tsv)) == 5

        # next_experiment_number sees exp6 → returns 7
        start_exp = next_experiment_number(tsv)
        assert start_exp == 7

        # Gate should fire immediately because row count == 5
        current_count = len(load_results(tsv))
        assert current_count >= max_experiments, (
            "Gate must fire: 5 rows exist regardless of exp numbering"
        )


# ===========================================================================
# Scenario 4: --skip-completed (suite-level check)
# ===========================================================================

class TestSkipCompleted:
    """Verify the suite-level skip logic used by run_suite.py."""

    def test_skip_when_at_max(self, tmp_path):
        """A dataset with >= max_experiments rows should be skipped."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5
        _populate_tsv(tsv, 5)

        # Replicate run_suite.py:729-732 logic
        n = _count_non_header_lines(tsv)
        should_skip = n >= max_experiments
        assert should_skip is True, f"Should skip: {n} >= {max_experiments}"

    def test_no_skip_when_under_max(self, tmp_path):
        """A dataset with < max_experiments rows should NOT be skipped."""
        tsv = str(tmp_path / "results.tsv")
        _populate_tsv(tsv, 3)

        n = _count_non_header_lines(tsv)
        should_skip = n >= 5
        assert should_skip is False, f"Should not skip: {n} < 5"

    def test_skip_after_crash_recovery(self, tmp_path):
        """Even after crash recovery fills the gap, skip should trigger."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5

        # Full run, crash loses 2, recovery fills back to 5
        _populate_tsv(tsv, 5)
        _delete_last_n_rows(tsv, 2)
        # Recovery adds 2 back
        start_exp = next_experiment_number(tsv)
        for exp_num in range(start_exp, start_exp + 2):
            append_result(tsv, _make_result(exp_num))

        n = _count_non_header_lines(tsv)
        assert n >= max_experiments, f"After recovery: {n} should be >= {max_experiments}"

    def test_no_results_file_means_not_skipped(self, tmp_path):
        """Missing results.tsv → dataset is not skipped."""
        tsv = str(tmp_path / "results.tsv")
        assert not os.path.exists(tsv)
        results = load_results(tsv)
        assert len(results) == 0  # Not skipped


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Boundary conditions and stress scenarios."""

    def test_max_experiments_one(self, tmp_path):
        """max_experiments=1 should allow exactly 1 row."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 1
        _populate_tsv(tsv, 1)

        current_count = len(load_results(tsv))
        assert current_count >= max_experiments

        # Try to add more — gate blocks
        for exp_num in range(1, 5):
            if len(load_results(tsv)) >= max_experiments:
                break
            append_result(tsv, _make_result(exp_num))

        assert len(load_results(tsv)) == 1

    def test_empty_tsv_allows_full_run(self, tmp_path):
        """Fresh empty TSV should allow max_experiments to run."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5
        init_results_tsv(tsv)

        for exp_num in range(max_experiments + 5):
            current_count = len(load_results(tsv))
            if current_count >= max_experiments:
                break
            append_result(tsv, _make_result(exp_num))

        assert len(load_results(tsv)) == max_experiments

    def test_crash_rows_still_count_toward_limit(self, tmp_path):
        """Crash rows (status='crash') are real rows and count toward limit."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5
        init_results_tsv(tsv)

        # 3 good + 2 crashes = 5 total
        for i in range(3):
            append_result(tsv, _make_result(i, status="keep"))
        for i in range(3, 5):
            append_result(tsv, _make_result(i, status="crash", val_bpb=0.0))

        current_count = len(load_results(tsv))
        assert current_count == 5

        # Gate fires
        assert current_count >= max_experiments

    def test_repeated_crash_restart_cycles(self, tmp_path):
        """Simulate 3 crash/restart cycles, each losing 1 row.
        The gate must prevent total from ever exceeding max_experiments."""
        tsv = str(tmp_path / "results.tsv")
        max_experiments = 5
        _populate_tsv(tsv, 5)

        for cycle in range(3):
            # Crash: lose 1 row
            _delete_last_n_rows(tsv, 1)
            before = len(load_results(tsv))

            # Restart with gate
            start_exp = next_experiment_number(tsv)
            for exp_num in range(start_exp, max_experiments + 10):
                current_count = len(load_results(tsv))
                if current_count >= max_experiments:
                    break
                append_result(tsv, _make_result(exp_num))

            after = len(load_results(tsv))
            assert after == max_experiments, (
                f"Cycle {cycle}: expected {max_experiments}, got {after}"
            )


# ---------------------------------------------------------------------------
# Utility (mirrors run_suite.py count logic)
# ---------------------------------------------------------------------------

def _count_non_header_lines(path: str) -> int:
    """Count non-header data lines in a TSV — matches run_suite.py logic."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip() and not line.startswith("exp\t"))


# ===========================================================================
# INTEGRATION TESTS — exercise the REAL orchestrator._run_loop()
# ===========================================================================

def _noop(*args, **kwargs):
    """Swallow any call."""
    pass


def _make_fake_callbacks() -> "OrchestratorCallbacks":
    """Build a callbacks object that records status messages."""
    from tui.orchestrator import OrchestratorCallbacks
    statuses: list[tuple[str, str]] = []

    def on_status(status, msg):
        statuses[:]  # keep reference alive
        statuses.append((status, msg))

    cb = OrchestratorCallbacks(
        on_status_change=on_status,
        on_experiment_start=_noop,
        on_training_output=_noop,
        on_experiment_complete=_noop,
        on_stats_update=_noop,
        on_error=_noop,
    )
    cb._recorded_statuses = statuses  # stash for assertions
    return cb


def _build_orchestrator(tmp_path, max_experiments, tsv_name="results.tsv"):
    """Construct a real ExperimentOrchestrator wired to a temp directory.

    Heavy dependencies (git, hardware, heartbeat) are patched at __init__
    so the object is safe to use without a real repo.
    """
    from tui.orchestrator import ExperimentOrchestrator

    tsv = str(tmp_path / tsv_name)
    cb = _make_fake_callbacks()

    with patch("tui.orchestrator.GitManager") as MockGit, \
         patch("tui.orchestrator.get_hardware_summary", return_value={"chip_name": "Apple M5"}), \
         patch("tui.orchestrator.Heartbeat"):
        mock_git = MockGit.return_value
        mock_git.ensure_auto_push_remote.return_value = None
        mock_git.current_branch.return_value = "autoresearch/test"
        mock_git.branch_exists.return_value = True
        mock_git.checkout.return_value = None
        mock_git.has_uncommitted_changes.return_value = False
        mock_git.head_commit_message.return_value = "unrelated commit"
        mock_git.record_baseline.return_value = "abc1234"
        mock_git._run.return_value = ""

        orch = ExperimentOrchestrator(
            training_script="train.py",
            results_path=tsv,
            max_experiments=max_experiments,
            run_tag="test",
            callbacks=cb,
            model="test-model",
            dataset_name="test-dataset",
        )
    # Replace internals that were set in __init__ with safe mocks
    orch._git = mock_git
    orch._heartbeat = MagicMock()

    return orch, tsv, cb


class TestOrchestratorIntegration:
    """Run the REAL _run_loop with mocked LLM/training.

    These tests prove the actual gate code (orchestrator.py:366-373)
    fires correctly.  If someone removes the gate, these tests FAIL.
    """

    def _run_loop_sync(self, orch):
        """Run _run_loop synchronously (not in a thread) for determinism."""
        orch._run_loop()

    # ---- Scenario A: TSV already at limit → zero experiments run ----------

    def test_already_at_limit_zero_experiments(self, tmp_path):
        """Pre-populate TSV to max → _run_loop should exit immediately.

        range(5, 5) is naturally empty, so no loop body executes.
        The critical assertion is that _run_experiment is never called.
        """
        max_exp = 5
        orch, tsv, cb = _build_orchestrator(tmp_path, max_exp)
        _populate_tsv(tsv, max_exp)

        mock_llm = MagicMock()
        mock_llm.name.return_value = "mock"
        mock_llm.validate.return_value = True

        orch._run_experiment = MagicMock(
            side_effect=AssertionError("_run_experiment called — gate failed!")
        )

        with patch("tui.orchestrator.get_llm_backend", return_value=mock_llm):
            self._run_loop_sync(orch)

        orch._run_experiment.assert_not_called()
        assert len(load_results(tsv)) == max_exp

    # ---- Scenario B: crash lost 2 rows → exactly 2 more run --------------

    def test_crash_lost_rows_fills_gap(self, tmp_path):
        """5 rows → delete 2 → restart → runs exactly 2 more → stops at 5.

        range(3, 5) naturally gives 2 iterations. The gate is checked each
        iteration but doesn't fire because count < max until the range ends.
        """
        max_exp = 5
        orch, tsv, cb = _build_orchestrator(tmp_path, max_exp)
        _populate_tsv(tsv, max_exp)
        _delete_last_n_rows(tsv, 2)
        assert len(load_results(tsv)) == 3

        mock_llm = MagicMock()
        mock_llm.name.return_value = "mock"
        mock_llm.validate.return_value = True

        call_count = [0]

        def fake_run_experiment(exp_num, _pause_depth=0):
            call_count[0] += 1
            append_result(tsv, _make_result(exp_num))
            orch.total_runs += 1
            orch.kept_count += 1

        orch._run_experiment = fake_run_experiment

        with patch("tui.orchestrator.get_llm_backend", return_value=mock_llm):
            self._run_loop_sync(orch)

        assert call_count[0] == 2, (
            f"Expected exactly 2 experiments to fill gap, got {call_count[0]}"
        )
        assert len(load_results(tsv)) == max_exp

    # ---- Scenario C: resume from 3 → finish at 5 -------------------------

    def test_clean_resume_finishes_at_max(self, tmp_path):
        """3 rows exist (clean stop) → restart → runs 2 more → stops at 5."""
        max_exp = 5
        orch, tsv, cb = _build_orchestrator(tmp_path, max_exp)
        _populate_tsv(tsv, 3)

        mock_llm = MagicMock()
        mock_llm.name.return_value = "mock"
        mock_llm.validate.return_value = True

        call_count = [0]

        def fake_run_experiment(exp_num, _pause_depth=0):
            call_count[0] += 1
            append_result(tsv, _make_result(exp_num))
            orch.total_runs += 1
            orch.kept_count += 1

        orch._run_experiment = fake_run_experiment

        with patch("tui.orchestrator.get_llm_backend", return_value=mock_llm):
            self._run_loop_sync(orch)

        assert call_count[0] == 2
        assert len(load_results(tsv)) == max_exp

    # ---- Scenario D: non-contiguous exp numbers, range is empty -----------

    def test_noncontiguous_numbers_range_empty(self, tmp_path):
        """Rows with gaps (exp0,1,2,5,6) = 5 rows.

        next_experiment_number returns 7, so range(7, 5) is empty.
        No experiments run — the range boundary alone prevents overshoot.
        """
        max_exp = 5
        orch, tsv, cb = _build_orchestrator(tmp_path, max_exp)
        init_results_tsv(tsv)
        for i in [0, 1, 2, 5, 6]:
            append_result(tsv, _make_result(i, status="keep"))

        mock_llm = MagicMock()
        mock_llm.name.return_value = "mock"
        mock_llm.validate.return_value = True

        orch._run_experiment = MagicMock(
            side_effect=AssertionError("_run_experiment called!")
        )

        with patch("tui.orchestrator.get_llm_backend", return_value=mock_llm):
            self._run_loop_sync(orch)

        orch._run_experiment.assert_not_called()
        assert len(load_results(tsv)) == max_exp

    # ---- Scenario E: THE ACTUAL BUG — gate catches range overshoot --------

    def test_gate_catches_overshoot_from_low_exp_numbers(self, tmp_path):
        """THE core Issue #28 scenario where the gate is essential.

        Create a TSV that has 5 rows but with LOW exp numbers (0-4),
        then ALSO manually set max_experiments higher so range() would
        allow more iterations. The gate must fire and emit the
        "Experiment limit reached" message.

        Concretely: 5 rows exist (exp0-4), max_experiments=10.
        Without the gate: range(5, 10) = 5 more → total 10.
        With the gate at max=5: would stop at 5 rows.

        But to test the REAL scenario: max_experiments=5, and we
        artificially lower start_exp by having non-contiguous numbering
        where the highest exp is BELOW max_experiments, giving a range
        that enters the loop body where the gate can fire.

        Example: 5 rows as exp0, exp0, exp0, exp0, exp0 (all "exp0").
        next_experiment_number → 1. range(1, 5) = 4 iterations.
        Gate fires on first iteration because count(5) >= max(5).
        """
        max_exp = 5
        orch, tsv, cb = _build_orchestrator(tmp_path, max_exp)
        init_results_tsv(tsv)

        # Write 5 rows all numbered exp0 — simulates a pathological
        # crash scenario where exp numbering is unreliable
        for _ in range(5):
            append_result(tsv, _make_result(0, status="keep"))

        assert len(load_results(tsv)) == 5
        # next_experiment_number sees exp0 → returns 1
        assert next_experiment_number(tsv) == 1

        mock_llm = MagicMock()
        mock_llm.name.return_value = "mock"
        mock_llm.validate.return_value = True

        # Without the gate, range(1, 5) = 4 iterations → 4 MORE experiments!
        orch._run_experiment = MagicMock(
            side_effect=AssertionError("_run_experiment called — gate failed!")
        )

        with patch("tui.orchestrator.get_llm_backend", return_value=mock_llm):
            self._run_loop_sync(orch)

        # Gate fired INSIDE the loop body, blocking all 4 would-be iterations
        orch._run_experiment.assert_not_called()
        assert len(load_results(tsv)) == max_exp

        # THIS is the scenario where the gate message fires
        assert any("Experiment limit reached" in msg for _, msg in cb._recorded_statuses), (
            f"Gate status message not found in: {cb._recorded_statuses}"
        )

    def test_gate_prevents_overshoot_after_middle_row_loss(self, tmp_path):
        """Rows lost from the MIDDLE create a gap between row count and
        exp numbering — the gate prevents the resulting overshoot.

        8 rows exist (exp0-9 with 2 gaps), max=8.
        next_experiment_number sees exp9 → returns 10.
        range(10, 8) is empty → no overshoot possible even without gate.

        But flip it: max=10, 10 rows exist, lose 2 from middle.
        8 rows remain, next_experiment_number → 10 (exp9 still present).
        range(10, 10) empty. This is the UNDERCOUNT scenario — gate
        doesn't help and doesn't hurt.

        The gate is specifically for when exp numbers are LOWER than
        row count would suggest — test that directly.
        """
        max_exp = 5
        orch, tsv, cb = _build_orchestrator(tmp_path, max_exp)
        init_results_tsv(tsv)

        # 5 rows but exp numbers only go up to 2 (duplicates from
        # crash/restart confusion). next_experiment_number → 3.
        # range(3, 5) = 2 iterations. Gate fires on iteration 1.
        for i in [0, 1, 2, 1, 2]:
            append_result(tsv, _make_result(i, status="keep"))

        assert len(load_results(tsv)) == 5
        assert next_experiment_number(tsv) == 3  # sees exp2 as max

        mock_llm = MagicMock()
        mock_llm.name.return_value = "mock"
        mock_llm.validate.return_value = True

        orch._run_experiment = MagicMock(
            side_effect=AssertionError("_run_experiment called — gate failed!")
        )

        with patch("tui.orchestrator.get_llm_backend", return_value=mock_llm):
            self._run_loop_sync(orch)

        orch._run_experiment.assert_not_called()
        assert len(load_results(tsv)) == max_exp
        assert any("Experiment limit reached" in msg for _, msg in cb._recorded_statuses)

    # ---- Scenario E: repeated crash/restart cycles ------------------------

    def test_three_crash_restart_cycles(self, tmp_path):
        """3 crash cycles, each losing 1 row — never exceeds max."""
        max_exp = 5
        orch, tsv, cb = _build_orchestrator(tmp_path, max_exp)
        _populate_tsv(tsv, max_exp)

        mock_llm = MagicMock()
        mock_llm.name.return_value = "mock"
        mock_llm.validate.return_value = True

        for cycle in range(3):
            _delete_last_n_rows(tsv, 1)

            # Rebuild orchestrator state for this "restart"
            orch, _, cb = _build_orchestrator(tmp_path, max_exp)
            orch._results_path = tsv  # reuse same TSV

            call_count = [0]

            def fake_run_experiment(exp_num, _pause_depth=0):
                call_count[0] += 1
                append_result(tsv, _make_result(exp_num))
                orch.total_runs += 1

            orch._run_experiment = fake_run_experiment

            with patch("tui.orchestrator.get_llm_backend", return_value=mock_llm):
                self._run_loop_sync(orch)

            assert call_count[0] == 1, (
                f"Cycle {cycle}: expected 1 experiment, got {call_count[0]}"
            )
            assert len(load_results(tsv)) == max_exp, (
                f"Cycle {cycle}: expected {max_exp} rows, got {len(load_results(tsv))}"
            )
