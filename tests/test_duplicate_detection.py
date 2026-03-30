"""Tests for duplicate proposal detection (PR #25 Enhancement 4).

Covers:
- Exact duplicate detection
- Same-param-same-value detection
- Non-duplicate cases (different params, different values)
- Integration with orchestrator
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.results import (
    ExperimentResult,
    append_result,
    init_results_tsv,
)


# ---------------------------------------------------------------------------
# Helper to create orchestrator with test results
# ---------------------------------------------------------------------------

def _make_orchestrator(results_path):
    """Create a minimal orchestrator pointing at a test results file."""
    with patch("tui.orchestrator.get_hardware_summary", return_value={}), \
         patch("tui.orchestrator.GitManager"):
        from tui.orchestrator import ExperimentOrchestrator
        orch = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
        orch._results_path = results_path
        return orch


def _populate(path, descriptions, statuses=None):
    """Write a results.tsv with the given descriptions."""
    init_results_tsv(path)
    if statuses is None:
        statuses = ["discard"] * len(descriptions)
    for i, (desc, status) in enumerate(zip(descriptions, statuses)):
        append_result(path, ExperimentResult(
            exp=f"exp{i}", description=desc,
            val_bpb=1.1 if status != "crash" else 0.0,
            peak_mem_gb=8.0, tok_sec=30000, mfu=20.0,
            steps=500 if status != "crash" else 0,
            status=status, notes="",
        ))


# ---------------------------------------------------------------------------
# _same_param_same_value (static method)
# ---------------------------------------------------------------------------

class TestSameParamSameValue:
    def test_identical_descriptions(self):
        from tui.orchestrator import ExperimentOrchestrator
        assert ExperimentOrchestrator._same_param_same_value(
            "decrease matrix_lr from 0.04 to 0.03",
            "decrease matrix_lr from 0.04 to 0.03",
        )

    def test_different_wording_same_change(self):
        from tui.orchestrator import ExperimentOrchestrator
        assert ExperimentOrchestrator._same_param_same_value(
            "decrease matrix_lr from 0.04 to 0.03",
            "lower matrix_lr to 0.03",
        )

    def test_different_target_values(self):
        from tui.orchestrator import ExperimentOrchestrator
        assert not ExperimentOrchestrator._same_param_same_value(
            "decrease matrix_lr from 0.04 to 0.03",
            "decrease matrix_lr from 0.04 to 0.02",
        )

    def test_different_params(self):
        from tui.orchestrator import ExperimentOrchestrator
        assert not ExperimentOrchestrator._same_param_same_value(
            "decrease matrix_lr from 0.04 to 0.03",
            "decrease scalar_lr from 0.5 to 0.03",
        )

    def test_no_to_pattern(self):
        from tui.orchestrator import ExperimentOrchestrator
        assert not ExperimentOrchestrator._same_param_same_value(
            "enable activation_checkpointing",
            "enable activation_checkpointing",
        )

    def test_case_insensitive(self):
        from tui.orchestrator import ExperimentOrchestrator
        assert ExperimentOrchestrator._same_param_same_value(
            "Increase MATRIX_LR from 0.04 to 0.06",
            "increase matrix_lr from 0.04 to 0.06",
        )


# ---------------------------------------------------------------------------
# _is_near_duplicate
# ---------------------------------------------------------------------------

class TestIsNearDuplicate:
    def test_exact_duplicate(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        _populate(path, [
            "baseline (no modifications)",
            "Increase MATRIX_LR from 0.04 to 0.06",
        ], ["baseline", "discard"])

        orch = _make_orchestrator(path)
        assert orch._is_near_duplicate("Increase MATRIX_LR from 0.04 to 0.06")

    def test_exact_duplicate_case_insensitive(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        _populate(path, [
            "Increase MATRIX_LR from 0.04 to 0.06",
        ])

        orch = _make_orchestrator(path)
        assert orch._is_near_duplicate("increase matrix_lr from 0.04 to 0.06")

    def test_same_param_same_value_duplicate(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        _populate(path, [
            "Decrease MATRIX_LR from 0.04 to 0.03",
        ])

        orch = _make_orchestrator(path)
        assert orch._is_near_duplicate("Lower MATRIX_LR to 0.03")

    def test_not_duplicate_different_value(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        _populate(path, [
            "Decrease MATRIX_LR from 0.04 to 0.03",
        ])

        orch = _make_orchestrator(path)
        assert not orch._is_near_duplicate("Decrease MATRIX_LR from 0.04 to 0.025")

    def test_not_duplicate_different_param(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        _populate(path, [
            "Decrease MATRIX_LR from 0.04 to 0.03",
        ])

        orch = _make_orchestrator(path)
        assert not orch._is_near_duplicate("Decrease SCALAR_LR from 0.5 to 0.3")

    def test_not_duplicate_empty_history(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        init_results_tsv(path)

        orch = _make_orchestrator(path)
        assert not orch._is_near_duplicate("Increase MATRIX_LR from 0.04 to 0.06")

    def test_not_duplicate_no_results_file(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        # Don't create the file

        orch = _make_orchestrator(path)
        assert not orch._is_near_duplicate("Increase MATRIX_LR from 0.04 to 0.06")

    def test_real_world_duplicate_from_control_run(self, tmp_path):
        """Based on actual duplicates observed in the control A/B run."""
        path = str(tmp_path / "results.tsv")
        _populate(path, [
            "baseline (no modifications)",
            "Increase EMBEDDING_LR from 0.6 to 0.8",
            "Decrease MATRIX_LR from 0.04 to 0.035",
        ], ["baseline", "discard", "discard"])

        orch = _make_orchestrator(path)
        # exp21 and exp28 were exact duplicates in the control run
        assert orch._is_near_duplicate("Increase EMBEDDING_LR from 0.6 to 0.8")
        # Different value should not match
        assert not orch._is_near_duplicate("Increase EMBEDDING_LR from 0.6 to 0.9")

    def test_batch_size_crash_duplicate(self, tmp_path):
        """DEVICE_BATCH_SIZE crashes were repeated 4 times in the control run."""
        path = str(tmp_path / "results.tsv")
        _populate(path, [
            "Decrease DEVICE_BATCH_SIZE to increase gradient steps within time budget",
        ], ["crash"])

        orch = _make_orchestrator(path)
        # Exact same description
        assert orch._is_near_duplicate(
            "Decrease DEVICE_BATCH_SIZE to increase gradient steps within time budget"
        )
        # Similar but reworded (no "to X" pattern — only caught by exact match)
        assert not orch._is_near_duplicate(
            "Decrease DEVICE_BATCH_SIZE by 1 step to increase the number of gradient steps"
        )
