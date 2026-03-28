"""Tests for strategy diversity enhancements (PR #25 implementation).

Covers:
- Experiment classification into strategy categories
- Category summary in formatted history
- Stagnation detection logic
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
    categorize_experiments,
    classify_experiment,
    format_history_for_prompt,
    init_results_tsv,
    load_results,
)


# ---------------------------------------------------------------------------
# classify_experiment
# ---------------------------------------------------------------------------

class TestClassifyExperiment:
    def test_lr_changes(self):
        assert classify_experiment("Increase MATRIX_LR from 0.04 to 0.06") == "learning_rate"
        assert classify_experiment("Lower SCALAR_LR to 0.5") == "learning_rate"
        assert classify_experiment("Reduce EMBEDDING_LR from 0.6 to 0.3") == "learning_rate"
        assert classify_experiment("Bump learning rate by 10%") == "learning_rate"

    def test_batch_size(self):
        assert classify_experiment("Halve TOTAL_BATCH_SIZE from 2^19 to 2^18") == "batch_size"
        assert classify_experiment("Reduce batch size to 32") == "batch_size"
        assert classify_experiment("Double TOTAL_BATCH_SIZE") == "batch_size"

    def test_architecture(self):
        assert classify_experiment("Increase DEPTH from 8 to 12") == "architecture"
        assert classify_experiment("Change HEAD_DIM from 64 to 128") == "architecture"
        assert classify_experiment("Set WINDOW_PATTERN to SSLL") == "architecture"
        assert classify_experiment("Adjust MLP_RATIO to 3.0") == "architecture"
        assert classify_experiment("Modify ASPECT_RATIO") == "architecture"

    def test_schedule(self):
        assert classify_experiment("Set WARMUP_RATIO to 0.05") == "schedule"
        assert classify_experiment("Reduce WARMDOWN_RATIO from 0.5 to 0.3") == "schedule"
        assert classify_experiment("Set FINAL_LR_FRAC to 0.1") == "schedule"

    def test_regularization(self):
        assert classify_experiment("Reduce WEIGHT_DECAY from 0.2 to 0.1") == "regularization"
        assert classify_experiment("Tune ADAM_BETAS to (0.9, 0.95)") == "regularization"

    def test_infrastructure(self):
        assert classify_experiment("Enable ACTIVATION_CHECKPOINTING") == "infrastructure"
        assert classify_experiment("Change COMPILE_MODE to reduce-overhead") == "infrastructure"

    def test_other(self):
        assert classify_experiment("baseline (no modifications)") == "other"
        assert classify_experiment("Something completely different") == "other"

    def test_batch_beats_lr_when_both_present(self):
        # "batch_size" is checked before "learning_rate" in the function
        assert classify_experiment("Change batch_size and _lr together") == "batch_size"


# ---------------------------------------------------------------------------
# categorize_experiments
# ---------------------------------------------------------------------------

class TestCategorizeExperiments:
    def test_empty(self):
        assert categorize_experiments([]) == {}

    def test_skips_baseline(self):
        results = [
            ExperimentResult(
                exp="exp0", description="baseline (no modifications)",
                val_bpb=1.1, peak_mem_gb=8.0, tok_sec=30000, mfu=20.0,
                steps=500, status="baseline", notes="",
            ),
        ]
        assert categorize_experiments(results) == {}

    def test_mixed_categories(self):
        results = [
            ExperimentResult(
                exp="exp0", description="baseline", val_bpb=1.1,
                peak_mem_gb=8.0, tok_sec=30000, mfu=20.0, steps=500,
                status="baseline", notes="",
            ),
            ExperimentResult(
                exp="exp1", description="Increase MATRIX_LR to 0.06",
                val_bpb=1.08, peak_mem_gb=8.0, tok_sec=30000, mfu=20.0,
                steps=500, status="keep", notes="",
            ),
            ExperimentResult(
                exp="exp2", description="Halve TOTAL_BATCH_SIZE",
                val_bpb=1.05, peak_mem_gb=4.0, tok_sec=30000, mfu=20.0,
                steps=1000, status="keep", notes="",
            ),
            ExperimentResult(
                exp="exp3", description="Increase MATRIX_LR to 0.08",
                val_bpb=1.12, peak_mem_gb=8.0, tok_sec=30000, mfu=20.0,
                steps=500, status="discard", notes="",
            ),
        ]
        cats = categorize_experiments(results)
        assert cats["learning_rate"] == 2
        assert cats["batch_size"] == 1


# ---------------------------------------------------------------------------
# format_history_for_prompt — strategy summary footer
# ---------------------------------------------------------------------------

class TestFormatHistoryStrategyFooter:
    def test_no_results(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        assert format_history_for_prompt(path) == "No experiments yet."

    def test_baseline_only_no_summary(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        init_results_tsv(path)
        append_result(path, ExperimentResult(
            exp="exp0", description="baseline (no modifications)",
            val_bpb=1.1, peak_mem_gb=8.0, tok_sec=30000, mfu=20.0,
            steps=500, status="baseline", notes="",
        ))
        text = format_history_for_prompt(path)
        # baseline-only runs should show the table but "other" is skipped
        # because classify returns "other" for baseline, and baseline status
        # is filtered out of categorize_experiments
        assert "Strategy summary" not in text

    def test_shows_category_counts(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        init_results_tsv(path)
        append_result(path, ExperimentResult(
            exp="exp0", description="baseline", val_bpb=1.1,
            peak_mem_gb=8.0, tok_sec=30000, mfu=20.0, steps=500,
            status="baseline", notes="",
        ))
        append_result(path, ExperimentResult(
            exp="exp1", description="Increase MATRIX_LR to 0.06",
            val_bpb=1.08, peak_mem_gb=8.0, tok_sec=30000, mfu=20.0,
            steps=500, status="keep", notes="",
        ))
        append_result(path, ExperimentResult(
            exp="exp2", description="Halve TOTAL_BATCH_SIZE",
            val_bpb=0.0, peak_mem_gb=0.0, tok_sec=0, mfu=0.0,
            steps=0, status="crash", notes="OOM",
        ))

        text = format_history_for_prompt(path)
        assert "Strategy summary" in text
        assert "learning_rate: 1 tried, 1 kept" in text
        assert "batch_size: 1 tried, 0 kept" in text


# ---------------------------------------------------------------------------
# Stagnation detection (orchestrator)
# ---------------------------------------------------------------------------

class TestStagnationDetection:
    """Test _detect_stagnation via the orchestrator.

    We construct a results.tsv file and call the method directly,
    avoiding the need to mock the LLM or training subprocess.
    """

    def _make_orchestrator(self, results_path):
        """Create a minimal orchestrator pointing at a test results file."""
        # Import here to avoid module-level hardware detection
        with patch("tui.orchestrator.get_hardware_summary", return_value={}), \
             patch("tui.orchestrator.GitManager"):
            from tui.orchestrator import ExperimentOrchestrator
            orch = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            orch._results_path = results_path
            return orch

    def _populate_results(self, path, descriptions, statuses):
        """Write a results.tsv with the given descriptions and statuses."""
        init_results_tsv(path)
        for i, (desc, status) in enumerate(zip(descriptions, statuses)):
            append_result(path, ExperimentResult(
                exp=f"exp{i}", description=desc,
                val_bpb=1.1 if status != "crash" else 0.0,
                peak_mem_gb=8.0, tok_sec=30000, mfu=20.0,
                steps=500 if status != "crash" else 0,
                status=status, notes="",
            ))

    def test_no_stagnation_under_15(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        descs = [f"Increase MATRIX_LR to 0.0{i}" for i in range(10)]
        statuses = ["discard"] * 10
        self._populate_results(path, descs, statuses)

        orch = self._make_orchestrator(path)
        assert orch._detect_stagnation() is None

    def test_no_stagnation_with_keeps(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        descs = [f"Increase MATRIX_LR to 0.0{i}" for i in range(15)]
        statuses = ["keep", "keep", "keep"] + ["discard"] * 12
        self._populate_results(path, descs, statuses)

        orch = self._make_orchestrator(path)
        assert orch._detect_stagnation() is None

    def test_stagnation_detected_lr_heavy(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        # 15 experiments, only 1 keep, 10 are LR changes
        descs = (
            ["Increase MATRIX_LR to 0.06"] * 10
            + ["Increase DEPTH to 12"] * 5
        )
        statuses = ["keep"] + ["discard"] * 14
        self._populate_results(path, descs, statuses)

        orch = self._make_orchestrator(path)
        nudge = orch._detect_stagnation()
        assert nudge is not None
        assert "exhausted" in nudge
        assert "batch size" in nudge.lower()

    def test_no_stagnation_diverse_strategies(self, tmp_path):
        path = str(tmp_path / "results.tsv")
        # 15 experiments, only 1 keep, but diverse strategies (only 3 LR)
        descs = (
            ["Increase MATRIX_LR to 0.06"] * 3
            + ["Halve TOTAL_BATCH_SIZE"] * 3
            + ["Increase DEPTH to 12"] * 3
            + ["Set WARMUP_RATIO to 0.05"] * 3
            + ["Reduce WEIGHT_DECAY to 0.1"] * 3
        )
        statuses = ["keep"] + ["discard"] * 14
        self._populate_results(path, descs, statuses)

        orch = self._make_orchestrator(path)
        assert orch._detect_stagnation() is None


# ---------------------------------------------------------------------------
# System prompt contains strategy hints
# ---------------------------------------------------------------------------

class TestSystemPromptStrategyHints:
    def test_strategy_guidance_present(self):
        _FAKE_HW = {
            "chip_name": "Test GPU",
            "memory_gb": 16,
            "gpu_cores": 64,
            "peak_tflops": 10.0,
            "chip_tier": "test",
        }
        with patch("backends.get_hardware_info", return_value=_FAKE_HW):
            from tui.llm_backend import get_system_prompt
            prompt = get_system_prompt(_FAKE_HW)

        assert "Strategy guidance" in prompt
        assert "ADAM_BETAS" in prompt
        assert "WARMUP_RATIO" in prompt
        assert "TOTAL_BATCH_SIZE" in prompt
        assert "halving batch = 2x steps" in prompt
        assert "ACTIVATION_CHECKPOINTING" in prompt
