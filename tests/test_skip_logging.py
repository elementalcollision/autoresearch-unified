"""Tests for Issue #43 — silent experiment skips now logged as status='skip' rows.

Verifies that pre-training failures (API unavailable, code apply error, syntax
error, git commit failure) produce a TSV row with status='skip' instead of
silently skipping and leaving a gap in experiment numbering.
"""

import csv
import os

from tui.results import ExperimentResult, append_result, load_results, init_results_tsv, HEADER


class TestSkipStatusInResults:
    """Test that 'skip' is a valid status and round-trips through TSV."""

    def test_skip_status_writes_to_tsv(self, tmp_path):
        """A skip result should be writable and readable from TSV."""
        path = str(tmp_path / "results.tsv")
        init_results_tsv(path)

        result = ExperimentResult(
            exp="exp5",
            description="API unavailable",
            val_bpb=0.0,
            peak_mem_gb=0.0,
            tok_sec=0,
            mfu=0.0,
            steps=0,
            status="skip",
            notes="API unavailable after 30 min of retries",
            gpu_name="Test GPU",
            baseline_sha="abc123",
        )
        append_result(path, result)

        loaded = load_results(path)
        assert len(loaded) == 1
        assert loaded[0].status == "skip"
        assert loaded[0].exp == "exp5"
        assert loaded[0].val_bpb == 0.0
        assert loaded[0].description == "API unavailable"

    def test_skip_does_not_break_load(self, tmp_path):
        """Mixed statuses including skip should all load correctly."""
        path = str(tmp_path / "results.tsv")
        init_results_tsv(path)

        statuses = ["baseline", "keep", "discard", "crash", "skip"]
        for i, status in enumerate(statuses):
            result = ExperimentResult(
                exp=f"exp{i}",
                description=f"test {status}",
                val_bpb=1.0 if status in ("baseline", "keep", "discard") else 0.0,
                peak_mem_gb=0.0,
                tok_sec=0,
                mfu=0.0,
                steps=0,
                status=status,
                notes="test",
                gpu_name="Test GPU",
                baseline_sha="abc123",
            )
            append_result(path, result)

        loaded = load_results(path)
        assert len(loaded) == 5
        assert [r.status for r in loaded] == statuses

    def test_skip_preserves_description(self, tmp_path):
        """Skip rows should preserve the proposal description for debugging."""
        path = str(tmp_path / "results.tsv")
        init_results_tsv(path)

        result = ExperimentResult(
            exp="exp10",
            description="Increase MATRIX_LR from 0.027 to 0.030",
            val_bpb=0.0,
            peak_mem_gb=0.0,
            tok_sec=0,
            mfu=0.0,
            steps=0,
            status="skip",
            notes="syntax error: invalid syntax at line 42",
            gpu_name="Test GPU",
            baseline_sha="def456",
        )
        append_result(path, result)

        loaded = load_results(path)
        assert loaded[0].description == "Increase MATRIX_LR from 0.027 to 0.030"
        assert "syntax error" in loaded[0].notes

    def test_skip_power_fields_default_zero(self, tmp_path):
        """Skip rows should have zero power fields since no training ran."""
        path = str(tmp_path / "results.tsv")
        init_results_tsv(path)

        result = ExperimentResult(
            exp="exp7",
            description="test skip",
            val_bpb=0.0,
            peak_mem_gb=0.0,
            tok_sec=0,
            mfu=0.0,
            steps=0,
            status="skip",
            notes="code apply failed",
            gpu_name="Test GPU",
            baseline_sha="abc123",
        )
        append_result(path, result)

        loaded = load_results(path)
        assert loaded[0].watts == 0.0
        assert loaded[0].joules_per_token == 0.0
        assert loaded[0].total_energy_joules == 0.0

    def test_row_count_equals_experiments(self, tmp_path):
        """Core invariant: every exp_num gets a row, even if skipped.

        This is the key property that Issue #43 fixes — before the fix,
        silent skips caused row_count < max_experiments.
        """
        path = str(tmp_path / "results.tsv")
        init_results_tsv(path)

        # Simulate 5 experiments: baseline + 3 normal + 1 skip + 1 normal
        scenarios = [
            ("exp0", "baseline", "Baseline run", 1.089),
            ("exp1", "keep", "Lower LR", 1.070),
            ("exp2", "skip", "API unavailable", 0.0),
            ("exp3", "crash", "Bad config", 0.0),
            ("exp4", "discard", "Worse LR", 1.080),
        ]

        for exp, status, desc, bpb in scenarios:
            result = ExperimentResult(
                exp=exp, description=desc, val_bpb=bpb,
                peak_mem_gb=0.0, tok_sec=0, mfu=0.0, steps=0,
                status=status, notes="test",
                gpu_name="Test GPU", baseline_sha="abc123",
            )
            append_result(path, result)

        loaded = load_results(path)
        assert len(loaded) == 5
        # No gaps in experiment numbering
        exp_nums = [r.exp for r in loaded]
        assert exp_nums == ["exp0", "exp1", "exp2", "exp3", "exp4"]


class TestBenchmarkSkipHandling:
    """Test that build_benchmark.py correctly handles skip rows."""

    def test_skip_counted_in_benchmark(self, tmp_path):
        """Skip rows should be counted by compute_run_stats."""
        # Import here to avoid import errors if build_benchmark has issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_benchmark",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "build_benchmark.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rows = [
            {"exp": "exp0", "val_bpb": "1.089", "status": "baseline", "description": "baseline"},
            {"exp": "exp1", "val_bpb": "1.070", "status": "keep", "description": "lower LR"},
            {"exp": "exp2", "val_bpb": "0.0", "status": "skip", "description": "API unavailable"},
            {"exp": "exp3", "val_bpb": "0.0", "status": "crash", "description": "bad config"},
            {"exp": "exp4", "val_bpb": "1.080", "status": "discard", "description": "worse LR"},
        ]

        stats = mod.compute_run_stats(rows)
        assert stats["total_exps"] == 4  # excludes baseline
        assert stats["keeps"] == 1
        assert stats["crashes"] == 1
        assert stats["skips"] == 1
        assert stats["discards"] == 1
        # keep + crash + skip + discard = total
        assert stats["keeps"] + stats["crashes"] + stats["skips"] + stats["discards"] == stats["total_exps"]
