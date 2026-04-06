"""Tests for autoresearch/results.py — TSV management and experiment classification."""

import os
import pytest

from autoresearch.results import (
    ExperimentResult, HEADER, init_results_tsv, append_result,
    load_results, get_best_result, next_experiment_number,
    classify_experiment, categorize_experiments, format_history_for_prompt,
)


@pytest.fixture
def results_path(tmp_path):
    return str(tmp_path / "results.tsv")


def _make_result(exp="exp0", desc="baseline", val_bpb=1.5, status="baseline", **kwargs):
    defaults = dict(
        peak_mem_gb=8.0, tok_sec=10000, mfu=20.0, steps=100,
        notes="", gpu_name="H100", baseline_sha="abc",
        watts=300.0, joules_per_token=0.001, total_energy_joules=500.0,
    )
    defaults.update(kwargs)
    return ExperimentResult(exp=exp, description=desc, val_bpb=val_bpb, status=status, **defaults)


class TestResultsTsv:
    def test_init_creates_header(self, results_path):
        init_results_tsv(results_path)
        content = open(results_path).read()
        assert content == HEADER

    def test_append_and_load(self, results_path):
        init_results_tsv(results_path)
        result = _make_result()
        append_result(results_path, result)
        loaded = load_results(results_path)
        assert len(loaded) == 1
        assert loaded[0].exp == "exp0"
        assert loaded[0].val_bpb == 1.5

    def test_get_best_result(self, results_path):
        init_results_tsv(results_path)
        append_result(results_path, _make_result("exp0", "baseline", 1.5, "baseline"))
        append_result(results_path, _make_result("exp1", "lr change", 1.4, "keep"))
        append_result(results_path, _make_result("exp2", "batch change", 1.6, "discard"))

        best_bpb, best_exp = get_best_result(results_path)
        assert best_bpb == pytest.approx(1.4)
        assert "exp1" in best_exp

    def test_next_experiment_number(self, results_path):
        init_results_tsv(results_path)
        assert next_experiment_number(results_path) == 0

        append_result(results_path, _make_result("exp0", "baseline", 1.5, "baseline"))
        assert next_experiment_number(results_path) == 1

        append_result(results_path, _make_result("exp1", "test", 1.4, "keep"))
        assert next_experiment_number(results_path) == 2


class TestClassifyExperiment:
    @pytest.mark.parametrize("desc,expected", [
        ("Increase batch_size from 32 to 64", "batch_size"),
        ("Change DEPTH from 12 to 16", "architecture"),
        ("Increase warmup ratio", "schedule"),
        ("Reduce weight_decay", "regularization"),
        ("Increase MATRIX_LR to 0.06", "learning_rate"),
        ("Enable activation_checkpointing", "infrastructure"),
        ("Random experiment", "other"),
    ])
    def test_classification(self, desc, expected):
        assert classify_experiment(desc) == expected

    def test_categorize_experiments(self):
        results = [
            _make_result("exp0", "baseline", 1.5, "baseline"),
            _make_result("exp1", "Increase MATRIX_LR", 1.4, "keep"),
            _make_result("exp2", "Reduce batch_size", 1.45, "discard"),
            _make_result("exp3", "Change DEPTH", 1.38, "keep"),
        ]
        counts = categorize_experiments(results)
        assert counts.get("learning_rate") == 1
        assert counts.get("batch_size") == 1
        assert counts.get("architecture") == 1


class TestFormatHistory:
    def test_empty_results(self, results_path):
        text = format_history_for_prompt(results_path)
        assert "No experiments" in text

    def test_formatted_output(self, results_path):
        init_results_tsv(results_path)
        append_result(results_path, _make_result("exp0", "baseline", 1.5, "baseline"))
        text = format_history_for_prompt(results_path)
        assert "exp0" in text
        assert "1.5000" in text
