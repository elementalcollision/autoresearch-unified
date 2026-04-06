"""Tests for drift/strategy.py — repetition, diversity, similarity detection."""

import pytest

from autoresearch.results import ExperimentResult
from drift.strategy import StrategyDrift, StrategyConfig


def _make_result(exp_num, desc, status="discard", val_bpb=1.4):
    return ExperimentResult(
        exp=f"exp{exp_num}", description=desc,
        val_bpb=val_bpb, peak_mem_gb=8.0, tok_sec=10000,
        mfu=20.0, steps=100, status=status, notes="",
    )


class TestRepetition:
    def test_repetition_detected(self):
        drift = StrategyDrift(StrategyConfig(repetition_threshold=4))
        results = [
            _make_result(0, "baseline", "baseline"),
            _make_result(1, "Increase MATRIX_LR to 0.05"),
            _make_result(2, "Increase MATRIX_LR to 0.06"),
            _make_result(3, "Decrease MATRIX_LR to 0.03"),
            _make_result(4, "Increase SCALAR_LR to 0.5"),
        ]
        alerts = drift.check(results)
        repetition = [a for a in alerts if a.category == "repetition"]
        assert len(repetition) > 0

    def test_no_repetition_with_mixed_categories(self):
        drift = StrategyDrift(StrategyConfig(repetition_threshold=4))
        results = [
            _make_result(0, "baseline", "baseline"),
            _make_result(1, "Increase MATRIX_LR"),
            _make_result(2, "Reduce batch_size"),
            _make_result(3, "Change DEPTH"),
            _make_result(4, "Increase warmup ratio"),
        ]
        alerts = drift.check(results)
        repetition = [a for a in alerts if a.category == "repetition"]
        assert len(repetition) == 0


class TestDiversity:
    def test_low_diversity_detected(self):
        drift = StrategyDrift(StrategyConfig(diversity_window=10, diversity_min_entropy=1.0))
        # All learning rate experiments
        results = [_make_result(0, "baseline", "baseline")]
        results += [_make_result(i, f"Change MATRIX_LR to {0.01 * i}") for i in range(1, 11)]
        alerts = drift.check(results)
        diversity = [a for a in alerts if a.category == "diversity"]
        assert len(diversity) > 0

    def test_good_diversity_no_alert(self):
        drift = StrategyDrift(StrategyConfig(diversity_window=10, diversity_min_entropy=0.5))
        results = [_make_result(0, "baseline", "baseline")]
        descs = [
            "Change MATRIX_LR", "Reduce batch_size", "Change DEPTH",
            "Increase warmup ratio", "Reduce weight_decay",
            "Change SCALAR_LR", "Halve batch_size", "Change DEPTH again",
            "Increase warmdown", "Reduce weight_decay more",
        ]
        results += [_make_result(i + 1, d) for i, d in enumerate(descs)]
        alerts = drift.check(results)
        diversity = [a for a in alerts if a.category == "diversity"]
        assert len(diversity) == 0


class TestSimilarity:
    def test_near_duplicate_detected(self):
        drift = StrategyDrift(StrategyConfig(similarity_overlap_threshold=0.5))
        results = [
            _make_result(0, "baseline", "baseline"),
            _make_result(1, "Reduce batch_size from 64 to 32"),
            _make_result(2, "Increase MATRIX_LR from 0.04 to 0.06 for better convergence"),
            _make_result(3, "Increase MATRIX_LR from 0.04 to 0.06 for faster convergence"),  # near-duplicate of exp2
        ]
        alerts = drift.check(results)
        similarity = [a for a in alerts if a.category == "similarity"]
        assert len(similarity) > 0
