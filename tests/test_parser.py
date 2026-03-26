"""Tests for tui.parser -- training output parsing."""
import pytest
from tui.parser import OutputParser, StepMetrics, FinalMetrics


class TestStepParsing:
    """Test parsing of training step output lines."""

    def test_normal_step_line(self):
        parser = OutputParser()
        line = "step 00192 (62.3%) | loss: 4.168331 | lrm: 0.66 | dt: 1001ms | tok/sec: 32,737 | mfu: 23.0% | epoch: 1 | remaining: 118s"
        results = parser.parse_line(line)
        assert len(results) == 1
        m = results[0]
        assert isinstance(m, StepMetrics)
        assert m.step == 192
        assert m.pct_done == 62.3
        assert m.loss == pytest.approx(4.168331)
        assert m.lrm == 0.66
        assert m.dt_ms == 1001
        assert m.tok_per_sec == 32737
        assert m.mfu == 23.0
        assert m.epoch == 1
        assert m.remaining == 118

    def test_step_zero(self):
        parser = OutputParser()
        line = "step 00000 (0.0%) | loss: 9.019033 | lrm: 1.00 | dt: 1713ms | tok/sec: 19,132 | mfu: 0.5% | epoch: 1 | remaining: 300s"
        results = parser.parse_line(line)
        assert len(results) == 1
        assert results[0].step == 0
        assert results[0].loss == pytest.approx(9.019033)

    def test_high_tok_sec_with_commas(self):
        parser = OutputParser()
        line = "step 00100 (50.0%) | loss: 3.5 | lrm: 0.80 | dt: 117ms | tok/sec: 279,159 | mfu: 6.7% | epoch: 1 | remaining: 150s"
        results = parser.parse_line(line)
        assert results[0].tok_per_sec == 279159

    def test_carriage_return_splits(self):
        """Training output uses \\r for progress -- parser should handle multiple steps in one line."""
        parser = OutputParser()
        line = "step 00001 (0.1%) | loss: 8.5 | lrm: 1.00 | dt: 100ms | tok/sec: 100,000 | mfu: 5.0% | epoch: 1 | remaining: 299s\rstep 00002 (0.2%) | loss: 8.4 | lrm: 1.00 | dt: 100ms | tok/sec: 100,000 | mfu: 5.0% | epoch: 1 | remaining: 298s"
        results = parser.parse_line(line)
        assert len(results) == 2
        assert results[0].step == 1
        assert results[1].step == 2

    def test_non_step_line_stored_as_startup(self):
        parser = OutputParser()
        line = "GPU: NVIDIA GeForce RTX 5070 Ti (16303MB VRAM)"
        results = parser.parse_line(line)
        assert len(results) == 1
        assert isinstance(results[0], str)
        assert "5070 Ti" in results[0]
        assert len(parser.startup_lines) == 1

    def test_empty_line(self):
        parser = OutputParser()
        results = parser.parse_line("")
        assert results == []

    def test_whitespace_only(self):
        parser = OutputParser()
        results = parser.parse_line("   \t  ")
        assert results == []


class TestFinalParsing:
    """Test parsing of the final results block (after ---)."""

    def test_full_final_block(self):
        parser = OutputParser()
        lines = [
            "---",
            "val_bpb:          0.916588",
            "training_seconds: 300.1",
            "total_seconds:    340.2",
            "peak_vram_mb:     5998.2",
            "mfu_percent:      6.12",
            "total_tokens_M:   76.3",
            "num_steps:        2329",
            "num_params_M:     50.3",
            "depth:            8",
            "backend:          cuda",
            "chip:             NVIDIA GeForce RTX 5070 Ti",
        ]
        for line in lines:
            parser.parse_line(line)

        assert parser.final is not None
        assert parser.final.val_bpb == pytest.approx(0.916588)
        assert parser.final.num_steps == 2329
        assert parser.final.depth == 8
        assert parser.final.backend == "cuda"
        assert parser.final.chip == "NVIDIA GeForce RTX 5070 Ti"

    def test_separator_triggers_final_mode(self):
        parser = OutputParser()
        results = parser.parse_line("---")
        assert parser.in_final_block is True
        assert any("complete" in str(r).lower() for r in results)

    def test_step_after_separator_not_parsed_as_step(self):
        """After ---, lines should be parsed as final metrics, not steps."""
        parser = OutputParser()
        parser.parse_line("---")
        results = parser.parse_line("val_bpb:          1.234")
        # Should NOT produce StepMetrics
        assert not any(isinstance(r, StepMetrics) for r in results)

    def test_final_not_triggered_without_separator(self):
        parser = OutputParser()
        parser.parse_line("val_bpb:          1.234")
        assert parser.final is None
        assert parser.in_final_block is False


class TestEnergyParsing:
    """Test parsing of energy/power metrics in the final block."""

    def test_full_final_block_with_energy(self):
        """Parser captures energy metrics when training script emits them."""
        parser = OutputParser()
        lines = [
            "---",
            "val_bpb:          0.916588",
            "training_seconds: 300.1",
            "total_seconds:    340.2",
            "peak_vram_mb:     5998.2",
            "mfu_percent:      6.12",
            "total_tokens_M:   76.3",
            "num_steps:        2329",
            "num_params_M:     50.3",
            "depth:            8",
            "backend:          cuda",
            "chip:             NVIDIA GeForce RTX 5070 Ti",
            "avg_watts:        350.5",
            "joules_per_token: 0.001378",
            "total_energy_j:   105210.5",
        ]
        for line in lines:
            parser.parse_line(line)
        assert parser.final is not None
        assert parser.final.avg_watts == pytest.approx(350.5)
        assert parser.final.joules_per_token == pytest.approx(0.001378)
        assert parser.final.total_energy_j == pytest.approx(105210.5)

    def test_backward_compat_no_energy_lines(self):
        """Parser finalizes correctly when training script has no energy lines."""
        parser = OutputParser()
        lines = [
            "---",
            "val_bpb:          1.234",
            "training_seconds: 200.0",
            "total_seconds:    220.0",
            "peak_vram_mb:     4096.0",
            "mfu_percent:      5.0",
            "total_tokens_M:   50.0",
            "num_steps:        1000",
            "num_params_M:     40.0",
            "depth:            6",
            "backend:          cuda",
            "chip:             NVIDIA RTX 3090",
        ]
        for line in lines:
            parser.parse_line(line)
        assert parser.final is not None
        assert parser.final.val_bpb == pytest.approx(1.234)
        assert parser.final.avg_watts == 0.0
        assert parser.final.joules_per_token == 0.0
        assert parser.final.total_energy_j == 0.0

    def test_format_final_emitted_only_once(self):
        """The formatted summary string should be emitted once, not per-field after chip."""
        parser = OutputParser()
        all_results = []
        lines = [
            "---",
            "val_bpb:          0.9",
            "training_seconds: 100.0",
            "total_seconds:    110.0",
            "peak_vram_mb:     2048.0",
            "mfu_percent:      4.0",
            "total_tokens_M:   30.0",
            "num_steps:        800",
            "num_params_M:     35.0",
            "depth:            5",
            "backend:          rocm",
            "chip:             AMD Instinct MI300X",
            "avg_watts:        700.0",
            "joules_per_token: 0.002",
            "total_energy_j:   70000.0",
        ]
        for line in lines:
            results = parser.parse_line(line)
            all_results.extend(results)

        # Count formatted summary strings (contain "Evaluation complete")
        summary_count = sum(1 for r in all_results if isinstance(r, str) and "Evaluation complete" in r)
        assert summary_count == 1
