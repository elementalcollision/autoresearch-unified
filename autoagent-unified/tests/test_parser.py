"""Tests for autoresearch/parser.py — training output parsing."""

import pytest

from autoresearch.parser import OutputParser, StepMetrics, FinalMetrics


class TestOutputParser:
    def test_parse_step_line(self):
        parser = OutputParser()
        line = "step 00192 (62.3%) | loss: 4.168331 | lrm: 0.66 | dt: 1001ms | tok/sec: 32,737 | mfu: 23.0% | epoch: 1 | remaining: 118s"
        results = parser.parse_line(line)

        assert len(results) == 1
        m = results[0]
        assert isinstance(m, StepMetrics)
        assert m.step == 192
        assert m.pct_done == pytest.approx(62.3)
        assert m.loss == pytest.approx(4.168331)
        assert m.tok_per_sec == 32737
        assert m.mfu == pytest.approx(23.0)

    def test_parse_final_block(self):
        parser = OutputParser()
        parser.parse_line("---")
        parser.parse_line("val_bpb:          1.329263")
        parser.parse_line("training_seconds: 300.5")
        parser.parse_line("peak_vram_mb:     8192.0")
        parser.parse_line("mfu_percent:      23.5")
        parser.parse_line("total_tokens_M:   100.0")
        parser.parse_line("num_steps:        300")
        parser.parse_line("num_params_M:     124.0")
        parser.parse_line("depth:            12")
        parser.parse_line("backend:          cuda")
        parser.parse_line("chip:             H100")

        assert parser.final is not None
        assert parser.final.val_bpb == pytest.approx(1.329263)
        assert parser.final.num_steps == 300
        assert parser.final.chip == "H100"

    def test_parse_carriage_return_segments(self):
        parser = OutputParser()
        # Multiple step updates on one line separated by \r
        line = "step 00001 (1.0%) | loss: 5.0 | lrm: 0.01 | dt: 500ms | tok/sec: 10,000 | mfu: 10.0% | epoch: 1 | remaining: 295s\rstep 00002 (2.0%) | loss: 4.9 | lrm: 0.02 | dt: 501ms | tok/sec: 10,100 | mfu: 10.1% | epoch: 1 | remaining: 290s"
        results = parser.parse_line(line)

        steps = [r for r in results if isinstance(r, StepMetrics)]
        assert len(steps) == 2
        assert steps[0].step == 1
        assert steps[1].step == 2

    def test_startup_lines_captured(self):
        parser = OutputParser()
        parser.parse_line("Loading dataset...")
        parser.parse_line("Compiling model...")
        assert len(parser.startup_lines) == 2
        assert "Loading dataset..." in parser.startup_lines
