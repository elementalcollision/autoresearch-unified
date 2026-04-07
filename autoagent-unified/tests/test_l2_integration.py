"""L2 Component Integration Tests.

Tests cross-module interactions:
- L2.1: Results round-trip (init -> append -> load -> format -> classify -> best -> crash recovery)
- L2.2: Drift monitor end-to-end (synthetic data -> check -> inject -> format)
- L2.3: Tool wrappers (analyze, drift, hardware)
- L2.4: LLM backend factory (mocked env vars)
- L2.5: LLM response parsing
"""

import json
import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from autoresearch.results import (
    ExperimentResult,
    append_result,
    categorize_experiments,
    classify_experiment,
    format_history_for_prompt,
    get_best_result,
    init_results_tsv,
    load_results,
    next_experiment_number,
    HEADER,
)
from autoresearch.resilience import validate_results_tsv


def _has_openai() -> bool:
    try:
        import openai
        return True
    except ImportError:
        return False
from drift.monitor import DriftConfig, DriftMonitor, DriftReport
from drift.performance import PerformanceConfig
from drift.strategy import StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(n: int, desc: str, val_bpb: float = 1.3, status: str = "keep",
                 tok_sec: int = 10000, watts: float = 300.0, **kw) -> ExperimentResult:
    return ExperimentResult(
        exp=f"exp{n}", description=desc, val_bpb=val_bpb,
        peak_mem_gb=kw.get("peak_mem_gb", 8.0), tok_sec=tok_sec,
        mfu=kw.get("mfu", 35.0), steps=kw.get("steps", 500),
        status=status, notes="", gpu_name="RTX 4090",
        baseline_sha="abc123", watts=watts,
        joules_per_token=kw.get("joules_per_token", 0.03),
        total_energy_joules=kw.get("total_energy_joules", 9000.0),
    )


# ===========================================================================
# L2.1 — Results round-trip
# ===========================================================================

class TestResultsRoundTrip:
    """Full lifecycle: init -> append -> load -> format -> classify -> best -> crash recovery."""

    def test_full_lifecycle(self, tmp_path):
        tsv = str(tmp_path / "results.tsv")

        # 1. init creates 14-column header
        init_results_tsv(tsv)
        header = Path(tsv).read_text()
        assert header == HEADER
        assert header.count("\t") == 13  # 14 columns = 13 tabs

        # 2. append 10 results
        descriptions = [
            "baseline defaults",
            "Increase MATRIX_LR from 0.04 to 0.06",
            "Reduce batch_size from 64 to 32",
            "Increase DEPTH from 12 to 16",
            "Reduce weight_decay from 0.1 to 0.05",
            "Increase warmup ratio from 0 to 0.1",
            "Switch compile_mode to reduce-overhead",
            "Decrease learning_rate to 0.02",
            "Increase ADAM_BETA2 from 0.95 to 0.98",
            "Try WINDOW_PATTERN = [2,6]",
        ]
        statuses = ["baseline", "keep", "discard", "keep", "discard",
                     "keep", "discard", "keep", "discard", "keep"]
        val_bpbs = [1.4, 1.35, 1.42, 1.32, 1.45,
                     1.31, 1.38, 1.30, 1.43, 1.29]

        for i, (desc, status, bpb) in enumerate(zip(descriptions, statuses, val_bpbs)):
            append_result(tsv, _make_result(i, desc, val_bpb=bpb, status=status))

        # 3. load returns 10 ExperimentResult objects
        results = load_results(tsv)
        assert len(results) == 10
        assert all(isinstance(r, ExperimentResult) for r in results)
        assert results[0].exp == "exp0"
        assert results[9].description == "Try WINDOW_PATTERN = [2,6]"

        # 4. format_history_for_prompt produces formatted table
        formatted = format_history_for_prompt(tsv)
        assert "Exp" in formatted
        assert "Status" in formatted
        assert "val_bpb" in formatted
        assert "baseline defaults" in formatted
        assert "Strategy summary" in formatted

        # 5. classify_experiment on each -> correct categories
        expected_cats = [
            "other",           # baseline defaults
            "learning_rate",   # MATRIX_LR
            "batch_size",      # batch_size
            "architecture",    # DEPTH
            "regularization",  # weight_decay
            "schedule",        # warmup
            "infrastructure",  # compile_mode
            "learning_rate",   # learning_rate
            "regularization",  # ADAM_BETA
            "architecture",    # WINDOW_PATTERN
        ]
        for desc, expected in zip(descriptions, expected_cats):
            assert classify_experiment(desc) == expected, f"Failed for: {desc}"

        # 6. get_best_result -> correct minimum val_bpb
        best_bpb, best_exp = get_best_result(tsv)
        assert abs(best_bpb - 1.29) < 0.001
        assert "exp9" in best_exp

        # 7. next_experiment_number
        assert next_experiment_number(tsv) == 10

    def test_crash_recovery_truncated_line(self, tmp_path):
        """Simulate crash mid-append: truncated trailing line is detected."""
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)

        # Write 2 complete results
        append_result(tsv, _make_result(0, "baseline", val_bpb=1.4, status="baseline"))
        append_result(tsv, _make_result(1, "good exp", val_bpb=1.35, status="keep"))

        # Simulate crash: append a truncated line (missing columns, no trailing newline)
        with open(tsv, "a") as f:
            f.write("exp2\ttruncated line\t1.30")  # no \n = mid-write crash

        # validate_results_tsv should detect the truncated trailing line
        is_valid, warnings = validate_results_tsv(tsv)
        assert not is_valid
        assert any("truncated" in w.lower() or "fields" in w.lower() for w in warnings)

        # load_results should still load the 2 valid rows (skip truncated)
        results = load_results(tsv)
        assert len(results) >= 2

    def test_categorize_experiments(self, tmp_path):
        """categorize_experiments counts by strategy category."""
        results = [
            _make_result(0, "baseline", status="baseline"),
            _make_result(1, "Increase MATRIX_LR", status="keep"),
            _make_result(2, "Decrease learning_rate", status="discard"),
            _make_result(3, "Reduce batch_size", status="keep"),
            _make_result(4, "Increase DEPTH", status="keep"),
        ]
        cats = categorize_experiments(results)
        assert cats["learning_rate"] == 2
        assert cats["batch_size"] == 1
        assert cats["architecture"] == 1
        assert "baseline" not in str(cats)  # baselines are excluded


# ===========================================================================
# L2.2 — Drift monitor end-to-end
# ===========================================================================

class TestDriftMonitorEndToEnd:
    """DriftMonitor aggregates alerts from all three detectors."""

    def _write_plateau_tsv(self, tmp_path) -> str:
        """Write synthetic results.tsv with a performance plateau."""
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)
        append_result(tsv, _make_result(0, "baseline", val_bpb=1.4, status="baseline"))
        # 25 kept experiments with tiny val_bpb variation
        for i in range(1, 26):
            bpb = 1.300 + (i % 3) * 0.0001
            append_result(tsv, _make_result(i, f"Tweak lr variant {i}",
                                            val_bpb=bpb, status="keep"))
        return tsv

    def test_check_from_file(self, tmp_path):
        tsv = self._write_plateau_tsv(tmp_path)
        monitor = DriftMonitor(DriftConfig(
            performance=PerformanceConfig(plateau_window=15, regression_patience=10),
        ))
        report = monitor.check_from_file(tsv)

        assert isinstance(report, DriftReport)
        assert report.has_alerts
        # Should detect plateau (slope near zero over 15+ kept experiments)
        perf_messages = [a.message for a in report.performance]
        assert any("plateau" in m.lower() for m in perf_messages)

    def test_inject_into_prompt(self, tmp_path):
        tsv = self._write_plateau_tsv(tmp_path)
        results = load_results(tsv)
        monitor = DriftMonitor(DriftConfig(
            performance=PerformanceConfig(plateau_window=15, regression_patience=10),
        ))

        base_prompt = "You are an autonomous researcher."
        injected = monitor.inject_into_prompt(base_prompt, results)

        assert injected.startswith(base_prompt)
        assert "DRIFT MONITOR ALERTS" in injected
        assert "END DRIFT ALERTS" in injected

    def test_format_summary(self, tmp_path):
        tsv = self._write_plateau_tsv(tmp_path)
        monitor = DriftMonitor(DriftConfig(
            performance=PerformanceConfig(plateau_window=15, regression_patience=10),
        ))
        report = monitor.check_from_file(tsv)

        summary = report.format_summary()
        assert "Drift alerts:" in summary
        # Should mention counts
        assert any(x in summary for x in ["critical", "warnings", "info"])

    def test_no_drift_on_steady_improvement(self, tmp_path):
        """Steadily improving results should NOT trigger drift alerts."""
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)
        append_result(tsv, _make_result(0, "baseline", val_bpb=1.5, status="baseline"))
        categories = ["batch_size", "learning_rate", "architecture",
                       "schedule", "regularization"]
        for i in range(1, 21):
            cat = categories[i % len(categories)]
            descs = {
                "batch_size": f"Adjust batch_size v{i}",
                "learning_rate": f"Tune MATRIX_LR v{i}",
                "architecture": f"Change DEPTH v{i}",
                "schedule": f"Modify warmup v{i}",
                "regularization": f"Adjust weight_decay v{i}",
            }
            bpb = 1.5 - i * 0.02  # steady improvement
            append_result(tsv, _make_result(i, descs[cat], val_bpb=bpb,
                                            status="keep" if i % 3 != 0 else "discard"))

        monitor = DriftMonitor()
        report = monitor.check_from_file(tsv)

        # Performance plateau should NOT fire
        perf_plateau = [a for a in report.performance if "plateau" in a.message.lower()]
        assert len(perf_plateau) == 0


# ===========================================================================
# L2.3 — Tool wrappers
# ===========================================================================

class TestToolWrappers:
    """Test analyze, drift, and hardware tools with real data."""

    def _setup_tsv(self, tmp_path) -> str:
        tsv = str(tmp_path / "results.tsv")
        init_results_tsv(tsv)
        append_result(tsv, _make_result(0, "baseline", val_bpb=1.4, status="baseline"))
        append_result(tsv, _make_result(1, "Increase MATRIX_LR", val_bpb=1.35, status="keep"))
        append_result(tsv, _make_result(2, "Reduce batch_size", val_bpb=1.42, status="discard"))
        append_result(tsv, _make_result(3, "Increase DEPTH", val_bpb=1.30, status="keep"))
        append_result(tsv, _make_result(4, "Bad experiment", val_bpb=0.0, status="crash"))
        return tsv

    def test_analyze_summary(self, tmp_path):
        from tools.analyze_tool import analyze_experiment_history
        tsv = self._setup_tsv(tmp_path)
        with patch.dict(os.environ, {"RESULTS_TSV": tsv}):
            # Need to reload module-level RESULTS_PATH
            import tools.analyze_tool as at
            at.RESULTS_PATH = tsv
            result = at.analyze_experiment_history("summary")

        data = json.loads(result)
        assert data["total_experiments"] == 5
        assert data["kept"] == 2
        assert data["discarded"] == 1
        assert data["crashes"] == 1
        assert data["best_val_bpb"] == pytest.approx(1.30, abs=0.01)

    def test_analyze_best(self, tmp_path):
        from tools.analyze_tool import analyze_experiment_history
        tsv = self._setup_tsv(tmp_path)
        import tools.analyze_tool as at
        at.RESULTS_PATH = tsv

        result = at.analyze_experiment_history("best")
        assert "1.30" in result
        assert "exp3" in result

    def test_analyze_categories(self, tmp_path):
        from tools.analyze_tool import analyze_experiment_history
        tsv = self._setup_tsv(tmp_path)
        import tools.analyze_tool as at
        at.RESULTS_PATH = tsv

        result = at.analyze_experiment_history("categories")
        assert "learning_rate" in result
        assert "batch_size" in result
        assert "architecture" in result

    def test_analyze_recent(self, tmp_path):
        from tools.analyze_tool import analyze_experiment_history
        tsv = self._setup_tsv(tmp_path)
        import tools.analyze_tool as at
        at.RESULTS_PATH = tsv

        result = at.analyze_experiment_history("recent")
        assert "exp0" in result
        assert "exp4" in result
        assert "crash" in result

    def test_analyze_full(self, tmp_path):
        from tools.analyze_tool import analyze_experiment_history
        tsv = self._setup_tsv(tmp_path)
        import tools.analyze_tool as at
        at.RESULTS_PATH = tsv

        result = at.analyze_experiment_history("full")
        assert "Strategy summary" in result
        assert "baseline" in result

    def test_drift_summary(self, tmp_path):
        from tools.drift_tool import check_drift_status
        tsv = self._setup_tsv(tmp_path)
        import tools.drift_tool as dt
        dt.RESULTS_PATH = tsv

        result = dt.check_drift_status("summary")
        # With only 5 experiments, most drift detectors won't fire
        assert isinstance(result, str)
        assert len(result) > 0

    def test_drift_full(self, tmp_path):
        from tools.drift_tool import check_drift_status
        tsv = self._setup_tsv(tmp_path)
        import tools.drift_tool as dt
        dt.RESULTS_PATH = tsv

        result = dt.check_drift_status("full")
        assert isinstance(result, str)
        # Either "No drift detected" or a report with icons
        assert "drift" in result.lower() or "nominal" in result.lower()

    def test_drift_no_results(self, tmp_path):
        from tools.drift_tool import check_drift_status
        import tools.drift_tool as dt
        dt.RESULTS_PATH = str(tmp_path / "nonexistent.tsv")

        result = dt.check_drift_status()
        assert "No experiments" in result

    def test_hardware_profile(self):
        from tools.hardware_tool import get_hardware_profile

        result = get_hardware_profile()
        data = json.loads(result)
        assert "chip_name" in data
        assert "gpu_cores" in data
        assert "memory_gb" in data
        assert "peak_tflops" in data


# ===========================================================================
# L2.4 — LLM backend factory
# ===========================================================================

class TestLLMBackendFactory:
    """Test provider selection via environment variables (no actual API calls)."""

    def _clean_env(self):
        """Return env dict with all LLM keys removed."""
        keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_DEPLOYMENT",
                "OLLAMA_MODEL"]
        return {k: "" for k in keys}

    @pytest.mark.skipif(
        not _has_openai(), reason="openai package not installed"
    )
    def test_openai_backend(self):
        from autoresearch.llm_backend import get_llm_backend, OpenAIBackend
        env = self._clean_env()
        env["OPENAI_API_KEY"] = "sk-test-fake-key"
        with patch.dict(os.environ, env, clear=False):
            backend = get_llm_backend()
        assert isinstance(backend, OpenAIBackend)

    @pytest.mark.skipif(
        not _has_openai(), reason="openai package not installed"
    )
    def test_openrouter_backend(self):
        from autoresearch.llm_backend import get_llm_backend, OpenRouterBackend
        env = self._clean_env()
        env["OPENROUTER_API_KEY"] = "sk-or-test-fake"
        with patch.dict(os.environ, env, clear=False):
            backend = get_llm_backend()
        assert isinstance(backend, OpenRouterBackend)

    @pytest.mark.skipif(
        not _has_openai(), reason="openai package not installed"
    )
    def test_azure_backend(self):
        from autoresearch.llm_backend import get_llm_backend, AzureOpenAIBackend
        env = self._clean_env()
        env["AZURE_OPENAI_API_KEY"] = "azure-test-key"
        env["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
        with patch.dict(os.environ, env, clear=False):
            backend = get_llm_backend()
        assert isinstance(backend, AzureOpenAIBackend)

    def test_ollama_backend(self):
        from autoresearch.llm_backend import get_llm_backend, OllamaBackend
        env = self._clean_env()
        env["OLLAMA_MODEL"] = "llama3.3"
        with patch.dict(os.environ, env, clear=False):
            backend = get_llm_backend()
        assert isinstance(backend, OllamaBackend)

    def test_no_backend_raises(self):
        from autoresearch.llm_backend import get_llm_backend
        env = self._clean_env()
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(RuntimeError, match="No LLM backend configured"):
                get_llm_backend()

    def test_priority_order(self):
        """Ollama > OpenRouter > Azure > OpenAI > Claude."""
        from autoresearch.llm_backend import get_llm_backend, OllamaBackend
        env = self._clean_env()
        env["OLLAMA_MODEL"] = "llama3.3"
        env["OPENROUTER_API_KEY"] = "sk-or-test"
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env, clear=False):
            backend = get_llm_backend()
        # Ollama has highest priority
        assert isinstance(backend, OllamaBackend)

    @pytest.mark.skipif(
        not _has_openai(), reason="openai package not installed"
    )
    def test_openrouter_over_openai(self):
        from autoresearch.llm_backend import get_llm_backend, OpenRouterBackend
        env = self._clean_env()
        env["OPENROUTER_API_KEY"] = "sk-or-test"
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env, clear=False):
            backend = get_llm_backend()
        assert isinstance(backend, OpenRouterBackend)


# ===========================================================================
# L2.5 — LLM response parsing
# ===========================================================================

class TestLLMResponseParsing:
    """Test parse_llm_response() with various response formats."""

    def test_well_formed_response(self):
        from autoresearch.llm_backend import parse_llm_response

        response = """\
DESCRIPTION: Increase MATRIX_LR from 0.04 to 0.06
REASONING: Higher learning rate may help the model converge faster within the 5-minute budget. The current rate might be too conservative.
CODE:
# --- hyperparameter block start ---
MATRIX_LR = 0.06
SCALAR_LR = 0.01
# --- hyperparameter block end ---"""

        proposal = parse_llm_response(response)
        assert proposal.description == "Increase MATRIX_LR from 0.04 to 0.06"
        assert "converge faster" in proposal.reasoning
        assert "MATRIX_LR = 0.06" in proposal.code
        assert "SCALAR_LR = 0.01" in proposal.code

    def test_response_with_markdown_fences(self):
        from autoresearch.llm_backend import parse_llm_response

        response = """\
DESCRIPTION: Reduce batch_size from 64 to 32
REASONING: Smaller batches mean more gradient steps in the same time budget.
CODE:
```python
# --- hyperparameter block start ---
TOTAL_BATCH_SIZE = 32
DEVICE_BATCH_SIZE = 16
# --- hyperparameter block end ---
```"""

        proposal = parse_llm_response(response)
        assert proposal.description == "Reduce batch_size from 64 to 32"
        assert "TOTAL_BATCH_SIZE = 32" in proposal.code
        # Markdown fences should be stripped
        assert "```" not in proposal.code

    def test_missing_description_raises(self):
        from autoresearch.llm_backend import parse_llm_response

        response = """\
REASONING: Some reasoning.
CODE:
some code"""

        with pytest.raises(ValueError, match="DESCRIPTION"):
            parse_llm_response(response)

    def test_missing_code_raises(self):
        from autoresearch.llm_backend import parse_llm_response

        # REASONING regex requires \nCODE: lookahead, so without CODE:
        # the parser fails on REASONING first. Test that *some* ValueError fires.
        response = """\
DESCRIPTION: Do something
REASONING: For some reason."""

        with pytest.raises(ValueError):
            parse_llm_response(response)

    def test_extra_whitespace(self):
        from autoresearch.llm_backend import parse_llm_response

        response = """\
DESCRIPTION:   Increase DEPTH from 12 to 16
REASONING:   Deeper models capture more complex patterns.
CODE:

# --- hyperparameter block start ---
DEPTH = 16
# --- hyperparameter block end ---
  """

        proposal = parse_llm_response(response)
        assert proposal.description == "Increase DEPTH from 12 to 16"
        assert "DEPTH = 16" in proposal.code
