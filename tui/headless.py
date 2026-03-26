"""Headless runner for the experiment orchestrator.

Runs the full autonomous experiment loop without a TUI -- output goes to
stdout/stderr. Used by run_suite.py for unattended overnight runs where
Textual's terminal requirements would cause OSError [Errno 5].

Resilience features:
- Signal handlers for graceful shutdown (SIGINT/SIGTERM/SIGHUP)
- Heartbeat file for external monitoring
- try/finally to ensure heartbeat cleanup on any exit

Platform-agnostic: training script is auto-detected from backends.registry
if not explicitly provided.

Usage:
    # As a module (from run_suite.py):
    from tui.headless import run_headless
    success = run_headless(tag="mar17-fineweb", max_experiments=80)

    # As a standalone script:
    python -m tui.headless --tag mar17 --max 80
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tui.orchestrator import ExperimentOrchestrator, OrchestratorCallbacks
from tui.resilience import install_signal_handlers
from tui.results import ExperimentResult


def _timestamp():
    return datetime.now().strftime("%H:%M:%S")


def _make_callbacks():
    """Create callbacks that print to stdout."""

    def on_status_change(status, message):
        icon = {
            "initializing": "[INIT]",
            "baseline": "[BASE]",
            "thinking": "[THINK]",
            "committing": "[COMMIT]",
            "training": "[TRAIN]",
            "evaluating": "[EVAL]",
            "stopped": "[STOP]",
        }.get(status, "[...]")
        print(f"[{_timestamp()}] {icon} {message}", flush=True)

    def on_experiment_start(exp_num, desc, reasoning):
        print(f"\n[{_timestamp()}] === Experiment {exp_num}: {desc}", flush=True)
        if reasoning:
            first_line = reasoning.split("\n")[0][:120]
            print(f"[{_timestamp()}]     Reasoning: {first_line}", flush=True)

    def on_training_output(line):
        line = line.strip()
        if not line:
            return
        if "step " in line:
            try:
                step_str = line.split("step ")[1].split(" ")[0]
                step_num = int(step_str)
                if step_num % 50 == 0 or step_num <= 2:
                    print(f"  {line[:120]}", flush=True)
            except (ValueError, IndexError):
                pass
        elif "val_bpb" in line or "---" in line:
            print(f"  {line}", flush=True)

    def on_experiment_complete(result: ExperimentResult):
        if result.status == "keep":
            print(f"[{_timestamp()}] KEEP -- val_bpb={result.val_bpb:.6f} | "
                  f"mem={result.peak_mem_gb:.1f}GB | tok/s={result.tok_sec:,} | "
                  f"mfu={result.mfu:.1f}% | steps={result.steps}", flush=True)
        elif result.status == "discard":
            print(f"[{_timestamp()}] DISCARD -- val_bpb={result.val_bpb:.6f}", flush=True)
        elif result.status == "crash":
            print(f"[{_timestamp()}] CRASH -- {result.notes}", flush=True)
        elif result.status == "baseline":
            print(f"[{_timestamp()}] BASELINE -- val_bpb={result.val_bpb:.6f} | "
                  f"mem={result.peak_mem_gb:.1f}GB | tok/s={result.tok_sec:,} | "
                  f"mfu={result.mfu:.1f}% | steps={result.steps}", flush=True)

    def on_stats_update(total, kept, discarded, best_bpb):
        bpb_str = f"{best_bpb:.6f}" if best_bpb < float("inf") else "--"
        print(f"[{_timestamp()}] Stats: Total={total} Kept={kept} Discarded={discarded} Best={bpb_str}", flush=True)

    def on_error(message):
        print(f"[{_timestamp()}] ERROR: {message}", file=sys.stderr, flush=True)

    return OrchestratorCallbacks(
        on_status_change=on_status_change,
        on_experiment_start=on_experiment_start,
        on_training_output=on_training_output,
        on_experiment_complete=on_experiment_complete,
        on_stats_update=on_stats_update,
        on_error=on_error,
    )


def run_headless(
    training_script: str | None = None,
    results_path: str = "results.tsv",
    tag: str | None = None,
    max_experiments: int = 80,
    model: str | None = None,
    dataset_name: str = "",
) -> bool:
    """Run the orchestrator headless (no TUI).

    Args:
        training_script: Path to training script. Auto-detected if None.
        model: Optional Claude model override (e.g. "claude-sonnet-4-20250514").
               Falls back to CLAUDE_MODEL env var, then default.
        dataset_name: Name of the dataset being trained on (for heartbeat/logging).

    Returns True if the run completed without fatal errors.
    """
    tag = tag or time.strftime("%b%d").lower()
    callbacks = _make_callbacks()

    print(f"\n{'='*60}")
    print(f"  Headless Experiment Runner")
    print(f"  Tag: {tag}")
    print(f"  Dataset: {dataset_name or 'unknown'}")
    print(f"  Training script: {training_script or '(auto-detect)'}")
    print(f"  Results: {results_path}")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    orchestrator = ExperimentOrchestrator(
        training_script=training_script,
        results_path=results_path,
        max_experiments=max_experiments,
        run_tag=tag,
        callbacks=callbacks,
        model=model,
        dataset_name=dataset_name,
    )

    # Install signal handlers for graceful shutdown
    install_signal_handlers(orchestrator.stop)

    try:
        orchestrator._run_loop()
    finally:
        orchestrator.cleanup()

    print(f"\n{'='*60}")
    print(f"  Run complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total: {orchestrator.total_runs} experiments")
    print(f"  Kept: {orchestrator.kept_count}")
    print(f"  Discarded: {orchestrator.discarded_count}")
    print(f"  Crashes: {orchestrator.crash_count}")
    if orchestrator.best_val_bpb < float("inf"):
        print(f"  Best val_bpb: {orchestrator.best_val_bpb:.6f} ({orchestrator.best_experiment})")
    print(f"{'='*60}\n")

    return orchestrator.total_runs > 0


def main():
    parser = argparse.ArgumentParser(description="Headless experiment runner (no TUI)")
    parser.add_argument("--tag", type=str, default=None, help="Run tag")
    parser.add_argument("--max", type=int, default=80, help="Max experiments")
    parser.add_argument("--training-script", default=None, help="Training script path (auto-detected if omitted)")
    parser.add_argument("--results", default="results.tsv", help="Results TSV path")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name (for heartbeat/logging)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model override (e.g. 'claude-opus-4-6', 'gpt-4.1', 'anthropic/claude-sonnet-4.6' for OpenRouter). Provider auto-detected from API key env vars.")
    args = parser.parse_args()

    # PID lock -- prevent duplicate experiment runs
    from tui.resilience import acquire_pidlock, release_pidlock
    if not acquire_pidlock():
        sys.exit(1)
    import atexit
    atexit.register(release_pidlock)

    success = run_headless(
        training_script=args.training_script,
        results_path=args.results,
        tag=args.tag,
        max_experiments=args.max,
        model=args.model,
        dataset_name=args.dataset,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
