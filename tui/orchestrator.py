"""Autonomous experiment orchestrator.

Drives the experiment loop: LLM generates code -> commit -> train -> evaluate ->
keep/discard -> repeat. Runs in a background thread and communicates with the
TUI via callbacks.

Resilience features:
- Resumes from interrupted runs (reads existing results.tsv)
- Exponential backoff on API errors (rate limits, 5xx, connection issues)
- Mid-experiment cleanup on startup (reverts orphaned commits)
- Heartbeat file for external monitoring

Platform-agnostic: the training script is determined by backends.registry
based on the detected or configured backend.
"""

import os
import re
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from tui.git_manager import GitManager
from tui.hardware import get_hardware_summary
from tui.llm_backend import LLMBackend, get_llm_backend
from tui.parser import OutputParser, StepMetrics, FinalMetrics
from tui.resilience import Heartbeat
from tui.results import (
    ExperimentResult,
    append_result,
    classify_experiment,
    format_history_for_prompt,
    get_best_result,
    init_results_tsv,
    load_results,
    next_experiment_number,
)


# Marker comments that delimit the hyperparameter block in training scripts.
# These are identical across all platform training scripts.
HP_BLOCK_START = "# ---------------------------------------------------------------------------\n# Hyperparameters"
HP_BLOCK_END = "# ---------------------------------------------------------------------------\n# Setup"


@dataclass
class OrchestratorCallbacks:
    """Callbacks for communicating with the TUI.

    All callbacks are called from the orchestrator's background thread.
    The TUI app must wrap them with call_from_thread() for thread safety.
    """
    on_status_change: Callable[[str, str], None]       # (status, message)
    on_experiment_start: Callable[[int, str, str], None]  # (exp_num, desc, reasoning)
    on_training_output: Callable[[str], None]           # raw line from training stdout
    on_experiment_complete: Callable[[ExperimentResult], None]
    on_stats_update: Callable[[int, int, int, float], None]  # (total, kept, discarded, best_bpb)
    on_error: Callable[[str], None]


def _default_training_script() -> str:
    """Determine the training script based on detected backend."""
    from backends import detect_backend
    from backends.registry import get_training_script
    return get_training_script(detect_backend())


class ExperimentOrchestrator:
    """Manages the autonomous experiment loop.

    Runs in a dedicated background thread. Uses the LLM to generate
    code modifications, trains, evaluates, and decides keep/discard.

    Resilience: if the process dies and restarts, it reads results.tsv
    to determine the last completed experiment and resumes from there.
    """

    def __init__(
        self,
        training_script: str | None = None,
        results_path: str = "results.tsv",
        max_experiments: int = 100,
        run_tag: str | None = None,
        callbacks: OrchestratorCallbacks | None = None,
        model: str | None = None,
        dataset_name: str = "",
    ):
        self._training_script = training_script or _default_training_script()
        self._results_path = results_path
        self._max_experiments = max_experiments
        self._run_tag = run_tag or time.strftime("%b%d").lower()
        self._callbacks = callbacks
        self._model_override = model  # e.g. "claude-sonnet-4-6"
        self._dataset_name = dataset_name

        self._git = GitManager()
        self._git.ensure_auto_push_remote()  # prevent silent push failures on new branches
        self._hw_info = get_hardware_summary()
        self._llm: LLMBackend | None = None  # lazy init
        self._last_sync_time = 0.0  # epoch time of last git push

        # Heartbeat for external monitors
        heartbeat_dir = os.path.dirname(os.path.abspath(results_path)) if results_path else "."
        self._heartbeat = Heartbeat(os.path.join(heartbeat_dir, ".runner_status.json"))

        # State
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

        # Counters
        self.total_runs = 0
        self.kept_count = 0
        self.discarded_count = 0
        self.crash_count = 0
        self.best_val_bpb = float("inf")
        self.best_experiment = "none"
        self._baseline_sha: str | None = None  # commit SHA with zero HP changes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the experiment loop in a background thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="orchestrator",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the orchestrator to stop after the current experiment."""
        self._stop_event.set()
        # Kill any running training subprocess
        with self._lock:
            if self._proc and self._proc.returncode is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

    def cleanup(self) -> None:
        """Final cleanup -- close heartbeat, etc."""
        self._heartbeat.close()

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Callbacks (thread-safe dispatchers)
    # ------------------------------------------------------------------

    def _cb_status(self, status: str, message: str) -> None:
        if self._callbacks:
            self._callbacks.on_status_change(status, message)

    def _cb_experiment_start(self, exp_num: int, desc: str, reasoning: str) -> None:
        if self._callbacks:
            self._callbacks.on_experiment_start(exp_num, desc, reasoning)

    def _cb_training_output(self, line: str) -> None:
        if self._callbacks:
            self._callbacks.on_training_output(line)

    def _cb_experiment_complete(self, result: ExperimentResult) -> None:
        if self._callbacks:
            self._callbacks.on_experiment_complete(result)

    def _cb_stats_update(self) -> None:
        if self._callbacks:
            self._callbacks.on_stats_update(
                self.total_runs, self.kept_count,
                self.discarded_count, self.best_val_bpb,
            )

    def _cb_error(self, message: str) -> None:
        if self._callbacks:
            self._callbacks.on_error(message)

    def _update_heartbeat(self, experiment: int = 0, status: str = "running") -> None:
        """Update heartbeat with current state."""
        model_name = self._model_override or "default"
        self._heartbeat.update(
            experiment=experiment,
            status=status,
            dataset=self._dataset_name,
            best_bpb=self.best_val_bpb,
            model=model_name,
            total=self.total_runs,
            kept=self.kept_count,
            discarded=self.discarded_count,
            crashes=self.crash_count,
        )

    # ------------------------------------------------------------------
    # Resume and cleanup
    # ------------------------------------------------------------------

    def _restore_counters(self) -> None:
        """Restore kept/discarded/crash counters from existing results."""
        results = load_results(self._results_path)
        self.total_runs = len(results)
        self.kept_count = sum(1 for r in results if r.status in ("keep", "baseline"))
        self.discarded_count = sum(1 for r in results if r.status == "discard")
        self.crash_count = sum(1 for r in results if r.status == "crash")

    def _cleanup_interrupted_experiment(self) -> None:
        """Clean up state from a previously interrupted experiment.

        Handles two cases:
        1. Uncommitted changes (code modified, never committed) -> restore file
        2. Orphaned commit (committed but training never finished) -> revert
        """
        # Case 1: Dirty working tree
        if self._git.has_uncommitted_changes():
            self._cb_status("initializing", "Cleaning up uncommitted changes from interrupted run")
            try:
                self._git.reset_working_tree(self._training_script)
            except Exception as e:
                self._cb_error(f"Failed to clean working tree: {e}")

        # Case 2: Orphaned commit (committed but no results recorded)
        try:
            head_msg = self._git.head_commit_message()
            match = re.match(r"exp(\d+):", head_msg)
            if match:
                exp_num = int(match.group(1))
                results = load_results(self._results_path)
                recorded = {int(r.exp.replace("exp", ""))
                           for r in results if r.exp.startswith("exp")}
                if exp_num not in recorded:
                    self._cb_status("initializing",
                        f"Reverting orphaned commit: {head_msg[:60]}")
                    self._git.revert_last_experiment()
        except Exception as e:
            self._cb_error(f"Orphan commit check failed: {e}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main experiment loop (runs in background thread)."""
        self._running = True

        try:
            # Initialize LLM backend
            self._cb_status("initializing", "Connecting to LLM backend...")
            try:
                self._llm = get_llm_backend(model=self._model_override)
                self._cb_status("initializing", f"LLM: {self._llm.name()}")
            except Exception as e:
                self._cb_error(f"LLM backend error: {e}")
                return

            # Validate credentials with a preflight API call (with retry).
            # A single-shot validate() caused issue #9: a transient API 529
            # during validation abandoned the entire dataset run.
            self._cb_status("initializing", "Validating API credentials...")
            validated = False
            for attempt in range(5):
                try:
                    if self._llm.validate():
                        validated = True
                        break
                    else:
                        # validate() returned False — credentials are bad (not transient)
                        source = getattr(self._llm, '_cred_source', 'unknown')
                        self._cb_error(
                            f"API credential validation failed (source: {source}). "
                            f"Run: python dashboard.py --setup-key"
                        )
                        return
                except Exception as e:
                    err_str = str(e).lower()
                    # Fatal auth/billing errors — don't retry
                    if any(k in err_str for k in [
                        "authentication", "401", "invalid_api_key",
                        "credit balance", "insufficient_quota",
                    ]):
                        self._cb_error(f"API credential error (fatal): {e}")
                        return
                    # Transient errors (529, 5xx, connection) — retry with backoff
                    wait = min(15 * (2 ** attempt), 120)
                    self._cb_status("initializing",
                        f"API validation error (attempt {attempt+1}/5, retrying in {wait}s): {e}")
                    for _ in range(max(1, wait // 5)):
                        if self._stop_event.is_set():
                            return
                        time.sleep(5)

            if not validated:
                self._cb_error("API validation failed after 5 attempts — aborting")
                return
            self._cb_status("initializing", "API credentials validated")

            # Ensure new branches auto-track remote (prevents sync failures)
            self._git.ensure_auto_push_remote()

            # Set up branch
            branch_name = f"autoresearch/{self._run_tag}"
            current = self._git.current_branch()
            if current != branch_name:
                if self._git.branch_exists(branch_name):
                    self._cb_status("initializing", f"Checking out existing branch {branch_name}")
                    self._git.checkout(branch_name)
                else:
                    self._cb_status("initializing", f"Creating branch {branch_name}")
                    self._git.create_branch(branch_name)

            # Clean up any interrupted experiment from a previous run
            self._cleanup_interrupted_experiment()

            # Initialize results (validates existing file)
            init_results_tsv(self._results_path)

            # Load existing state (for resuming)
            existing_best, existing_exp = get_best_result(self._results_path)
            if existing_best < float("inf"):
                self.best_val_bpb = existing_best
                self.best_experiment = existing_exp

            # Restore counters from existing results
            self._restore_counters()

            start_exp = next_experiment_number(self._results_path)

            if start_exp > 0:
                self._cb_status("initializing",
                    f"Resuming from exp{start_exp} -- "
                    f"{self.total_runs} completed, best: {self.best_val_bpb:.4f}")

            self._update_heartbeat(start_exp, "initializing")

            # Run baseline if this is a fresh start
            if start_exp == 0:
                # Ensure the training script is at zero modifications before
                # starting a new dataset run. This prevents HP optimizations
                # from a previous dataset leaking into this one.
                self._ensure_clean_baseline()
                self._cb_status("baseline", "Running baseline (no modifications)...")
                self._run_baseline()
                start_exp = 1

                if self._stop_event.is_set():
                    return

            # Main experiment loop
            # max_experiments is the TOTAL target, not "more from here"
            for exp_num in range(start_exp, self._max_experiments):
                if self._stop_event.is_set():
                    break

                # Hard gate: check actual TSV row count to prevent overshoot
                # after crash/restart where lost rows reset start_exp lower.
                current_count = len(load_results(self._results_path))
                if current_count >= self._max_experiments:
                    self._cb_status("stopped",
                        f"Experiment limit reached: {current_count} rows "
                        f">= {self._max_experiments} max")
                    break

                self._update_heartbeat(exp_num, "running")
                self._run_experiment(exp_num)

                if self._stop_event.is_set():
                    break

                # Brief pause between experiments
                time.sleep(2)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._cb_error(f"Orchestrator error: {e}")
            self._cb_error(f"Traceback: {tb}")
        finally:
            self._running = False
            self._update_heartbeat(status="stopped")
            self._maybe_sync_results(force=True)  # final sync before exit
            self._cb_status("stopped", "Experiment loop stopped")

    # ------------------------------------------------------------------
    # Baseline enforcement
    # ------------------------------------------------------------------

    def _ensure_clean_baseline(self) -> None:
        """Ensure the training script has zero HP modifications.

        Called at the START of a new dataset run (before the baseline
        experiment). This prevents optimizations from a previous dataset
        run leaking into this one.

        How it works:
        - Reads the training script from the FIRST commit on the branch
          that introduced it (i.e., the initial unmodified version)
        - If the current working-tree copy differs from that version,
          restores it and commits the reset

        This is the cross-dataset isolation boundary. Within a single
        dataset run, KEPT experiments accumulate as intended.
        """
        try:
            # Get the original training script content from the branch point
            # (the commit where the branch diverged from main, or the first
            # commit that contains the training script).
            script_path = self._training_script
            original = self._git._run(
                "show", f"main:{script_path}", check=False
            )
            if not original:
                # Fallback: try the first commit that touched this file
                log_output = self._git._run(
                    "log", "--follow", "--diff-filter=A",
                    "--format=%H", "--", script_path, check=False
                ).strip()
                first_commit = log_output.splitlines()[-1] if log_output else ""
                if first_commit:
                    original = self._git._run("show", f"{first_commit}:{script_path}")
                else:
                    self._cb_status("baseline",
                        "Could not determine original training script — proceeding with current version")
                    return

            # Compare with current working tree
            with open(script_path) as f:
                current = f.read()

            if current.strip() == original.strip():
                self._cb_status("baseline", "Training script is clean (zero modifications)")
                return

            # Reset to clean state
            self._cb_status("baseline",
                f"Resetting {os.path.basename(script_path)} to unmodified state (cross-dataset isolation)")
            with open(script_path, "w") as f:
                f.write(original if original.endswith("\n") else original + "\n")

            # Commit the reset so the git history is clear
            self._git.commit_changes(
                "Reset training script to zero modifications (new dataset run)",
                [script_path],
            )
        except Exception as e:
            self._cb_error(f"Baseline reset warning: {e} — proceeding with current state")

    # ------------------------------------------------------------------
    # Baseline run
    # ------------------------------------------------------------------

    def _run_baseline(self) -> None:
        """Run the training script without modifications to establish baseline.

        Records the current commit as the baseline SHA — the zero-modifications
        state. This SHA is written into every experiment result for traceability.
        Within the run, KEPT experiments accumulate on top of this baseline.
        """
        # Record the baseline commit AFTER _ensure_clean_baseline has run.
        # This is the "zero optimizations" state of the training script.
        self._baseline_sha = self._git.record_baseline()
        self._cb_status("baseline",
            f"Baseline commit: {self._baseline_sha[:7]} (training script at zero modifications)")

        self._cb_experiment_start(0, "baseline (no modifications)", "Establishing baseline with current defaults")
        self._update_heartbeat(0, "baseline")

        final = self._run_training()

        if final:
            result = ExperimentResult(
                exp="exp0",
                description="baseline (no modifications)",
                val_bpb=final.val_bpb,
                peak_mem_gb=final.peak_vram_mb / 1024,
                tok_sec=int(final.total_tokens_M * 1e6 / final.training_seconds) if final.training_seconds > 0 else 0,
                mfu=final.mfu_percent,
                steps=final.num_steps,
                status="baseline",
                notes=f"depth={final.depth}, {final.chip}",
                gpu_name=self._hw_info.get("chip_name", "unknown"),
                baseline_sha=self._baseline_sha,
                watts=final.avg_watts,
                joules_per_token=final.joules_per_token,
                total_energy_joules=final.total_energy_j,
            )
            self.best_val_bpb = final.val_bpb
            self.best_experiment = "exp0 (baseline)"
        else:
            result = ExperimentResult(
                exp="exp0",
                description="baseline (no modifications)",
                val_bpb=0.0, peak_mem_gb=0.0, tok_sec=0,
                mfu=0.0, steps=0, status="crash",
                notes="baseline training failed",
                gpu_name=self._hw_info.get("chip_name", "unknown"),
                baseline_sha=self._baseline_sha,
            )

        append_result(self._results_path, result)
        self.total_runs += 1
        if result.status == "baseline":
            self.kept_count += 1
        self._cb_experiment_complete(result)
        self._cb_stats_update()

    # ------------------------------------------------------------------
    # Stagnation detection
    # ------------------------------------------------------------------

    def _is_near_duplicate(self, description: str) -> bool:
        """Check if a very similar experiment was already attempted.

        Catches exact duplicates and same-parameter-same-direction proposals
        (e.g. "Decrease MATRIX_LR from 0.04 to 0.03" vs "Lower MATRIX_LR
        from 0.04 to 0.03"). This prevents the LLM from wasting experiments
        on proposals it has already tried.
        """
        results = load_results(self._results_path)
        desc_lower = description.lower().strip()

        for r in results:
            existing = r.description.lower().strip()

            # Exact match (case-insensitive)
            if desc_lower == existing:
                return True

            # Same hyperparameter + same target value
            # Extract param=value patterns like "MATRIX_LR to 0.03"
            if self._same_param_same_value(desc_lower, existing):
                return True

        return False

    @staticmethod
    def _same_param_same_value(desc_a: str, desc_b: str) -> bool:
        """Check if two descriptions modify the same param to the same value.

        Matches patterns like "PARAM_NAME ... to 0.03". The param name must
        contain an underscore (to avoid matching regular English words) and
        the target value must follow "to".
        """
        import re as _re

        def _extract_pairs(text: str) -> set[tuple[str, str]]:
            """Extract (PARAM_NAME, target_value) pairs from a description."""
            pairs: set[tuple[str, str]] = set()
            # Find all HP-style param names (must contain underscore)
            params = _re.finditer(r'\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b', text, _re.IGNORECASE)
            for m in params:
                param = m.group(1).upper()
                # Look for "to <number>" after the param name in the text
                after = text[m.end():]
                val_match = _re.search(r'\bto\s+([0-9]+\.?[0-9]*)', after)
                if val_match:
                    pairs.add((param, val_match.group(1)))
            return pairs

        pairs_a = _extract_pairs(desc_a)
        pairs_b = _extract_pairs(desc_b)

        if not pairs_a or not pairs_b:
            return False

        return bool(pairs_a & pairs_b)

    def _detect_stagnation(self) -> str | None:
        """Detect if the LLM is stuck in a narrow strategy space.

        Looks at the last 15 experiments. If very few were kept and
        most were learning rate changes, returns a nudge message to
        append to the prompt. Otherwise returns None.
        """
        results = load_results(self._results_path)
        if len(results) < 15:
            return None

        recent = results[-15:]
        recent_keeps = sum(1 for r in recent if r.status == "keep")

        if recent_keeps > 1:
            return None

        # Count how many recent experiments were learning rate changes
        lr_count = sum(
            1 for r in recent
            if r.status != "baseline"
            and classify_experiment(r.description) == "learning_rate"
        )

        if lr_count >= 8:
            return (
                "\n\nIMPORTANT: The last 15 experiments have yielded only "
                f"{recent_keeps} improvement(s), and {lr_count} were learning rate changes. "
                "Learning rate tuning appears exhausted. Try a fundamentally different "
                "approach: batch size changes, architectural modifications (DEPTH, "
                "WINDOW_PATTERN, HEAD_DIM), or schedule shape changes (WARMUP_RATIO>0, "
                "FINAL_LR_FRAC>0, ADAM_BETAS)."
            )
        return None

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------

    def _run_experiment(self, exp_num: int, _pause_depth: int = 0) -> None:
        """Run a single experiment: LLM -> modify -> commit -> train -> evaluate.

        Within a dataset run, KEPT experiments accumulate — the LLM builds on
        previous successes to find the best HP combination for this dataset.
        DISCARDED/CRASHED experiments are reverted to the last kept state.

        The baseline SHA is tracked for traceability but is NOT reset between
        experiments within the same run. Baseline reset happens at the START
        of a new dataset run (see _ensure_clean_baseline).

        If the API is completely unavailable after retries, pauses for 10
        minutes and retries (up to 3 pause cycles to avoid infinite loops).
        """

        # 1. Ask the LLM for a modification
        self._cb_status("thinking", f"Claude is designing experiment {exp_num}...")
        self._update_heartbeat(exp_num, "thinking")

        current_code = self._extract_hp_block()
        results_history = format_history_for_prompt(self._results_path)

        # Append stagnation nudge if the LLM is stuck in a narrow strategy space
        nudge = self._detect_stagnation()
        if nudge:
            results_history += nudge

        proposal = self._call_llm_with_backoff(
            current_code, results_history, exp_num
        )

        if proposal is None:
            # All retries exhausted -- pause and retry if we haven't exceeded depth
            if _pause_depth < 3:
                self._cb_status("thinking",
                    f"API unavailable -- pausing 10 min (attempt {_pause_depth + 1}/3)")
                self._update_heartbeat(exp_num, "paused_api_outage")
                for _ in range(120):  # 10 min in 5s chunks
                    if self._stop_event.is_set():
                        return
                    time.sleep(5)
                return self._run_experiment(exp_num, _pause_depth + 1)
            else:
                self._cb_error(f"Skipping exp{exp_num} -- API unavailable after 30 min of retries")
                return

        # Check for near-duplicate proposals and re-query if needed (max 2 retries)
        for _dup_attempt in range(2):
            if not self._is_near_duplicate(proposal.description):
                break
            self._cb_status("thinking",
                f"Duplicate detected: '{proposal.description[:60]}...' — re-querying LLM")
            nudge_dup = (
                f"\n\nIMPORTANT: Your proposal '{proposal.description}' is a near-duplicate "
                "of a previous experiment. Please propose a DIFFERENT modification "
                "that has not been tried before. Check the history carefully."
            )
            proposal = self._call_llm_with_backoff(
                current_code, results_history + nudge_dup, exp_num
            )
            if proposal is None:
                return
        else:
            # Still a duplicate after retries — proceed anyway rather than skip
            self._cb_status("thinking", "Duplicate persists after retries — proceeding")

        self._cb_experiment_start(exp_num, proposal.description, proposal.reasoning)
        self._update_heartbeat(exp_num, "committing")

        # 2. Apply code changes
        self._cb_status("committing", f"Applying: {proposal.description}")
        try:
            self._apply_hp_block(proposal.code)
        except Exception as e:
            self._cb_error(f"Failed to apply code: {e}")
            return

        # Validate the modified code parses
        try:
            with open(self._training_script) as f:
                compile(f.read(), self._training_script, "exec")
        except SyntaxError as e:
            self._cb_error(f"Syntax error in modified code: {e}")
            # Restore original
            self._apply_hp_block(current_code)
            return

        # 3. Commit
        try:
            commit_hash = self._git.commit_changes(
                f"exp{exp_num}: {proposal.description}",
                [self._training_script],
            )
        except Exception as e:
            self._cb_error(f"Git commit failed: {e}")
            self._apply_hp_block(current_code)
            return

        # 4. Train
        self._cb_status("training", f"Training exp{exp_num}: {proposal.description}")
        self._update_heartbeat(exp_num, "training")
        final = self._run_training()

        # 5. Evaluate and decide
        self._cb_status("evaluating", "Comparing results...")
        self._update_heartbeat(exp_num, "evaluating")

        if final and final.val_bpb > 0:
            improved = final.val_bpb < self.best_val_bpb

            if improved:
                status = "keep"
                old_best = self.best_val_bpb
                self.best_val_bpb = final.val_bpb
                self.best_experiment = f"exp{exp_num} ({proposal.description})"
                self.kept_count += 1
                self._cb_status("evaluating",
                    f"KEEP -- val_bpb improved: {final.val_bpb:.4f} (was {old_best:.4f})")
            else:
                status = "discard"
                self.discarded_count += 1
                self._cb_status("evaluating",
                    f"DISCARD -- val_bpb {final.val_bpb:.4f} >= best {self.best_val_bpb:.4f}")
                self._git.revert_last_experiment()

            result = ExperimentResult(
                exp=f"exp{exp_num}",
                description=proposal.description,
                val_bpb=final.val_bpb,
                peak_mem_gb=final.peak_vram_mb / 1024,
                tok_sec=int(final.total_tokens_M * 1e6 / final.training_seconds) if final.training_seconds > 0 else 0,
                mfu=final.mfu_percent,
                steps=final.num_steps,
                status=status,
                notes=proposal.reasoning[:80],
                gpu_name=self._hw_info.get("chip_name", "unknown"),
                baseline_sha=self._baseline_sha or "",
                watts=final.avg_watts,
                joules_per_token=final.joules_per_token,
                total_energy_joules=final.total_energy_j,
            )
        else:
            # Crash
            self.crash_count += 1
            self._cb_status("evaluating", "CRASH -- training failed")
            self._git.revert_last_experiment()

            result = ExperimentResult(
                exp=f"exp{exp_num}",
                description=proposal.description,
                val_bpb=0.0, peak_mem_gb=0.0, tok_sec=0,
                mfu=0.0, steps=0, status="crash",
                notes="training crashed or timed out",
                gpu_name=self._hw_info.get("chip_name", "unknown"),
                baseline_sha=self._baseline_sha or "",
            )

        append_result(self._results_path, result)
        self.total_runs += 1
        self._cb_experiment_complete(result)
        self._cb_stats_update()
        self._update_heartbeat(exp_num, "completed")

        # Sync results to GitHub every 10 minutes (built-in, no external loop needed)
        self._maybe_sync_results()

    # ------------------------------------------------------------------
    # Built-in results sync
    # ------------------------------------------------------------------

    def _maybe_sync_results(self, force: bool = False) -> None:
        """Push results to GitHub if 10+ minutes since last sync.

        This replaces the fragile background sync loop (which kept dying).
        Runs inline after each experiment — takes <5s, negligible overhead
        compared to the 5-min training cycle.
        """
        SYNC_INTERVAL = 600  # 10 minutes
        now = time.time()
        if not force and (now - self._last_sync_time) < SYNC_INTERVAL:
            return

        try:
            # Stage results and heartbeat
            results_files = [self._results_path]
            heartbeat = os.path.join(
                os.path.dirname(os.path.abspath(self._results_path)),
                ".runner_status.json"
            )
            if os.path.exists(heartbeat):
                results_files.append(heartbeat)

            for f in results_files:
                try:
                    self._git._run("add", f, check=False)
                except Exception:
                    pass

            # Only commit if there are staged changes
            diff = self._git._run("diff", "--cached", "--quiet", check=False)
            if diff is not None:  # non-zero exit = there are changes
                try:
                    summary = (
                        f"exp{self.total_runs} | kept={self.kept_count} "
                        f"disc={self.discarded_count} crash={self.crash_count} | "
                        f"best={self.best_val_bpb:.4f}"
                    )
                    self._git._run(
                        "commit", "-m",
                        f"Sync results ({time.strftime('%H:%M', time.gmtime())} UTC) — {summary}",
                        check=False
                    )
                except Exception:
                    pass

            # Push to remote. Use `push -u` to set upstream tracking on
            # the first push of a new branch — `push.autoSetupRemote=true`
            # is unreliable across pod restarts and git checkouts.
            try:
                branch = self._git._run("rev-parse", "--abbrev-ref", "HEAD", check=False)
                has_upstream = bool(self._git._run(
                    "rev-parse", "--abbrev-ref", f"{branch}@{{u}}", check=False
                ))
                if has_upstream:
                    self._git._run("push", check=False)
                else:
                    # First push — set upstream explicitly
                    self._git._run("push", "-u", "origin", branch, check=False)
            except Exception:
                pass

            self._last_sync_time = now
        except Exception:
            pass  # sync is best-effort, never crash the experiment loop

    # ------------------------------------------------------------------
    # LLM call with exponential backoff
    # ------------------------------------------------------------------

    def _call_llm_with_backoff(self, current_code, results_history, exp_num):
        """Call the LLM with exponential backoff for transient errors.

        Handles:
        - Rate limits (429): backoff 60s, 120s, 240s, 480s, 600s
        - Server errors (5xx): backoff 30s, 60s, 120s, 240s, 300s
        - Connection errors: backoff 30s, 60s, 120s, 240s, 300s
        - Other errors: backoff 10s, 20s, 30s, 40s, 50s

        Returns the proposal or None if all retries exhausted.
        """
        max_retries = 5

        for attempt in range(max_retries):
            try:
                proposal = self._llm.generate_experiment(
                    current_code=current_code,
                    results_history=results_history,
                    best_val_bpb=self.best_val_bpb,
                    best_experiment=self.best_experiment,
                    hw_info=self._hw_info,
                )
                return proposal

            except Exception as e:
                err_str = str(e).lower()
                err_type = type(e).__name__

                # Fatal errors that will never recover -- stop immediately
                # Covers Anthropic, OpenAI, and Azure OpenAI error patterns
                if "credit balance" in err_str or "insufficient_quota" in err_str:
                    self._cb_error(f"FATAL: API billing error -- stopping agent. {e}")
                    self._stop_event.set()
                    return None
                if "invalid_api_key" in err_str or ("authentication" in err_str and "401" in err_str):
                    self._cb_error(f"FATAL: API authentication failed -- stopping agent. {e}")
                    self._stop_event.set()
                    return None
                if "deploymentnotfound" in err_str or ("deployment" in err_str and "not found" in err_str):
                    self._cb_error(f"FATAL: Azure deployment not found -- stopping agent. {e}")
                    self._stop_event.set()
                    return None
                if "model_not_found" in err_str or "insufficient_credits" in err_str or "payment_required" in err_str:
                    self._cb_error(f"FATAL: OpenRouter error -- stopping agent. {e}")
                    self._stop_event.set()
                    return None

                # Classify the error and determine backoff
                if "rate" in err_str or "429" in err_str:
                    wait = min(60 * (2 ** attempt), 600)
                    self._cb_status("thinking",
                        f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                elif "500" in err_str or "502" in err_str or "503" in err_str or "server" in err_str:
                    wait = min(30 * (2 ** attempt), 300)
                    self._cb_status("thinking",
                        f"API server error, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                elif "connection" in err_str or "timeout" in err_str:
                    wait = min(30 * (2 ** attempt), 300)
                    self._cb_status("thinking",
                        f"Connection error, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                else:
                    wait = 10 * (attempt + 1)
                    self._cb_error(
                        f"LLM error ({err_type}, attempt {attempt+1}/{max_retries}): {e}")

                self._update_heartbeat(exp_num, f"retrying_api_{attempt+1}")

                # Sleep in 5s chunks so stop_event can interrupt
                for _ in range(max(1, wait // 5)):
                    if self._stop_event.is_set():
                        return None
                    time.sleep(5)

        return None

    # ------------------------------------------------------------------
    # Training subprocess
    # ------------------------------------------------------------------

    def _run_training(self) -> Optional[FinalMetrics]:
        """Run the training script and return parsed final metrics.

        Returns None if the training crashed or timed out.
        """
        parser = OutputParser()

        python = sys.executable
        cmd = [python, "-u", self._training_script]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["AUTORESEARCH_ORCHESTRATOR"] = "1"  # suppress standalone results writing
        # Ensure repo root is importable (training scripts live under platforms/)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        try:
            with self._lock:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    bufsize=0,
                )
        except Exception as e:
            self._cb_error(f"Failed to start training: {e}")
            return None

        # Read output byte-by-byte (same pattern as app.py)
        buffer = ""
        deadline = time.time() + 600  # 10 minute timeout

        try:
            while True:
                if self._stop_event.is_set():
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                    return None

                if time.time() > deadline:
                    self._cb_error("Training timed out (>10 min) -- killing")
                    self._proc.kill()
                    self._proc.wait()
                    return None

                # Use select() with 1s timeout so the deadline check above
                # actually fires when the process hangs (e.g. stuck in eval
                # compilation). Without this, read(1) blocks indefinitely and
                # the timeout never triggers.
                ready, _, _ = select.select([self._proc.stdout], [], [], 1.0)
                if not ready:
                    continue  # No data -- loop back to check deadline/stop
                byte = self._proc.stdout.read(1)
                if not byte:
                    break

                char = byte.decode("utf-8", errors="replace")

                if char in ("\n", "\r"):
                    if buffer.strip():
                        results = parser.parse_line(buffer)
                        for item in results:
                            if isinstance(item, StepMetrics):
                                self._cb_training_output(buffer)
                            elif isinstance(item, str):
                                self._cb_training_output(item)
                    buffer = ""
                else:
                    buffer += char

        except Exception as e:
            self._cb_error(f"Error reading training output: {e}")
        finally:
            if buffer.strip():
                parser.parse_line(buffer)

            with self._lock:
                if self._proc:
                    self._proc.wait()
                    self._proc = None

        return parser.final

    # ------------------------------------------------------------------
    # Code manipulation
    # ------------------------------------------------------------------

    def _extract_hp_block(self) -> str:
        """Extract the hyperparameter block from the training script."""
        with open(self._training_script) as f:
            content = f.read()

        start_idx = content.find(HP_BLOCK_START)
        end_idx = content.find(HP_BLOCK_END)

        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Could not find hyperparameter block markers in training script")

        return content[start_idx:end_idx]

    def _apply_hp_block(self, new_block: str) -> None:
        """Replace the hyperparameter block in the training script."""
        with open(self._training_script) as f:
            content = f.read()

        start_idx = content.find(HP_BLOCK_START)
        end_idx = content.find(HP_BLOCK_END)

        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Could not find hyperparameter block markers in training script")

        # Ensure the new block ends with a newline
        if not new_block.endswith("\n"):
            new_block += "\n"

        # Ensure there's a blank line before the Setup marker
        if not new_block.endswith("\n\n"):
            new_block += "\n"

        new_content = content[:start_idx] + new_block + content[end_idx:]

        with open(self._training_script, "w") as f:
            f.write(new_content)
