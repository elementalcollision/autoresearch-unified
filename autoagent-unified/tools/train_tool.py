"""Training experiment tool for autoagent's agent harness.

Wraps autoresearch's experiment execution: runs a 5-minute GPU training
experiment with specified hyperparameters and returns structured results.
"""

import json
import os
import subprocess
import time

from autoresearch.parser import OutputParser, FinalMetrics
from autoresearch.results import (
    ExperimentResult, append_result, get_best_result,
    init_results_tsv, next_experiment_number,
)
from autoresearch.power import PowerMonitor


RESULTS_PATH = os.environ.get("RESULTS_TSV", "results.tsv")


def run_training_experiment(
    description: str,
    hyperparameters: str,
    training_script: str = "platforms/cuda/train_cuda.py",
    timeout_seconds: int = 330,
) -> str:
    """Run a 5-min GPU training experiment with given hyperparameters.

    Args:
        description: One-line description of the change.
        hyperparameters: The replacement hyperparameter block code.
        training_script: Path to the training script.
        timeout_seconds: Maximum runtime (default: 5.5 min with buffer).

    Returns:
        JSON string with val_bpb, tok/sec, MFU, peak memory, power metrics,
        and keep/discard decision.
    """
    init_results_tsv(RESULTS_PATH)
    exp_num = next_experiment_number(RESULTS_PATH)
    exp_name = f"exp{exp_num}"

    # Detect backend for power monitoring
    backend = os.environ.get("AUTORESEARCH_BACKEND", "cuda")
    power = PowerMonitor(backend=backend)

    try:
        # Start training subprocess
        power.start()
        start_time = time.time()

        proc = subprocess.run(
            ["python", training_script],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        training_seconds = time.time() - start_time
        avg_watts, total_joules = power.stop(training_seconds)

        # Parse output
        parser = OutputParser()
        for line in proc.stdout.splitlines():
            parser.parse_line(line)

        if parser.final is None:
            # Training crashed or didn't complete
            result = ExperimentResult(
                exp=exp_name, description=description,
                val_bpb=0.0, peak_mem_gb=0.0, tok_sec=0,
                mfu=0.0, steps=0, status="crash",
                notes=proc.stderr[-200:] if proc.stderr else "No output",
                watts=avg_watts,
                joules_per_token=0.0,
                total_energy_joules=total_joules,
            )
            append_result(RESULTS_PATH, result)
            return json.dumps({"status": "crash", "error": result.notes})

        final = parser.final
        tok_sec = int(final.total_tokens_M * 1e6 / final.training_seconds) if final.training_seconds > 0 else 0
        joules_per_tok = total_joules / (final.total_tokens_M * 1e6) if final.total_tokens_M > 0 else 0.0

        # Decide keep/discard
        best_bpb, best_exp = get_best_result(RESULTS_PATH)
        status = "keep" if final.val_bpb < best_bpb else "discard"

        result = ExperimentResult(
            exp=exp_name, description=description,
            val_bpb=final.val_bpb,
            peak_mem_gb=final.peak_vram_mb / 1024,
            tok_sec=tok_sec,
            mfu=final.mfu_percent,
            steps=final.num_steps,
            status=status,
            notes=f"depth={final.depth} params={final.num_params_M:.0f}M",
            watts=avg_watts,
            joules_per_token=joules_per_tok,
            total_energy_joules=total_joules,
        )
        append_result(RESULTS_PATH, result)

        return json.dumps({
            "status": status,
            "experiment": exp_name,
            "val_bpb": final.val_bpb,
            "tok_sec": tok_sec,
            "mfu": final.mfu_percent,
            "peak_mem_gb": final.peak_vram_mb / 1024,
            "steps": final.num_steps,
            "watts": avg_watts,
            "joules_per_token": joules_per_tok,
            "best_val_bpb": min(best_bpb, final.val_bpb),
        })

    except subprocess.TimeoutExpired:
        avg_watts, total_joules = power.stop(timeout_seconds)
        result = ExperimentResult(
            exp=exp_name, description=description,
            val_bpb=0.0, peak_mem_gb=0.0, tok_sec=0,
            mfu=0.0, steps=0, status="crash",
            notes="Timeout exceeded",
            watts=avg_watts,
            joules_per_token=0.0,
            total_energy_joules=total_joules,
        )
        append_result(RESULTS_PATH, result)
        return json.dumps({"status": "crash", "error": "Timeout exceeded"})

    except Exception as e:
        power.stop(0)
        return json.dumps({"status": "error", "error": str(e)})
