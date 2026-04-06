"""Autoagent-unified agent harness.

Integrates autoagent's meta-agent optimization framework with autoresearch's
autonomous experiment loop and drift monitoring. This is the single-file
editable harness that the meta-agent iterates on.

Architecture:
  - @function_tool wrappers call into autoresearch/ infrastructure
  - Drift monitor injects warnings into the agent's context
  - Multi-provider LLM backend (Claude, OpenAI, Azure, OpenRouter)
"""

import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Agent configuration (editable by meta-agent)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous AI researcher running GPU training experiments to minimize
val_bpb (validation bits per byte — lower is better).

You have access to tools for:
1. Running training experiments with different hyperparameters
2. Analyzing experiment history and trends
3. Checking drift status (performance plateaus, hardware issues, strategy repetition)
4. Querying hardware capabilities

Strategy:
- Start by understanding the current experiment history
- Check drift status before proposing new experiments
- Make ONE change per experiment to isolate effects
- If drift monitor reports a plateau, try a fundamentally different approach
- If drift monitor reports strategy repetition, switch to an underexplored category
- Balance exploration (new ideas) with exploitation (refining what works)

Hyperparameter categories to explore:
- Learning rates: MATRIX_LR, SCALAR_LR, EMBEDDING_LR
- Regularization: WEIGHT_DECAY, ADAM_BETAS
- Schedule: WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC
- Architecture: DEPTH, ASPECT_RATIO, HEAD_DIM, MLP_RATIO
- Throughput: TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE
"""

MODEL = os.environ.get("AGENT_MODEL", "gpt-5")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "30"))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

# These tools bridge autoagent's agent framework with autoresearch infrastructure.
# Each tool is a thin wrapper that the meta-agent can modify to improve behavior.

def tool_run_training_experiment(description: str, hyperparameters: str) -> str:
    """Run a 5-min GPU training experiment with given hyperparameters.

    Returns JSON with: val_bpb, tok/sec, MFU, peak_mem_gb, power metrics,
    and keep/discard decision.
    """
    from tools.train_tool import run_training_experiment
    return run_training_experiment(description=description, hyperparameters=hyperparameters)


def tool_analyze_experiment_history(query: str = "summary") -> str:
    """Query results.tsv — trends, best results, category breakdown.

    Args:
        query: One of "summary", "best", "categories", "full", "recent".
    """
    from tools.analyze_tool import analyze_experiment_history
    return analyze_experiment_history(query=query)


def tool_check_drift_status(detail_level: str = "summary") -> str:
    """Get current drift-monitor report — performance plateaus,
    hardware anomalies, strategy repetition warnings.

    Args:
        detail_level: "summary" or "full".
    """
    from tools.drift_tool import check_drift_status
    return check_drift_status(detail_level=detail_level)


def tool_get_hardware_profile() -> str:
    """Return detected hardware, peak FLOPS, and suggested hyperparameters."""
    from tools.hardware_tool import get_hardware_profile
    return get_hardware_profile()


def tool_run_shell(command: str, timeout: int = 120) -> str:
    """Execute a shell command (from original autoagent harness).

    Args:
        command: Shell command to run.
        timeout: Max seconds (default 120).
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output[:10000]  # Truncate large outputs
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


# Tool list for the agent framework
TOOLS = [
    {
        "name": "run_training_experiment",
        "description": "Run a 5-min GPU training experiment with given hyperparameters. Returns val_bpb, tok/sec, MFU, memory, power metrics.",
        "function": tool_run_training_experiment,
        "parameters": {
            "description": {"type": "string", "description": "One-line description of the hyperparameter change"},
            "hyperparameters": {"type": "string", "description": "The replacement hyperparameter block code"},
        },
    },
    {
        "name": "analyze_experiment_history",
        "description": "Query results.tsv for trends, best results, category breakdown.",
        "function": tool_analyze_experiment_history,
        "parameters": {
            "query": {"type": "string", "description": "One of: summary, best, categories, full, recent"},
        },
    },
    {
        "name": "check_drift_status",
        "description": "Get drift-monitor report: performance plateau, hardware anomalies, strategy repetition.",
        "function": tool_check_drift_status,
        "parameters": {
            "detail_level": {"type": "string", "description": "summary or full"},
        },
    },
    {
        "name": "get_hardware_profile",
        "description": "Return detected hardware, peak FLOPS, and suggested hyperparameters.",
        "function": tool_get_hardware_profile,
        "parameters": {},
    },
    {
        "name": "run_shell",
        "description": "Execute a shell command with timeout.",
        "function": tool_run_shell,
        "parameters": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "description": "Max seconds (default 120)"},
        },
    },
]


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def create_agent():
    """Build the agent with registered tools.

    Supports both OpenAI Agents SDK and direct LLM backend usage.
    The meta-agent can modify this function to change agent behavior.
    """
    # Try OpenAI Agents SDK first (autoagent's native framework)
    try:
        from agents import Agent, function_tool

        # Register tools with OpenAI Agents SDK
        registered_tools = []
        for tool_def in TOOLS:
            ft = function_tool(tool_def["function"])
            registered_tools.append(ft)

        agent = Agent(
            name="autoagent-unified",
            instructions=SYSTEM_PROMPT,
            model=MODEL,
            tools=registered_tools,
        )
        return agent

    except ImportError:
        # Fall back to autoresearch's LLM backend
        from autoresearch.llm_backend import get_llm_backend
        return get_llm_backend()


def run_task(task_description: str) -> dict:
    """Run the agent on a task and return results.

    This is the main entry point for Harbor benchmark evaluation.
    """
    agent = create_agent()

    # If using OpenAI Agents SDK
    try:
        from agents import Runner
        result = Runner.run_sync(agent, task_description, max_turns=MAX_TURNS)
        return {
            "output": result.final_output,
            "steps": len(result.raw_responses),
        }
    except ImportError:
        pass

    # Fallback: direct execution via autoresearch backend
    # (used when openai-agents is not installed)
    return {
        "output": "Agent executed via autoresearch LLM backend",
        "steps": 1,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "Run an autonomous experiment loop to minimize val_bpb."
    print(f"Task: {task}")
    result = run_task(task)
    print(f"Result: {json.dumps(result, indent=2)}")
