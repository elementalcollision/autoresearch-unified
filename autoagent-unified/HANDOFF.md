# autoagent-unified: Cloud-to-Local Handoff

## Quick Start

```bash
git clone https://github.com/elementalcollision/autoresearch-unified.git
cd autoresearch-unified
git checkout claude/plan-integration-fork-mV1rj
cd autoagent-unified

# CPU-only (tests + development)
pip install -e "."

# GPU development (pick one)
pip install -e ".[all-cuda]"     # NVIDIA
pip install -e ".[all-rocm]"     # AMD
pip install -e ".[all-metal]"    # Apple Silicon

# Verify
python -m pytest tests/ -v       # expect 73+ pass, 4 skip if no openai pkg
```

---

## What This Is

**autoagent-unified** merges two systems:

1. **autoagent** (kevinrgu/autoagent) — meta-agent that hill-climbs on an `agent.py` harness, benchmarked via Harbor tasks
2. **autoresearch-unified** — autonomous LLM-driven GPU pretraining research (propose hyperparams -> train 5 min -> evaluate val_bpb -> keep/discard)

Plus a new **drift monitor** (performance plateau, hardware anomaly, strategy repetition detection).

**Two-level optimization:**
- **Outer loop** (autoagent): meta-agent iterates on `agent.py` to improve benchmark scores
- **Inner loop** (autoresearch): agent proposes hyperparameters -> trains -> evaluates -> keeps/discards

---

## Repository Layout

```
autoresearch-unified/              # parent repo
└── autoagent-unified/             # integration subdirectory
    ├── agent.py                   # THE agent harness (meta-agent edits this)
    ├── program.md                 # meta-agent directives
    ├── Dockerfile.base            # container image
    ├── pyproject.toml             # combined dependencies
    │
    ├── autoresearch/              # ported infrastructure
    │   ├── resilience.py          # atomic_write, atomic_append, Heartbeat, signals
    │   ├── results.py             # ExperimentResult, 14-col TSV, classify_experiment
    │   ├── parser.py              # OutputParser for training stdout
    │   ├── llm_backend.py         # Claude/OpenAI/Azure/OpenRouter/Ollama backends
    │   ├── credentials.py         # API key resolution (env > keychain > file)
    │   ├── git_manager.py         # experiment commit/revert tracking
    │   ├── hardware.py            # get_hardware_summary() — all platforms
    │   └── power.py               # PowerMonitor — background watt sampling
    │
    ├── backends/                  # hardware registry
    │   ├── __init__.py            # detect_backend(), get_hardware_info(), get_peak_flops()
    │   └── registry.py            # FLOPS tables (H100, A100, MI300X, M4 Max, etc.)
    │
    ├── drift/                     # three-axis drift detection
    │   ├── monitor.py             # DriftMonitor orchestrator + DriftReport
    │   ├── performance.py         # plateau (linear regression), regression, throughput
    │   ├── hardware.py            # memory pressure, thermal throttle, power efficiency
    │   └── strategy.py            # repetition, Shannon entropy diversity, Jaccard similarity
    │
    ├── tools/                     # agent tool wrappers
    │   ├── train_tool.py          # run_training_experiment() — 5-min GPU run
    │   ├── analyze_tool.py        # analyze_experiment_history() — query results.tsv
    │   ├── drift_tool.py          # check_drift_status() — surface alerts to agent
    │   └── hardware_tool.py       # get_hardware_profile() — JSON hardware info
    │
    └── tests/                     # 8 test files, 77 tests total
        ├── test_resilience.py     #  7 — atomic writes, TSV validation, heartbeat
        ├── test_results.py        # 11 — TSV lifecycle, classification, formatting
        ├── test_parser.py         #  4 — training output parsing
        ├── test_drift_performance.py  #  5 — plateau, regression, throughput, keep ratio
        ├── test_drift_hardware.py     #  4 — memory, thermal, power efficiency
        ├── test_drift_strategy.py     #  5 — repetition, diversity, similarity
        ├── test_drift_integration.py  #  6 — combined monitor, prompt injection
        └── test_l2_integration.py     # 28 — cross-module integration (L2)
```

---

## Git State

| Item | Value |
|------|-------|
| Remote | `elementalcollision/autoresearch-unified` |
| Branch | `claude/plan-integration-fork-mV1rj` |
| Latest commit | `aae11d1` — Add L2 component integration tests |
| Base | `5b6a90f` (main) — diverges by 3 commits |
| Working tree | Clean |

There is also a standalone fork at `elementalcollision/autoagent-unified` (cloned to `/home/user/autoagent-unified/`), but its commit signing was failing — the **canonical location** for the integration is the subdirectory in `autoresearch-unified` on the branch above.

---

## Test Verification Status

| Level | Scope | Result | Notes |
|-------|-------|--------|-------|
| L0.1 | 11 module imports | PASS | All modules import without GPU |
| L0.3 | License checks | PASS | MIT in LICENSE, NOTICE credits upstream |
| L1.1 | 49 unit tests | 49/49 PASS | < 0.1s |
| L2 | 28 integration tests | 24 PASS, 4 SKIP | Skips: `openai` pkg not installed |
| L3 | GPU smoke tests | NOT RUN | Requires GPU hardware |
| L4-L6 | Full integration | NOT RUN | Requires GPU + Harbor |

---

## What's Ready vs. What Needs Work

### Ready (fully tested, working)

- All `autoresearch/` infrastructure modules — crash safety, results tracking, parsing, hardware detection, power monitoring, git management
- All `drift/` detection — performance plateau, hardware anomaly, strategy repetition
- All `tools/` wrappers — train, analyze, drift, hardware
- Multi-provider LLM backend factory (Claude, OpenAI, Azure, OpenRouter, Ollama)
- LLM response parser (DESCRIPTION/REASONING/CODE format)
- `agent.py` with 5 registered tools and OpenAI Agents SDK + fallback architecture

### Needs GPU to validate (L3)

```bash
# L3.1 — Hardware detection
python -c "from backends import detect_backend, get_hardware_info; print(get_hardware_info())"

# L3.2 — Power monitor
python -c "from autoresearch.power import PowerMonitor; pm = PowerMonitor('cuda'); pm.start(); import time; time.sleep(3); print(pm.stop(3.0))"

# L3.3 — Single training experiment (needs training script)
python -c "from tools.train_tool import run_training_experiment; print(run_training_experiment('test', '# defaults'))"

# L3.4 — Hardware tool
python -c "from tools.hardware_tool import get_hardware_profile; print(get_hardware_profile())"
```

### Needs targeted development

1. **Training script integration** — `tools/train_tool.py` calls `python <training_script>` but the actual training scripts live in `autoresearch-unified/platforms/`. Need to resolve paths or copy/symlink scripts.

2. **`agent.py` tool registration** — Currently uses `openai-agents` SDK `@function_tool`. If pivoting to Claude Agent SDK, these need to be adapted (see `agent-claude.py` for the pattern).

3. **Orchestrator** — `autoresearch/orchestrator.py` was planned but not yet ported. The experiment loop currently lives in `agent.py`'s `run_task()` function. A standalone orchestrator would enable running experiments without the Harbor adapter.

4. **LLM backend tests with live API keys** — L2.4 factory tests pass for Ollama and error cases; OpenAI/Azure/OpenRouter tests skip without the package. Need `pip install openai` and real keys for L4.3.

5. **Harbor integration** — The Harbor adapter in `agent.py` assumes a Docker environment with `/task/instruction.md`. Testing the full outer loop requires Harbor infrastructure.

---

## Environment Variables

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `ANTHROPIC_API_KEY` | Claude API access | One of these |
| `OPENAI_API_KEY` | OpenAI API access | required for |
| `OPENROUTER_API_KEY` | OpenRouter access | LLM backend |
| `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | Azure OpenAI | |
| `OLLAMA_MODEL` | Local LLM model name | |
| `RESULTS_TSV` | Override results.tsv path | No (default: `results.tsv`) |
| `AUTORESEARCH_BACKEND` | Force backend (cuda/rocm/mlx/hpu) | No (auto-detect) |

---

## Key Architecture Decisions

1. **Subdirectory, not submodule** — autoagent-unified lives as a directory inside autoresearch-unified, not a git submodule. This keeps everything in one branch/checkout.

2. **Graceful degradation everywhere** — GPU code is behind try/except. PowerMonitor returns (0, 0) with no GPU. Hardware detection falls back to "Unknown". Tests run on CPU-only machines.

3. **LLM backend factory** — Priority: Ollama > OpenRouter > Azure > OpenAI > Claude. Set the env var for the provider you want. The factory auto-selects.

4. **14-column TSV** — The results format includes power metrics (watts, joules_per_token, total_energy_joules) from the start, even if they're zero on non-GPU runs.

5. **Drift monitor is advisory** — `inject_into_prompt()` appends alerts to the LLM system prompt so the agent can self-correct. It doesn't block or force behavior changes.

---

## Licensing

- **Combined code**: MIT (see `LICENSE`)
- **Upstream autoagent** (kevinrgu/autoagent): No explicit license yet — see action item in `NOTICE`
- **autoresearch-unified components**: MIT
- **Data artifacts**: CC-BY-4.0

Contact `kevinrgu` to clarify autoagent licensing before any public distribution.
