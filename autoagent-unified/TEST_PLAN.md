# autoagent-unified — Verification & Test Plan

## Environments

| Environment | Purpose | Hardware |
|-------------|---------|----------|
| **Local (no GPU)** | Unit tests, import checks, mock integration | CPU-only, any OS |
| **Local (GPU)** | Smoke tests, single-experiment validation | NVIDIA/AMD/Apple Silicon |
| **RunPod (GPU)** | Full end-to-end, overnight runs, multi-experiment | H100/A100/MI300X |

---

## Level 0: Static Checks (Local, no GPU, < 1 min)

### L0.1 — Import validation
Verify all modules import without GPU dependencies.

```bash
cd autoagent-unified
python -c "from autoresearch.resilience import atomic_write, Heartbeat"
python -c "from autoresearch.results import ExperimentResult, load_results, classify_experiment"
python -c "from autoresearch.parser import OutputParser, StepMetrics, FinalMetrics"
python -c "from autoresearch.credentials import resolve_api_key"  # will raise if no key, that's expected
python -c "from autoresearch.git_manager import GitManager"
python -c "from drift.monitor import DriftMonitor, DriftConfig, DriftReport"
python -c "from drift.performance import PerformanceDrift"
python -c "from drift.hardware import HardwareDrift"
python -c "from drift.strategy import StrategyDrift"
python -c "from tools.analyze_tool import analyze_experiment_history"
python -c "from tools.drift_tool import check_drift_status"
python -c "from tools.hardware_tool import get_hardware_profile"
```

**Pass criteria:** All imports succeed (or fail only on missing API keys, never on missing modules).

### L0.2 — Dependency consistency
```bash
pip install -e "." && python -c "import autoagent_unified"  # basic install
pip install -e ".[anthropic]" && python -c "import anthropic"
pip install -e ".[openai]" && python -c "import openai"
```

**Pass criteria:** Each optional dependency group installs cleanly without conflicts.

### L0.3 — License & attribution
```bash
test -f LICENSE && grep -q "MIT" LICENSE
test -f NOTICE && grep -q "kevinrgu/autoagent" NOTICE
grep -q 'license.*MIT' pyproject.toml
```

**Pass criteria:** LICENSE exists (MIT), NOTICE credits both upstream projects.

---

## Level 1: Unit Tests (Local, no GPU, < 30 sec)

### L1.1 — Run existing test suite
```bash
cd autoagent-unified
python -m pytest tests/ -v --tb=short
```

**Pass criteria:** 49/49 tests pass. This covers:

| Test file | Tests | What it verifies |
|-----------|-------|------------------|
| test_resilience.py | 7 | Atomic writes, TSV validation, heartbeat |
| test_results.py | 11 | TSV init/append/load, classification, formatting |
| test_parser.py | 4 | Step parsing, final block, CR segments, startup capture |
| test_drift_performance.py | 5 | Plateau, regression, throughput drop, keep ratio |
| test_drift_hardware.py | 4 | Memory pressure, thermal throttling, power efficiency |
| test_drift_strategy.py | 5 | Category repetition, diversity entropy, near-duplicates |
| test_drift_integration.py | 6 | Combined monitor, prompt injection, report formatting |

### L1.2 — Drift monitor with synthetic scenarios
Create a synthetic results.tsv with known drift patterns and verify each detector fires correctly.

**Scenario A — Performance plateau:**
```python
# 30 kept experiments with val_bpb hovering at 1.300 +/- 0.0002
# Expected: plateau alert (WARNING)
```

**Scenario B — Strategy repetition:**
```python
# 8 consecutive learning_rate experiments, all discarded
# Expected: repetition alert (CRITICAL) + diversity alert (WARNING)
```

**Scenario C — Thermal throttling:**
```python
# Normal: 300W / 10k tok/sec, then sudden: 200W / 6k tok/sec
# Expected: thermal alert (CRITICAL)
```

**Scenario D — Clean run (no drift):**
```python
# 20 experiments with steady improvement across diverse categories
# Expected: no alerts
```

**Pass criteria:** Each scenario triggers exactly the expected alert types and levels.

---

## Level 2: Component Integration (Local, no GPU, < 2 min)

### L2.1 — Results round-trip
```python
# 1. init_results_tsv() -> creates file with 14-column header
# 2. append_result() x 10 -> adds 10 rows
# 3. load_results() -> returns 10 ExperimentResult objects
# 4. format_history_for_prompt() -> produces formatted table
# 5. classify_experiment() on each -> correct categories
# 6. get_best_result() -> correct minimum val_bpb
# 7. Kill and restart mid-append -> validate_results_tsv() recovers
```

**Pass criteria:** Full lifecycle works, crash recovery detects and fixes truncated lines.

### L2.2 — Drift monitor end-to-end
```python
# 1. Write synthetic results.tsv with mixed scenarios
# 2. DriftMonitor.check_from_file("results.tsv")
# 3. Verify report.has_alerts, report.critical_count, report.warning_count
# 4. inject_into_prompt() adds "DRIFT MONITOR ALERTS" section
# 5. format_summary() produces one-line status
```

**Pass criteria:** Monitor correctly aggregates alerts from all three detectors.

### L2.3 — Tool wrappers (mocked)
```python
# analyze_tool: write a results.tsv, call each query type
#   - "summary" -> JSON with total/kept/discarded
#   - "best" -> best val_bpb string
#   - "categories" -> category breakdown
#   - "recent" -> last 10 experiments
#   - "full" -> formatted table with strategy footer

# drift_tool: write results.tsv with known drift, verify output
#   - "summary" -> "Drift alerts: 1 critical, 2 warnings"
#   - "full" -> multi-line report with [!]/[~]/[i] icons

# hardware_tool: call get_hardware_profile()
#   - Returns valid JSON with chip_name, gpu_cores, memory_gb, peak_tflops
```

**Pass criteria:** All tool wrappers return correctly formatted output.

### L2.4 — LLM backend factory
```python
# Test provider selection via environment variables:
# 1. Set ANTHROPIC_API_KEY only -> ClaudeBackend
# 2. Set OPENAI_API_KEY only -> OpenAIBackend
# 3. Set OPENROUTER_API_KEY only -> OpenRouterBackend
# 4. Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT -> AzureOpenAIBackend
# 5. Set OLLAMA_MODEL only -> OllamaBackend
# 6. Set nothing -> RuntimeError with setup instructions
# 7. Set multiple -> verify priority order (Ollama > OpenRouter > Azure > OpenAI > Claude)
```

**Pass criteria:** Factory returns correct backend type for each env var combination. (Use mock env vars, no actual API calls.)

### L2.5 — LLM response parsing
```python
# Test parse_llm_response() with:
# 1. Well-formed response -> ExperimentProposal with description, reasoning, code
# 2. Response with markdown fences -> strips ```python and ``` correctly
# 3. Missing DESCRIPTION -> ValueError
# 4. Missing CODE -> ValueError
# 5. Extra whitespace / trailing newlines -> clean parsing
```

**Pass criteria:** Parser handles all formats correctly.

---

## Level 3: GPU Smoke Tests (Local GPU or RunPod, < 10 min)

### L3.1 — Hardware detection
```bash
python -c "
from backends import detect_backend, get_hardware_info, get_peak_flops
backend = detect_backend()
hw = get_hardware_info()
flops = get_peak_flops(hw)
print(f'Backend: {backend}')
print(f'Chip: {hw[\"chip_name\"]}')
print(f'Memory: {hw[\"memory_gb\"]:.1f} GB')
print(f'Peak FLOPS: {flops:.2e}')
assert backend in ('cuda', 'rocm', 'rocm7', 'hpu', 'mlx', 'mps')
assert hw['memory_gb'] > 0
assert flops > 0
"
```

**Pass criteria:** Detects correct GPU, reasonable memory and FLOPS values.

### L3.2 — Power monitor probe
```bash
python -c "
from autoresearch.power import PowerMonitor
from backends import detect_backend
import time

pm = PowerMonitor(backend=detect_backend())
pm.start()
time.sleep(3)
avg_watts, total_joules = pm.stop(3.0)
print(f'Power: {avg_watts:.1f}W, Energy: {total_joules:.1f}J')
# May be 0 if no power API — that's acceptable (graceful degradation)
print('PASS' if avg_watts >= 0 else 'FAIL')
"
```

**Pass criteria:** Returns non-negative values. Zero is acceptable (graceful degradation).

### L3.3 — Single training experiment (requires training script)
```bash
# This test requires a training script at the expected path.
# On RunPod, clone autoresearch-unified and use its training script.
python -c "
from tools.train_tool import run_training_experiment
import json, os

os.environ['RESULTS_TSV'] = '/tmp/test_results.tsv'
result = json.loads(run_training_experiment(
    description='smoke test - baseline defaults',
    hyperparameters='# Use defaults',
    training_script='platforms/cuda/train_cuda.py',
    timeout_seconds=330,
))
print(json.dumps(result, indent=2))
assert result['status'] in ('keep', 'discard', 'crash')
if result['status'] != 'crash':
    assert result['val_bpb'] > 0
    assert result['tok_sec'] > 0
"
```

**Pass criteria:** Training completes (or crashes cleanly with error message). No hangs.

### L3.4 — Hardware tool integration
```bash
python -c "
from tools.hardware_tool import get_hardware_profile
import json
profile = json.loads(get_hardware_profile())
print(json.dumps(profile, indent=2))
assert profile['chip_name'] != 'Unknown'
assert profile['memory_gb'] > 0
"
```

**Pass criteria:** Returns correct hardware info for the detected GPU.

---

## Level 4: Integration Tests (RunPod GPU, 30-60 min)

### L4.1 — Three-experiment loop
Run the agent with MAX_TURNS=3 to verify the full loop works.

```bash
export AGENT_MAX_TURNS=3
export RESULTS_TSV=results/test_integration.tsv
python agent.py "Run 3 experiments to minimize val_bpb"
```

**Verification checklist:**
- [ ] results.tsv created with header + 3+ rows (baseline + experiments)
- [ ] Each row has 14 tab-separated fields
- [ ] val_bpb values are reasonable (0.9-2.5 for GPT-2 scale)
- [ ] At least one experiment has status "keep" or "discard" (not all crashes)
- [ ] tok/sec > 0 for non-crash experiments
- [ ] .runner_status.json updated (if heartbeat is active)

**Pass criteria:** Loop completes without hanging. Results TSV is valid.

### L4.2 — Drift monitor live integration
After L4.1, verify drift tools work against real results.

```bash
python -c "
from tools.drift_tool import check_drift_status
from tools.analyze_tool import analyze_experiment_history
import os
os.environ['RESULTS_TSV'] = 'results/test_integration.tsv'

print('=== Drift Status ===')
print(check_drift_status('full'))
print()
print('=== Analysis Summary ===')
print(analyze_experiment_history('summary'))
print()
print('=== Recent Experiments ===')
print(analyze_experiment_history('recent'))
"
```

**Pass criteria:** Both tools return coherent output matching the actual results.

### L4.3 — Multi-provider LLM validation
Test that both Claude and OpenAI can generate experiment proposals.

```bash
# Test Claude
ANTHROPIC_API_KEY=sk-ant-... python -c "
from autoresearch.llm_backend import get_llm_backend
backend = get_llm_backend()
print(f'Backend: {backend.name()}')
assert backend.validate()
print('Claude backend validated')
"

# Test OpenAI
OPENAI_API_KEY=sk-... python -c "
from autoresearch.llm_backend import get_llm_backend
backend = get_llm_backend()
print(f'Backend: {backend.name()}')
assert backend.validate()
print('OpenAI backend validated')
"
```

**Pass criteria:** Both providers authenticate successfully.

### L4.4 — Crash recovery
Simulate a mid-experiment crash and verify recovery.

```bash
# 1. Start agent with MAX_TURNS=5
# 2. After 2 experiments complete, send SIGTERM
# 3. Verify results.tsv has 2 complete rows (no corruption)
# 4. Restart agent — verify it resumes from experiment 3
# 5. Verify heartbeat file shows alive=false after stop, alive=true after restart
```

**Pass criteria:** No data loss. Clean resume from last completed experiment.

### L4.5 — Git experiment tracking
Verify that each experiment creates a proper git commit.

```bash
# 1. Run 3 experiments in a git repo
# 2. git log --oneline -> should show exp0, exp1, exp2 commits
# 3. For a discarded experiment: verify revert commit exists
# 4. For a kept experiment: verify training script diff shows HP changes
```

**Pass criteria:** Git history accurately reflects experiment decisions.

---

## Level 5: Extended Validation (RunPod GPU, 2-8 hours)

### L5.1 — 30-experiment overnight run
The full-scale validation that exercises all components under realistic conditions.

```bash
export AGENT_MAX_TURNS=30
export RESULTS_TSV=results/overnight_test.tsv
python agent.py "Minimize val_bpb through autonomous hyperparameter optimization"
```

**Verification checklist:**
- [ ] All 30 experiments complete (baseline + 29 modifications)
- [ ] results.tsv has 30 valid rows
- [ ] val_bpb improves from baseline (at least 1 "keep" experiment)
- [ ] Strategy diversity: at least 3 different categories explored
- [ ] No memory leaks: peak_mem_gb doesn't grow monotonically
- [ ] Power metrics populated (watts > 0 on supported hardware)
- [ ] Drift monitor would have fired if plateau occurred (verify retroactively)

### L5.2 — Drift monitor validation with real data
After the overnight run, retroactively verify drift detection.

```python
from autoresearch.results import load_results
from drift.monitor import DriftMonitor

results = load_results("results/overnight_test.tsv")
monitor = DriftMonitor()

# Check at each point in the run
for i in range(10, len(results)):
    report = monitor.check(results[:i])
    if report.has_alerts:
        print(f"At experiment {i}: {report.format_summary()}")
        for alert in report.all_alerts:
            print(f"  [{alert.level}] {alert.message[:80]}")
```

**Pass criteria:** Alerts correlate with actual performance trends. No false positives on steady improvement. Plateau alerts fire when val_bpb stalls.

### L5.3 — Energy efficiency tracking
```python
results = load_results("results/overnight_test.tsv")
for r in results:
    if r.watts > 0:
        print(f"{r.exp}: {r.val_bpb:.4f} bpb, {r.watts:.0f}W, {r.joules_per_token:.4f} J/tok")

# Verify joules_per_token = watts * training_seconds / total_tokens (approximately)
```

**Pass criteria:** Power metrics are consistent across experiments on the same hardware.

---

## Level 6: Docker Isolation (RunPod or local Docker, 30 min)

### L6.1 — Docker build
```bash
cd autoagent-unified
docker build -f Dockerfile.base -t autoagent-unified:test .
```

**Pass criteria:** Image builds without errors.

### L6.2 — Docker unit tests
```bash
docker run --rm autoagent-unified:test python -m pytest tests/ -v
```

**Pass criteria:** All 49 tests pass inside the container.

### L6.3 — Docker GPU experiment (requires nvidia-docker)
```bash
docker run --rm --gpus all \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e AGENT_MAX_TURNS=2 \
  autoagent-unified:test \
  python agent.py "Run 2 experiments"
```

**Pass criteria:** Agent runs and produces results inside the container.

---

## Regression Test Matrix

Run on every commit to `autoagent-unified/`:

| Test | Time | GPU | CI-friendly |
|------|------|-----|-------------|
| L0.1 Import validation | 5s | No | Yes |
| L0.2 Dependency install | 30s | No | Yes |
| L0.3 License check | 2s | No | Yes |
| L1.1 pytest suite (49 tests) | 10s | No | Yes |
| L2.1 Results round-trip | 5s | No | Yes |
| L2.2 Drift end-to-end | 5s | No | Yes |
| L2.4 LLM factory (mocked) | 5s | No | Yes |
| L2.5 Response parsing | 5s | No | Yes |

Run before releases / after significant changes:

| Test | Time | GPU | CI-friendly |
|------|------|-----|-------------|
| L3.1 Hardware detection | 5s | Yes | No |
| L3.3 Single experiment | 6min | Yes | No |
| L4.1 Three-experiment loop | 20min | Yes | No |
| L4.4 Crash recovery | 15min | Yes | No |
| L5.1 30-experiment overnight | 3hr | Yes | No |
| L6.1-6.3 Docker validation | 30min | Yes | No |

---

## Known Limitations / What This Plan Does NOT Cover

1. **Outer loop (meta-agent)** — The meta-agent that iterates on agent.py itself is from autoagent upstream and requires Harbor benchmark infrastructure. Testing the full two-level optimization loop requires Harbor setup (not covered here).

2. **Multi-GPU / distributed** — All tests assume single-GPU execution. Multi-device training (e.g., 8x Gaudi 3) is out of scope.

3. **OpenAI Agents SDK** — The `openai-agents` package integration in `create_agent()` requires that SDK to be installed. If unavailable, the agent falls back to direct LLM backend calls. Both paths should be tested but the Agents SDK path requires `pip install openai-agents`.

4. **Cross-platform matrix** — Ideally test on CUDA, ROCm, and Apple Silicon. This plan assumes CUDA (RunPod) as the primary target. Adjust training script paths for other backends.

5. **Network resilience** — API rate limits, connection drops, and retry logic are inherited from autoresearch-unified's orchestrator (not yet ported into the agent.py path). Full retry testing requires mocking API failures.
