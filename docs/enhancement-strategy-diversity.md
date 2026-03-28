# Enhancement: Improve LLM Strategy Diversity via Prompt & History Enhancements

## Context

Telemetry analysis of Run 1 (70/100 experiments, RTX 5090, GPT-4.1 via OpenRouter) reveals that while the system achieves a high keep rate (30%), it converges on a narrow optimization strategy (LR tuning) and fails to discover architectural improvements that drove PR#17's breakthrough results (0.917 val_bpb via batch halving + window changes). The LLM proposes from a limited strategy space: 51% of proposals are LR/schedule tweaks, only 5% attempt batch changes, and 0% attempt novel hyperparameters like ADAM_BETAS or WARMUP_RATIO > 0.

This document proposes enhancements to improve strategy diversity, reduce duplicate proposals, and help the LLM break out of local optima.

## Telemetry Evidence

### Proposal Category Distribution (63 LLM calls, 5090 R1)

| Category | Count | % | Keeps |
|----------|:-----:|:-:|:-----:|
| LR tuning (MATRIX_LR, SCALAR_LR, EMBEDDING_LR, UNEMBEDDING_LR) | 23 | 37% | ~5 |
| Architecture ratios (MLP_RATIO, ASPECT_RATIO) | 13 | 21% | ~4 |
| Warmdown/warmup ratio | 9 | 14% | ~2 |
| Weight decay | 8 | 13% | ~2 |
| Architecture structural (DEPTH, HEAD_DIM, WINDOW_PATTERN) | 7 | 11% | ~1 |
| Batch size | 3 | 5% | 0 |

### Never-Attempted Hyperparameters

- `ADAM_BETAS` (beta1, beta2) — never modified across any run
- `WARMUP_RATIO > 0` — always left at 0.0
- `ACTIVATION_CHECKPOINTING` — never toggled
- `COMPILE_MODE` — never changed
- `TOTAL_BATCH_SIZE` halving — attempted 3x, all crashed (vs PR#17 where this was THE breakthrough)

### Cost Analysis

| Metric | Value |
|--------|-------|
| Total API cost (63 calls) | $0.55 |
| Cost per experiment | $0.0088 |
| Cost per kept result | $0.026 |
| Projected full run (100 exp) | $0.88 |
| Cached input tokens | 43.2% |
| Reasoning tokens | 0 (non-reasoning mode) |

### Cross-Platform Comparison (5090 vs PR#17 5070 Ti)

| Metric | 5090/OpenRouter (n=70) | PR#17 R1/Azure (n=119) | PR#17 R2/Azure (n=133) | PR#17 R3/Azure (n=126) |
|--------|:-:|:-:|:-:|:-:|
| Keep rate | **30%** | 5.0% | 1.5% | 1.6% |
| Crash rate | **8.6%** | 20.2% | 32.3% | 33.3% |
| Best val_bpb | 1.0638 | **0.9166** | 1.0643 | 0.9700 |
| Strategy | Conservative LR tuning | Exploratory (batch/arch) | Mixed | Mixed |

## Files to Modify

| File | Change |
|------|--------|
| `tui/llm_backend.py` | System prompt enhancements, strategy diversity hints |
| `tui/results.py` | History formatting with strategy annotations |
| `tui/orchestrator.py` | Stagnation detection, strategy nudging, duplicate detection |
| `docs/telemetry-analysis-5090-r1.md` | New file: full telemetry analysis writeup |

## Enhancement 1: Strategy Diversity Hints in System Prompt

**File:** `tui/llm_backend.py` (system prompt in `get_system_prompt()`)

**Problem:** The current system prompt says "If many experiments have been discarded, try a different direction entirely" but provides no guidance on WHAT different directions exist. The LLM defaults to its training prior (conservative LR tuning).

**Solution:** Add an explicit "strategy menu" to the system prompt:

```python
Strategy guidance (use the full repertoire, not just LR tuning):
- Learning rate tuning: MATRIX_LR, SCALAR_LR, EMBEDDING_LR, UNEMBEDDING_LR
- Regularization: WEIGHT_DECAY, ADAM_BETAS (beta1, beta2)
- Schedule shape: WARMUP_RATIO (try >0), WARMDOWN_RATIO, FINAL_LR_FRAC (try >0)
- Architecture: DEPTH, ASPECT_RATIO, HEAD_DIM, MLP_RATIO, WINDOW_PATTERN
- Throughput: TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE (halving batch = 2x steps = often big wins)
- Untried levers: ACTIVATION_CHECKPOINTING (enables deeper models), COMPILE_MODE
```

**Hypothesis:** Explicitly listing batch halving as "often big wins" may prompt the LLM to attempt PR#17's breakthrough strategy.

## Enhancement 2: Stagnation Detection & Strategy Nudging

**File:** `tui/orchestrator.py`, new method + modification to `_run_experiment()`

**Problem:** After ~15 experiments, the 5090 run entered a diminishing-returns plateau where the LLM keeps proposing slight LR decreases that get discarded. The system has no mechanism to detect or break this loop.

**Solution:** Add stagnation detection that modifies the prompt when the LLM is stuck:

```python
def _detect_stagnation(self) -> str | None:
    """Detect if the LLM is stuck in a narrow strategy space.
    Returns a nudge message or None."""
    results = load_results(self._results_path)
    if len(results) < 15:
        return None

    recent = results[-15:]
    recent_keeps = sum(1 for r in recent if r.status == "keep")

    if recent_keeps <= 1:
        lr_count = sum(1 for r in recent if any(x in r.description.lower()
            for x in ["_lr", "learning rate"]))

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
```

**Integration point:** Append the nudge to the user prompt before calling `_call_llm_with_backoff()`.

**Hypothesis:** Explicit stagnation detection prevents the LLM from wasting 30+ experiments on exhausted strategy spaces.

## Enhancement 3: History Formatting with Strategy Category Annotations

**File:** `tui/results.py`, modify `format_history_for_prompt()`

**Problem:** The LLM sees a flat table of experiments but has no summary of what strategy categories have been explored. It can't easily see "I've tried 23 LR changes and only 5 architectural changes."

**Solution:** Add a strategy summary footer to the formatted history:

```python
def format_history_for_prompt(path: str = "results.tsv") -> str:
    results = load_results(path)
    if not results:
        return "No experiments yet."

    # ... existing table formatting ...

    # Add strategy summary
    categories = categorize_experiments(results)
    lines.append("")
    lines.append("Strategy summary:")
    for cat, count in categories.items():
        kept = sum(1 for r in results if r.status == "keep"
                   and classify_experiment(r) == cat)
        lines.append(f"  {cat}: {count} tried, {kept} kept")

    return "\n".join(lines)
```

**Hypothesis:** Making the strategy distribution visible helps the LLM self-correct its exploration pattern.

## Enhancement 4: Duplicate Proposal Detection

**File:** `tui/orchestrator.py`, new method

**Problem:** Telemetry shows 8 near-duplicate proposals (identical changes re-proposed). While the code includes all history, the LLM sometimes fails to notice it already tried a similar change.

**Solution:** Before applying a proposal, check for near-duplicates:

```python
def _is_near_duplicate(self, description: str) -> bool:
    """Check if a very similar experiment was already attempted."""
    results = load_results(self._results_path)
    desc_lower = description.lower().strip()

    for r in results:
        existing = r.description.lower().strip()
        if desc_lower == existing:
            return True
        if _same_param_same_direction(desc_lower, existing):
            return True
    return False
```

On duplicate detection, append a note to the prompt and re-query the LLM (max 2 retries).

**Hypothesis:** Preventing duplicates forces exploration of new strategy spaces.

## Enhancement 5: Telemetry Analysis Document

**File:** `docs/telemetry-analysis-5090-r1.md` (new)

Full analysis writeup including:
1. Cost analysis and token usage patterns
2. Proposal distribution across strategy categories
3. Never-attempted hyperparameters inventory
4. Cross-platform comparison (OpenRouter vs Azure behavioral differences)
5. Telemetry methodology (Arize Phoenix JSONL export structure)
6. Hypotheses for investigation

## Hypotheses to Investigate

| # | Hypothesis | Test Method | Priority |
|---|-----------|-------------|----------|
| H1 | Batch halving is the single highest-impact change regardless of GPU | Run 5090 R2 with strategy hint emphasizing batch changes | **High** |
| H2 | OpenRouter vs Azure GPT-4.1 have different sampling/temperature defaults causing strategy divergence | Compare raw API parameters in telemetry; run identical prompt through both endpoints | **High** |
| H3 | 32GB VRAM prevents the OOM crashes that accidentally force the LLM to try smaller models (which then run more steps) | Compare crash types: 5090 crashes are batch-related, 5070Ti crashes may include OOM-forced innovations | **Medium** |
| H4 | Strategy diversity hint in system prompt increases keep rate by >50% | A/B test: run with/without strategy hints on same hardware | **Medium** |
| H5 | Stagnation nudging reduces wasted experiments by >30% | Measure experiments-between-keeps with and without nudging | **Medium** |
| H6 | The one-change-per-experiment rule prevents compound improvements that would match PR#17's batch+architecture combos | Experimental: allow 2-change proposals after exp50 | **Low** |
| H7 | WINDOW_PATTERN "SSLL" discovery (exp38, best single-step gain) validates that architectural changes outperform LR tuning | Track category-vs-improvement-magnitude across all runs | **Low** |
| H8 | Arize JSONL export ordering is non-chronological, and "history resets" are an artifact not a runtime behavior | Add timestamps to JSONL records, verify against results.tsv ordering | **Low** |

## Implementation Plan

```
Branch: enhancement/strategy-diversity-and-telemetry
Files:
  tui/llm_backend.py               — Strategy hints in system prompt
  tui/results.py                    — Strategy summary in history formatting
  tui/orchestrator.py               — Stagnation detection + duplicate detection
  docs/telemetry-analysis-5090-r1.md — Full analysis writeup
  tests/test_stagnation.py          — Unit tests for stagnation detection
  tests/test_duplicate.py           — Unit tests for duplicate proposal detection
```

## Verification

1. **Unit tests:** `pytest tests/test_stagnation.py tests/test_duplicate.py`
2. **Prompt inspection:** Verify strategy hints appear in system prompt output
3. **Integration test:** Short headless session (`--max 5`) confirming no regressions
4. **Live validation:** Run 5090 R2 with enhancements, compare strategy distribution vs R1
5. **PR comment:** Post before/after strategy distribution comparison on PR#17
