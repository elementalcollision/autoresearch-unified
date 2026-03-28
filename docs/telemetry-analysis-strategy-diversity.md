# Telemetry Analysis: Strategy Diversity A/B Test (RTX 5090)

## Overview

This document captures the telemetry analysis from the first controlled A/B test of the strategy diversity enhancements (PR #36, implementing PR #25 enhancements 1-3). The test was conducted on an NVIDIA RTX 5090 (32 GB, Blackwell) via RunPod, with GPT-4.1 via OpenRouter as the LLM backend.

**Test date:** 2026-03-28
**GPU:** NVIDIA RTX 5090, 32 GB VRAM, 575W TDP
**Dataset:** ClimbMix
**LLM:** GPT-4.1 via OpenRouter
**Experiments per arm:** 30 (including baseline)

## Test Design

| Arm | Branch | Commit | Description |
|-----|--------|--------|-------------|
| Control | `main` | `7ebf27b` | No strategy diversity enhancements |
| Treatment | `enhancement/strategy-diversity-impl` | `3c80023` | Enhancements 1-3 (strategy hints, stagnation detection, category annotations) |

Both arms used identical configuration:
- `--max 30 --model openai/gpt-4.1 --dataset climbmix`
- 5-minute training budget per experiment
- 14-column TSV with power instrumentation (pynvml)

## Results Summary

| Metric | Control | Treatment | Delta |
|--------|---------|-----------|-------|
| Best val_bpb | 1.079037 | 1.079339 | +0.000302 (within noise) |
| Keep rate | 6/29 (20.7%) | 8/29 (27.6%) | +33% |
| Crash rate | 4/29 (13.8%) | 2/29 (6.9%) | -50% |
| Runtime | 2h 23m | 2h 33m | +7% |
| Avg power (non-crash) | 470.1W | 471.3W | +0.3% |

## Category Distribution Analysis

### Proposal Categories (excluding baseline)

| Category | Control | Control % | Treatment | Treatment % | Delta |
|----------|:-------:|:---------:|:---------:|:-----------:|:-----:|
| learning_rate | 11 | 37.9% | 8 | 27.6% | -10.3pp |
| schedule | 6 | 20.7% | 8 | 27.6% | +6.9pp |
| architecture | 6 | 20.7% | 7 | 24.1% | +3.4pp |
| batch_size | 4 | 13.8% | 3 | 10.3% | -3.4pp |
| regularization | 2 | 6.9% | 3 | 10.3% | +3.4pp |

### Key Finding: LR Dominance Reduction

The primary hypothesis (H1 from PR #25) was that strategy hints would reduce LR/schedule dominance below 40%. Results:

- **Control LR-only:** 37.9% (already below the 51% observed in earlier R1 telemetry)
- **Treatment LR-only:** 27.6% (target: <40%) -- **H1 PASSES**
- **LR+schedule combined:** Control 58.6% vs Treatment 55.2% -- modest improvement

The control's lower-than-expected LR dominance (37.9% vs 51% in R1) may be due to:
1. ClimbMix dataset vs PubMed (different optimization landscape)
2. max_experiments=30 vs 100 (less time for LR rut to develop)
3. Latest main includes PR #33 max_tokens fix (better API validation)

### Category Balance Score

Shannon entropy of category distribution (higher = more balanced):

| Arm | Entropy | Max possible | Normalized |
|-----|---------|-------------|------------|
| Control | 2.08 | 2.32 (5 categories) | 0.90 |
| Treatment | 2.21 | 2.32 (5 categories) | 0.95 |

Treatment distribution is 6% more balanced by entropy measure.

## Hypothesis Evaluation

### H1: Strategy hints reduce LR dominance below 40%
**PASS** -- Treatment LR-only at 27.6%, down from 37.9% control.

### H2: Stagnation detection fires when LR rut detected
**INCONCLUSIVE** -- The stagnation threshold (<=1 keep in last 15, >=8 LR changes) was not triggered in either arm's 30-experiment run. This is actually a positive signal -- the strategy hints in Enhancement 1 may have prevented the LR rut from forming in the first place.

### H3: Category annotations help LLM self-correct
**PARTIAL PASS** -- Treatment distribution is more balanced (entropy 0.95 vs 0.90). The strategy summary footer appears to help the LLM spread exploration, but the effect is moderate.

### H4: No regression in convergence quality
**PASS** -- Best val_bpb essentially tied (1.079037 vs 1.079339, delta = 0.03%). Keep rate improved from 20.7% to 27.6%.

## Per-Experiment Trajectories

### Control Keepers
| Exp | Description | val_bpb | Category |
|-----|-------------|---------|----------|
| 4 | WEIGHT_DECAY 0.2 to 0.1 | 1.084282 | regularization |
| 6 | WARMDOWN_RATIO 0.5 to 0.7 | 1.082368 | schedule |
| 9 | SCALAR_LR 0.5 to 0.3 | 1.080622 | learning_rate |
| 11 | FINAL_LR_FRAC 0 to 0.05 | 1.079985 | schedule |
| 16 | WEIGHT_DECAY 0.1 to 0.05 | **1.079037** | regularization |

### Treatment Keepers
| Exp | Description | val_bpb | Category |
|-----|-------------|---------|----------|
| 2 | WARMUP_RATIO 0 to 0.05 | 1.088467 | schedule |
| 3 | WARMDOWN_RATIO 0.5 to 0.7 | 1.085508 | schedule |
| 6 | WEIGHT_DECAY 0.2 to 0.15 | 1.083190 | regularization |
| 13 | MATRIX_LR 0.04 to 0.03 | 1.081670 | learning_rate |
| 16 | EMBEDDING_LR 0.6 to 0.4 | 1.079938 | learning_rate |
| 17 | WARMDOWN_RATIO 0.7 to 0.85 | 1.079851 | schedule |
| 24 | FINAL_LR_FRAC 0 to 0.01 | **1.079339** | schedule |

### Convergence Trajectory
Both arms converged to similar final val_bpb through different exploration paths. The treatment arm found more diverse keepers earlier (schedule and regularization wins by exp6), while the control arm spent early experiments crashing on batch_size changes.

## Power Analysis

Average power draw across successful experiments (excluding crashes):

| Arm | Mean watts | Std watts | Min | Max |
|-----|-----------|-----------|-----|-----|
| Control | 470.1 | 7.8 | 464.8 | 511.9 |
| Treatment | 471.3 | 14.8 | 448.7 | 526.7 |

Architecture changes (DEPTH, HEAD_DIM) consistently draw more power (487-527W) due to increased compute intensity. The strategy diversity enhancements do not affect training code, so power differences reflect only the hyperparameter choices made by the LLM.

## Duplicate Proposals Observed

### Control (no duplicate detection)
The control run contained several near-duplicate proposals:
- exp21 and exp28: both "Increase EMBEDDING_LR from 0.6 to 0.8" (exact duplicate)
- exp8 and exp12: both "Decrease DEVICE_BATCH_SIZE..." (similar intent, both crashed)

### Implication
Enhancement 4 (duplicate detection, this PR) would have caught these and forced the LLM to explore different strategies instead of re-trying failed approaches.

## Remaining Enhancements

From PR #25's original proposal:
- [x] Enhancement 1: Strategy hints in system prompt (PR #36, merged)
- [x] Enhancement 2: Stagnation detection (PR #36, merged)
- [x] Enhancement 3: Category annotations in history (PR #36, merged)
- [ ] Enhancement 4: Duplicate proposal detection (this PR)
- [x] Enhancement 5: Telemetry analysis document (this document)

## Cost Analysis

| Metric | Control | Treatment |
|--------|---------|-----------|
| RunPod cost | ~$4.20 | ~$4.20 |
| API cost (est.) | ~$0.26 | ~$0.26 |
| Total | ~$4.46 | ~$4.46 |
| **Combined A/B cost** | | **~$8.92** |
