# Autoresearch Unified Wiki

Autonomous LLM-driven hyperparameter optimization across GPU platforms. Claude proposes hyperparameter changes, trains a GPT model for 5 minutes, evaluates val_bpb, keeps or discards, repeats.

## Quick Links

- **[Cross-Platform Overview](Cross-Platform-Overview)** — Current results, environment details, and analysis
- **[HuggingFace Dataset](https://huggingface.co/datasets/davegraham/autoresearch-experiments)** — Structured experiment data (Croissant-compliant)
- **[Data Access Guide](Data-Access)** — How to load and query the dataset

## Current Run: Unified Codebase v2

> Running since March 24, 2026 on commit `dacda45` from `autoresearch-unified`. All platforms use identical code, identical LLM (Claude Sonnet 4), and 11-column TSV with `baseline_sha` traceability.

| Stat | Value |
|------|-------|
| **Active platforms** | 2 (NVIDIA CUDA, AMD ROCm) |
| **Active GPUs** | RTX PRO 6000 Blackwell, MI300X |
| **Datasets in queue** | 8 (ClimbMix through PubMed-Abstract) |
| **Experiments per dataset** | 80 |
| **Total target** | 1,280 experiments (8 datasets x 2 GPUs x 80 each) |
| **Primary metric** | val_bpb (validation bits-per-byte, lower = better) |
| **Codebase** | [autoresearch-unified](https://github.com/elementalcollision/autoresearch-unified) |

### Progress

| Dataset | RTX PRO 6000 | MI300X |
|---------|-------------|--------|
| **ClimbMix** | 79/80 (best: **1.057**) | 80/80 (best: **1.086**) |
| **FineWeb-Edu** | _in progress_ | _in progress_ |
| Cosmopedia-v2 | queued | queued |
| SlimPajama | queued | queued |
| FineWeb-Edu-High | queued | queued |
| FineWeb | queued | queued |
| GitHub-Code-Python | queued | queued |
| PubMed-Abstract | queued | queued |

## By Platform

| Platform | Status | GPUs | Page |
|----------|--------|------|------|
| NVIDIA CUDA | Active | RTX PRO 6000 Blackwell (102 GB) | [Platform: NVIDIA CUDA](Platform-NVIDIA-CUDA) |
| AMD ROCm | Active | MI300X (192 GB) | [Platform: AMD ROCm](Platform-AMD-ROCm) |
| Apple Metal (MLX/MPS) | Planned | M5 Max | [Platform: Apple Metal](Platform-Apple-Metal) |
| Intel Gaudi | Blocked (IBM quota) | Gaudi 3 | [Platform: Intel Gaudi](Platform-Intel-Gaudi) |

## By Dataset

| Dataset | GPUs Tested | Best val_bpb | Best GPU | Page |
|---------|------------|--------------|----------|------|
| ClimbMix | 2 | **1.057** | RTX PRO 6000 | [Dataset: ClimbMix](Dataset-Climbmix) |
| FineWeb-Edu | 2 | _running_ | — | [Dataset: FineWeb-Edu](Dataset-FineWeb-Edu) |
| Cosmopedia-v2 | — | _queued_ | — | [Dataset: Cosmopedia-v2](Dataset-Cosmopedia-v2) |
| SlimPajama | — | _queued_ | — | [Dataset: SlimPajama](Dataset-SlimPajama) |
| FineWeb-Edu-High | — | _queued_ | — | [Dataset: FineWeb-Edu-High](Dataset-FineWeb-Edu-High) |
| FineWeb | — | _queued_ | — | [Dataset: FineWeb](Dataset-FineWeb) |
| GitHub-Code-Python | — | _queued_ | — | [Dataset: GitHub-Code-Python](Dataset-GitHub-Code-Python) |
| PubMed-Abstract | — | _queued_ | — | — |

## Tools & Testing

- [Sanity Testing](Sanity-Testing) — Integration test results and data integrity validation
- [Data Access Guide](Data-Access) — How to load experiment data from HuggingFace
- [Archive: Cross-Platform Overview v1](Archive-Cross-Platform-Overview-v1) — Pre-unification results (4 separate repos)

## About

This project uses an autonomous LLM agent (Claude) to optimize hyperparameters for a GPT-style language model across multiple GPU platforms and datasets. Each experiment runs for exactly 5 minutes, producing a val_bpb score. The agent learns from previous results to propose better hyperparameters.

The unified codebase replaced 4 separate platform-specific repositories in March 2026. Key improvements include cross-dataset baseline isolation, `baseline_sha` traceability, atomic result writes, and automated GitHub sync.
