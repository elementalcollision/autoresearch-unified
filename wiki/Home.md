# Autoresearch Unified Wiki

Consolidated documentation for the Autoresearch cross-platform hyperparameter optimization project.

## Quick Links

- **[HuggingFace Dataset](https://huggingface.co/datasets/davegraham/autoresearch-experiments)** — All experiment data (Croissant-compliant)
- **[Cross-Platform Overview](Cross-Platform-Overview)** — Key findings and normalized comparisons
- **[Data Access Guide](Data-Access)** — How to load and query the dataset

## Experiment Summary

| Stat | Value |
|------|-------|
| **Total experiments** | 2,637 |
| **Platforms** | 3 (Apple Metal, NVIDIA CUDA, AMD ROCm) |
| **GPU models** | 5 |
| **Datasets** | 7 |
| **Primary metric** | val_bpb (validation bits-per-byte, lower = better) |

## By Platform

| Platform | Experiments | GPUs | Page |
|----------|-------------|------|------|
| Apple Metal (MLX/MPS) | 713 | M5 Max | [Platform: Apple Metal](Platform-Apple-Metal) |
| NVIDIA CUDA | 1,602 | RTX 4000 Ada, A100 40GB, RTX Pro 6000 Blackwell | [Platform: NVIDIA CUDA](Platform-NVIDIA-CUDA) |
| AMD ROCm | 322 | MI300X | [Platform: AMD ROCm](Platform-AMD-ROCm) |
| Intel Gaudi | 0 (pending) | Gaudi 3 | [Platform: Intel Gaudi](Platform-Intel-Gaudi) |

## By Dataset

| Dataset | Platforms Tested | Best val_bpb | Best GPU | Page |
|---------|-----------------|--------------|----------|------|
| ClimbMix | 4 GPUs | 1.036 | MI300X | [Dataset: ClimbMix](Dataset-Climbmix) |
| Cosmopedia-v2 | 4 GPUs | 0.697 | A100 40GB | [Dataset: Cosmopedia-v2](Dataset-Cosmopedia-v2) |
| FineWeb-Edu | 4 GPUs | 1.015 | MI300X | [Dataset: FineWeb-Edu](Dataset-FineWeb-Edu) |
| FineWeb-Edu-High | 3 GPUs | 1.099 | RTX Pro 6000 | [Dataset: FineWeb-Edu-High](Dataset-FineWeb-Edu-High) |
| FineWeb | 3 GPUs | 1.231 | A100 40GB | [Dataset: FineWeb](Dataset-FineWeb) |
| SlimPajama | 4 GPUs | 1.015 | MI300X | [Dataset: SlimPajama](Dataset-SlimPajama) |
| GitHub-Code-Python | 2 GPUs | 0.549 | A100 40GB | [Dataset: GitHub-Code-Python](Dataset-GitHub-Code-Python) |

## Tools

- [TUI Dashboard](Tools-TUI-Dashboard)
- [Remote Orchestration](Tools-Remote-Orchestration)

## About

This wiki consolidates documentation from 4 platform-specific wikis into a single, topic-organized resource. The structured experiment data lives on [HuggingFace](https://huggingface.co/datasets/davegraham/autoresearch-experiments) as a Croissant-compliant dataset. See [Data Access](Data-Access) for usage instructions.
