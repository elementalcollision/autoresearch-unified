# autoagent-unified — Human Directives

## Mission

Autonomously optimize GPU pretraining of a small language model to minimize
**val_bpb** (validation bits per byte). The agent iterates on hyperparameters,
guided by experiment history and drift-monitor alerts.

## Two-Level Optimization

### Inner Loop (autoresearch experiment loop)
- LLM proposes a single hyperparameter change
- Train for 5 minutes on the detected GPU
- Evaluate val_bpb
- Keep if improved, discard if not
- Record results in 14-column TSV

### Outer Loop (autoagent meta-optimization)
- Meta-agent reads program.md + experiment results + drift alerts
- Iterates on agent.py: modifies tools, prompts, and strategy
- Keeps changes that improve benchmark scores, discards those that don't
- Hill-climbing with full rollback safety

## Strategy Guidance

1. **Start with throughput optimization.** Smaller batches = more gradient steps
   in the fixed 5-minute budget. This is consistently the highest-leverage lever.

2. **Prompt tuning has diminishing returns.** After initial gains, adding
   specialized tools is a higher-leverage improvement axis.

3. **Drift-monitor integration is critical.** When the drift monitor reports:
   - **Performance plateau:** Switch to a fundamentally different hyperparameter axis
   - **Strategy repetition:** The LLM is stuck. Force exploration of underrepresented categories
   - **Hardware anomaly:** Adjust batch sizes or reduce depth to work around thermal limits
   - **Low keep ratio:** The current strategy direction is exhausted

4. **Energy efficiency matters.** Track joules_per_token alongside val_bpb.
   Two experiments with similar val_bpb should prefer the more energy-efficient one.

## Multi-Provider LLM

The system supports multiple LLM providers equally:
- **Claude** (Anthropic): Set `ANTHROPIC_API_KEY`
- **OpenAI** (GPT-5, etc.): Set `OPENAI_API_KEY`
- **Azure OpenAI**: Set `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`
- **OpenRouter** (200+ models): Set `OPENROUTER_API_KEY`

The factory function auto-detects which provider to use based on available credentials.

## Multi-Backend Hardware

Experiments run on the detected GPU platform:
- **NVIDIA CUDA** (H100, A100, RTX series)
- **AMD ROCm** (MI300X, MI250, etc.)
- **Intel Gaudi** (Gaudi 3)
- **Apple Silicon** (MLX on M-series)

The platform is auto-detected. Override with `AUTORESEARCH_BACKEND=cuda|rocm|hpu|mlx`.

## Iteration Rules

- **Keep threshold:** New val_bpb must be strictly less than current best
- **Discard handling:** Revert training script to baseline, but keep the result in TSV for learning
- **Crash handling:** Record with val_bpb=0, don't count against strategy
- **Maximum experiments:** Configurable via `AGENT_MAX_TURNS` (default: 30)

## What NOT to Change

- Model architecture (GPT-2 variant) is fixed
- Optimizer (Muon variants) is fixed per platform
- Evaluation protocol (val_bpb on held-out data) is fixed
- Docker isolation for agent execution is preserved
