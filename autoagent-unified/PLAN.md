# autoagent-unified Integration Plan

See the full plan at: `/root/.claude/plans/vast-bubbling-avalanche.md`

## Summary

This project merges two autonomous AI systems:
- **autoagent** (kevinrgu/autoagent): Meta-agent optimization via hill-climbing
- **autoresearch-unified**: Autonomous LLM-driven GPU pretraining research

The combined system uses autoagent's meta-optimization layer to drive
autoresearch's experiment loop, while autoresearch's battle-tested
infrastructure replaces autoagent's simpler primitives. A new drift-monitor
detects performance plateaus, hardware anomalies, and strategy repetition.

## Architecture

```
Outer Loop (autoagent meta-agent)
  reads program.md + benchmark results
  iterates on agent.py (tools, prompts, strategy)
    |
    v
Inner Loop (autoresearch orchestrator)
  LLM proposes hyperparameters
  train 5 min -> evaluate val_bpb
  keep/discard -> repeat
    |
    v
Drift Monitor
  performance drift (plateau, regression)
  hardware drift (thermal, power, memory)
  strategy drift (repetition, local optima)
  -> injects warnings into LLM prompt
```
