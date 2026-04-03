# Running Reasoning Models in Autoresearch

Reasoning models (Kimi K2.5, GPT-OSS-120B, o3, o4-mini) work differently from standard models. They spend completion tokens on internal chain-of-thought before producing the response you actually see. This creates specific failure modes in the autoresearch pipeline that don't occur with standard models like Sonnet 4.6 or GPT-4.1.

This guide documents what we learned running Kimi K2.5 through 300 controlled experiments (n=3, 100 each) on an RTX 5070 Ti via Azure AI Foundry.

## The core problem: token budget exhaustion

Standard models use ~500 completion tokens per experiment proposal. Reasoning models use 5,000-15,000 because the chain-of-thought is billed as completion tokens.

With the default `max_tokens=2048`, a reasoning model burns the entire budget on thinking and returns `content=None`. The pipeline sees an empty response, can't parse it, and skips the experiment. With 100 experiments, you lose 3-30 to this failure depending on prompt length.

The fix is already in the codebase (PR #45). `AzureOpenAIBackend` detects reasoning models and uses `max_tokens=32768` automatically. The detection is based on deployment name:

```python
REASONING_MODELS = {"kimi-k2.5", "o3", "o3-mini", "o4-mini"}
```

If your reasoning model isn't in this set, add it to `tui/llm_backend.py` or set `max_tokens` manually via environment variable (not yet supported -- see "Adding new reasoning models" below).

## Token budget grows with experiment history

This is the non-obvious part. Early experiments work fine because the prompt is short (~5K tokens). But as the experiment history grows (80+ experiments), the prompt reaches 10-15K tokens. The model's chain-of-thought scales with prompt complexity. What worked at experiment 5 fails at experiment 50.

Our Kimi K2.5 R1 run worked perfectly for 9 experiments, then hit an infinite unparseable loop at experiment 10. We bumped `max_completion_tokens` from 4,096 to 16,384 and it ran to experiment 93. Bumped again to 32,768 and it completed. The current default of 32,768 should be sufficient for 100-experiment runs.

If you're running longer sessions (200+ experiments) or using models with unusually verbose reasoning, you may need to increase `REASONING_MAX_TOKENS` in `tui/llm_backend.py`.

## The silent failure mode

Before PR #45, when a reasoning model returned `content=None`, the pipeline would:

1. Pass `None` to `parse_llm_response()`
2. Get back `None` (unparseable)
3. Skip the experiment
4. Decrement the loop counter

No error was raised. No retry was attempted. The experiment slot was permanently wasted. With a 100-experiment budget, losing 10-30 slots to silent failures meant your final TSV had 70-90 rows instead of 100.

After PR #45, the `AzureOpenAIBackend.generate_experiment()` method raises `ValueError` when `content` is `None`. This is caught by the orchestrator's `_call_llm_with_backoff()` retry loop, which retries with exponential backoff (10s, 20s, 30s, 40s, 50s). The retry usually succeeds because the model's chain-of-thought is non-deterministic -- it may use fewer reasoning tokens on the next attempt.

## Validation differences

The `validate()` method sends "Say OK" with a small token budget to test credentials. Standard models respond in 2-3 tokens. Reasoning models may spend all tokens on thinking about how to say "OK" and return `content=None`.

The backend now uses 512 tokens for reasoning model validation (vs 16 for standard models). This is enough for the model to think and respond, without burning significant API cost on a preflight check.

## Cost differences

Reasoning tokens are billed as completion tokens. A single experiment proposal costs:

| Model | Completion tokens | Cost per proposal |
|-------|------------------|-------------------|
| Sonnet 4.6 | ~500 | ~$0.008 |
| GPT-4.1 | ~400 | ~$0.003 |
| Kimi K2.5 | ~9,000 | ~$0.023 |
| GPT-OSS-120B | ~250 | ~$0.0002 |

Kimi K2.5 generates verbose reasoning (10-15K chars of chain-of-thought per proposal) but the actual pricing is cheap ($0.60/$2.50 per 1M tokens). GPT-OSS-120B is extremely efficient -- short reasoning, cheap tokens.

For a full 100-experiment run, API cost is dominated by the number of proposals, not the per-proposal cost. Budget $2-5 for a reasoning model run via Azure.

## Azure AI Foundry deployment notes

Reasoning models on Azure AI Foundry use the same Chat Completions API as standard Azure OpenAI models, but there are deployment differences:

1. **Endpoint format.** Third-party models (Kimi K2.5) may use `*.services.ai.azure.com` instead of `*.openai.azure.com`. Azure has been migrating endpoints. If you get `400 DeploymentError: invalid content filter policy`, try the services endpoint.

2. **Content filter policy.** Some deployments require an explicit content filter policy. If you see the error above and the endpoint is correct, check the deployment's content filter settings in the Azure AI Foundry portal.

3. **API version.** Both endpoints work with `api-version=2024-12-01-preview`.

4. **Deployment name.** Pass the exact deployment name (case-sensitive) as the `--model` argument. Example: `--model Kimi-K2.5`, not `--model kimi-k2.5`.

## Adding new reasoning models

To add support for a new reasoning model:

1. Add the deployment name (lowercase) to `REASONING_MODELS` in `tui/llm_backend.py`:
   ```python
   REASONING_MODELS = {"kimi-k2.5", "o3", "o3-mini", "o4-mini", "your-model"}
   ```

2. If the model uses OpenRouter instead of Azure, update `OpenRouterBackend.generate_experiment()` with the same `max_tokens` logic. Currently only `AzureOpenAIBackend` has reasoning model detection.

3. Run a quick smoke test:
   ```bash
   python -m tui.headless --model your-model --max 3 --tag test
   ```
   Check that all 3 experiments produce TSV rows. If you see "Unparseable response" in the output, the token budget may need increasing.

## Observed behavior differences

From 300 Kimi K2.5 experiments:

- **Crash rate is lower.** Kimi crashed 8.8% of the time vs 23% for Sonnet 4.6 and 29% for GPT-4.1. The chain-of-thought helps the model reason about `torch.compile` constraints before proposing code.

- **Strategy fixation is higher.** Kimi spent ~60% of proposals on architecture changes in all 3 runs and never attempted batch size reduction (the single highest-impact optimization). The chain-of-thought appears to create momentum toward whatever strategy the model starts exploring.

- **Run-to-run variance is high.** Kimi's R1 found 5 improvements, R2 found 16, R3 found 15. Standard models show more consistent behavior. The stochastic nature of early chain-of-thought reasoning creates path dependence.

- **Each run finds different strategies.** R1 discovered DropPath. R2 discovered narrow architectures (ASPECT_RATIO=32). R3 discovered focal loss. Standard models tend to converge on similar strategies across runs.
