# Streaming vs Blocking

Guardrails supports two approaches for handling LLM output: non-streaming (safe, default) and streaming (fast). The choice balances safety vs. speed.

## Non-Streaming: Safe and Reliable (Default)

![Safe Pipeline](assets/images/slow_pipeline.png)

Default behavior (`stream=False`):

- **All guardrails complete** before showing output
- **Complete safety** - no unsafe content exposure
- **Higher latency** - user waits for full validation

**Best for**: High-assurance, compliance-critical scenarios

```python
response = await client.responses.create(
    model="gpt-5",
    input="Your input",
    stream=False  # Safe and reliable (default)
)
```

## Streaming: Fast but Less Safe

![Fast Pipeline](assets/images/fast_pipeline.png)

Set `stream=True` for real-time output:

- **Pre-flight & Input guardrails** run first
- **LLM output streams** to user immediately
- **Output guardrails** run in parallel with streaming
- **Risk**: Violative content may briefly appear before guardrails trigger

**Best for**: Low-risk, latency-sensitive applications

```python
response = await client.responses.create(
    model="gpt-5",
    input="Your input",
    stream=True  # Fast but some risk
)
```

## Implementation Examples

See complete examples:

- [Non-streaming (blocking)](https://github.com/openai/openai-guardrails-python/tree/main/examples/implementation_code/blocking)
- [Streaming](https://github.com/openai/openai-guardrails-python/tree/main/examples/implementation_code/streaming)