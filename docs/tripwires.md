# Tripwires

Tripwires are the core mechanism by which Guardrails enforce safety policies. When a guardrail detects a violation, it triggers a tripwire that blocks execution by default.

## How Tripwires Work

1. **Automatic Execution**: Guardrails run on every API call
2. **Tripwire Detection**: Violations trigger tripwires
3. **Default Behavior**: Tripwires raise `GuardrailTripwireTriggered` exceptions
4. **Custom Handling**: Suppress tripwires to handle violations manually

## Default Behavior: Blocking

Tripwires raise exceptions by default:

Python
```python
from pathlib import Path
from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered

client = GuardrailsAsyncOpenAI(config=Path("guardrails_config.json"))

try:
    response = await client.responses.create(
        model="gpt-5",
        input="Tell me a secret"
    )
    print(response.llm_response.output_text)
    
except GuardrailTripwireTriggered as exc:
    print(f"Guardrail triggered: {exc.guardrail_result.info}")
```


## Suppressing Tripwires

Handle violations manually with `suppress_tripwire=True`:

Python
```python
response = await client.responses.create(
    model="gpt-5",
    input="Tell me a secret",
    suppress_tripwire=True
)

# Check guardrail results
for result in response.guardrail_results.all_results:
    if result.tripwire_triggered:
        print(f"Guardrail '{result.info.get('guardrail_name')}' triggered!")
```


## Tripwire Results

The `GuardrailTripwireTriggered` exception contains:

- **`tripwire_triggered`** (bool): Always `True`
- **`info`** (dict): Guardrail-specific information

Python
```python
except GuardrailTripwireTriggered as exc:
    result = exc.guardrail_result
    guardrail_name = result.info.get('guardrail_name')
    stage = result.info.get('stage_name')
```


## Next Steps

- Learn about [streaming considerations](./streaming_output.md)
- Explore [examples](./examples.md) for usage patterns
