# OpenAI Guardrails: Python (Preview)

This is the Python version of OpenAI Guardrails, a package for adding configurable safety and compliance guardrails to LLM applications. It provides a drop-in wrapper for OpenAI's Python client, enabling automatic input/output validation and moderation using a wide range of guardrails.

Most users can simply follow the guided configuration and installation instructions at [guardrails.openai.com](https://guardrails.openai.com/).

## Installation

You can download [openai-guardrails package](https://pypi.org/project/openai-guardrails/) this way:

```bash
pip install openai-guardrails
```

### Usage

Follow the configuration and installation instructions at [guardrails.openai.com](https://guardrails.openai.com/).

### Local Development

Clone the repository and install locally:

```bash
# Clone the repository
git clone https://github.com/openai/openai-guardrails-python.git
cd openai-guardrails-python

# Install the package (editable), plus example extras if desired
pip install -e .
pip install -e ".[examples]"
```

## Integration Details

### Drop-in OpenAI Replacement

The easiest way to use Guardrails Python is as a drop-in replacement for the OpenAI client:

```python
from pathlib import Path
from guardrails import GuardrailsOpenAI, GuardrailTripwireTriggered

# Use GuardrailsOpenAI instead of OpenAI
client = GuardrailsOpenAI(config=Path("guardrail_config.json"))

try:
    # Works with standard Chat Completions
    chat = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Hello world"}],
    )
    print(chat.llm_response.choices[0].message.content)

    # Or with the Responses API
    resp = client.responses.create(
        model="gpt-5",
        input="What are the main features of your premium plan?",
    )
    print(resp.llm_response.output_text)
except GuardrailTripwireTriggered as e:
    print(f"Guardrail triggered: {e}")
```

### Agents SDK Integration

You can integrate guardrails with the OpenAI Agents SDK via `GuardrailAgent`:

```python
import asyncio
from pathlib import Path
from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, Runner
from agents.run import RunConfig
from guardrails import GuardrailAgent

# Create agent with guardrails automatically configured
agent = GuardrailAgent(
    config=Path("guardrails_config.json"),
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
)

async def main():
    try:
        result = await Runner.run(agent, "Hello, can you help me?", run_config=RunConfig(tracing_disabled=True))
        print(result.final_output)
    except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered):
        print("🛑 Guardrail triggered!")

if __name__ == "__main__":
    asyncio.run(main())
```

> For more details, see [`docs/agents_sdk_integration.md`](./docs/agents_sdk_integration.md).

## Evaluation Framework

Evaluate guardrail performance on labeled datasets and run benchmarks.

### Running Evaluations

```bash
# Basic evaluation
python -m guardrails.evals.guardrail_evals \
  --config-path guardrails_config.json \
  --dataset-path data.jsonl

# Benchmark mode (compare models, generate ROC curves, latency)
python -m guardrails.evals.guardrail_evals \
  --config-path guardrails_config.json \
  --dataset-path data.jsonl \
  --mode benchmark \
  --models gpt-5 gpt-5-mini gpt-4.1-mini
```

### Dataset Format

Datasets must be in JSONL format, with each line containing a JSON object:

```json
{
  "id": "sample_1",
  "data": "Text or conversation to evaluate",
  "expected_triggers": {
    "Moderation": true,
    "NSFW Text": false
  }
}
```

### Programmatic Usage

```python
from pathlib import Path
from guardrails.evals.guardrail_evals import GuardrailEval

eval = GuardrailEval(
    config_path=Path("guardrails_config.json"),
    dataset_path=Path("data.jsonl"),
    batch_size=32,
    output_dir=Path("results"),
)

import asyncio
asyncio.run(eval.run())
```

### Project Structure

- `src/guardrails/` - Python source code
- `src/guardrails/checks/` - Built-in guardrail checks
- `src/guardrails/evals/` - Evaluation framework
- `examples/` - Example usage and sample configs

## Examples

The package includes examples in the [`examples/` directory](./examples):

- `examples/basic/hello_world.py` — Basic chatbot with guardrails using `GuardrailsOpenAI`
- `examples/basic/agents_sdk.py` — Agents SDK integration with `GuardrailAgent`
- `examples/basic/local_model.py` — Using local models with guardrails
- `examples/basic/structured_outputs_example.py` — Structured outputs
- `examples/basic/pii_mask_example.py` — PII masking
- `examples/basic/suppress_tripwire.py` — Handling violations gracefully

### Running Examples

#### Prerequisites

```bash
pip install -e .
pip install "openai-guardrails[examples]"
```

#### Run

```bash
python examples/basic/hello_world.py
python examples/basic/agents_sdk.py
```

## Available Guardrails

The Python implementation includes the following built-in guardrails:

- **Moderation**: Content moderation using OpenAI's moderation API
- **URL Filter**: URL filtering and domain allowlist/blocklist
- **Contains PII**: Personally Identifiable Information detection
- **Hallucination Detection**: Detects hallucinated content using vector stores
- **Jailbreak**: Detects jailbreak attempts
- **NSFW Text**: Detects workplace-inappropriate content in model outputs
- **Off Topic Prompts**: Ensures responses stay within business scope
- **Custom Prompt Check**: Custom LLM-based guardrails

For full details, advanced usage, and API reference, see: [OpenAI Guardrails Documentation](https://openai.github.io/openai-guardrails-python/).

## License

MIT License - see LICENSE file for details.

## Disclaimers

Please note that Guardrails may use Third-Party Services such as the [Presidio open-source framework](https://github.com/microsoft/presidio), which are subject to their own terms and conditions and are not developed or verified by OpenAI.

Developers are responsible for implementing appropriate safeguards to prevent storage or misuse of sensitive or prohibited content (including but not limited to personal data, child sexual abuse material, or other illegal content). OpenAI disclaims liability for any logging or retention of such content by developers. Developers must ensure their systems comply with all applicable data protection and content safety laws, and should avoid persisting any blocked content generated or intercepted by Guardrails. Guardrails calls paid OpenAI APIs, and developers are responsible for associated charges.