# OpenAI Guardrails: Python (Preview)

Add configurable safety and compliance checks to LLM applications. OpenAI Guardrails wraps the OpenAI Python client to validate inputs and outputs, and integrates with the OpenAI Agents SDK.

[Configure guardrails](https://guardrails.openai.com/) · [Documentation](https://openai.github.io/openai-guardrails-python/) · [Examples](./examples)

[![OpenAI Guardrails configuration screenshot](https://raw.githubusercontent.com/openai/openai-guardrails-python/main/docs/assets/images/guardrails-python-config-screenshot-100pct-q70.webp)](https://guardrails.openai.com/)

## Installation

Requires **Python 3.11+**. Install [openai-guardrails](https://pypi.org/project/openai-guardrails/):

```bash
pip install openai-guardrails
```

If your configuration uses **Contains PII**, also install its spaCy model during build or deployment:

```bash
python -m spacy download en_core_web_sm
```

For uv with spaCy 3.8, install the model wheel directly:

```bash
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Contains PII validates and loads this model during client initialization; a missing or unloadable model fails configuration before any request. See the [Contains PII guide](https://openai.github.io/openai-guardrails-python/ref/checks/pii/) for configuration options.

## Quickstart

1. Create and export a pipeline configuration with the [Guardrails wizard](https://guardrails.openai.com/). Save it as `guardrails_config.json` in your working directory.
2. Set the `OPENAI_API_KEY` environment variable to your OpenAI API key.
3. Use `GuardrailsOpenAI` in place of `OpenAI`:

```python
from pathlib import Path

from guardrails import GuardrailsOpenAI, GuardrailTripwireTriggered

client = GuardrailsOpenAI(config=Path("guardrails_config.json"))

try:
    # Chat Completions API
    chat = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": "Hello world"}],
    )
    print(chat.choices[0].message.content)

    # Responses API
    response = client.responses.create(
        model="gpt-5",
        input="What are the main features of your premium plan?",
    )
    print(response.output_text)
except GuardrailTripwireTriggered:
    print("Message blocked by guardrails.")
```

For async and Azure clients, see the [quickstart guide](https://openai.github.io/openai-guardrails-python/quickstart/). Read about [tripwire handling](https://openai.github.io/openai-guardrails-python/tripwires/) and [streaming behavior](https://openai.github.io/openai-guardrails-python/streaming_output/) before integrating those flows.

### Agents SDK

Use `GuardrailAgent` with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). This example uses the same config file and API key as the quickstart:

```python
import asyncio
from pathlib import Path

from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, Runner
from agents.run import RunConfig

from guardrails import GuardrailAgent

agent = GuardrailAgent(
    config=Path("guardrails_config.json"),
    name="Customer support agent",
    instructions="You help customers with their questions.",
)


async def main():
    try:
        result = await Runner.run(
            agent, "Hello, can you help me?", run_config=RunConfig(tracing_disabled=True)
        )
        print(result.final_output)
    except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered):
        print("Message blocked by guardrails.")


if __name__ == "__main__":
    asyncio.run(main())
```

See the [Agents SDK integration guide](https://openai.github.io/openai-guardrails-python/agents_sdk_integration/) for configuration and tool guardrails.

## Evaluations

Measure guardrail performance on labeled datasets using the same exported configuration. Install the evaluation dependencies first (currently required for basic evaluation as well as benchmarking):

```bash
pip install "openai-guardrails[benchmark]"
python -m guardrails.evals.guardrail_evals \
  --config-path guardrails_config.json \
  --dataset-path data.jsonl
```

Save one JSON object per line in `data.jsonl`, with labels for the guardrails in your configuration. For a configuration containing Moderation and NSFW Text:

```json
{"id": "sample_1", "data": "Hello world", "expected_triggers": {"Moderation": false, "NSFW Text": false}}
```

For the programmatic API, model comparisons, benchmark dependencies, and dataset options, see the [evaluation guide](https://openai.github.io/openai-guardrails-python/evals/).

## Examples and Local Development

Clone the repository and install it with the example dependencies:

```bash
git clone https://github.com/openai/openai-guardrails-python.git
cd openai-guardrails-python
pip install -e ".[examples]"
```

Set `OPENAI_API_KEY` as above. Install the spaCy model from the installation section for examples that use Contains PII, including `agents_sdk.py`.

```bash
python examples/basic/hello_world.py
python examples/basic/agents_sdk.py
```

Explore the examples:

- [Basic chatbot](./examples/basic/hello_world.py) — async client with guardrails
- [Agents SDK](./examples/basic/agents_sdk.py) — agent with input and output checks
- [Local models](./examples/basic/local_model.py) — OpenAI-compatible model endpoints
- [Structured outputs](./examples/basic/structured_outputs_example.py)
- [PII masking](./examples/basic/pii_mask_example.py)
- [Tripwire suppression](./examples/basic/suppress_tripwire.py)

## Available Guardrails

| Guardrail | Checks for |
| --- | --- |
| [Keyword Filter](https://openai.github.io/openai-guardrails-python/ref/checks/keywords/) | Configured keywords and phrases |
| [Competitors](https://openai.github.io/openai-guardrails-python/ref/checks/competitors/) | Mentions of configured competitors |
| [Moderation](https://openai.github.io/openai-guardrails-python/ref/checks/moderation/) | Content flagged by OpenAI's moderation API |
| [URL Filter](https://openai.github.io/openai-guardrails-python/ref/checks/urls/) | URLs against domain allowlists or blocklists |
| [Secret Keys](https://openai.github.io/openai-guardrails-python/ref/checks/secret_keys/) | Potential API keys, secrets, and credentials |
| [Contains PII](https://openai.github.io/openai-guardrails-python/ref/checks/pii/) | Personally identifiable information |
| [Hallucination Detection](https://openai.github.io/openai-guardrails-python/ref/checks/hallucination_detection/) | Claims against reference material in vector stores |
| [Jailbreak](https://openai.github.io/openai-guardrails-python/ref/checks/jailbreak/) | Jailbreak attempts |
| [Prompt Injection Detection](https://openai.github.io/openai-guardrails-python/ref/checks/prompt_injection_detection/) | Misaligned tool calls and tool outputs |
| [NSFW Text](https://openai.github.io/openai-guardrails-python/ref/checks/nsfw/) | Workplace-inappropriate content |
| [Off Topic Prompts](https://openai.github.io/openai-guardrails-python/ref/checks/off_topic_prompts/) | Content outside a configured topic or scope |
| [Custom Prompt Check](https://openai.github.io/openai-guardrails-python/ref/checks/custom_prompt_check/) | Violations of custom instructions |

## License

[MIT](./LICENSE).

## Disclaimers

Guardrails may use Third-Party Services such as the [Presidio open-source framework](https://github.com/microsoft/presidio), which are subject to their own terms and conditions and are not developed or verified by OpenAI.

Developers are responsible for implementing appropriate safeguards to prevent storage or misuse of sensitive or prohibited content (including but not limited to personal data, child sexual abuse material, or other illegal content). OpenAI disclaims liability for any logging or retention of such content by developers. Developers must ensure their systems comply with all applicable data protection and content safety laws, and should avoid persisting any blocked content generated or intercepted by Guardrails. Guardrails calls paid OpenAI APIs, and developers are responsible for associated charges.
