import asyncio
from typing import Any

from openai import AsyncOpenAI

from guardrails.guardrail import (
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
)

from .exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
from .items import TResponseInputItem
from .run_context import TContext


async def run_input_guardrails(
    model: AsyncOpenAI,
    guardrails: list[InputGuardrail[TContext]],
    input: str | list[TResponseInputItem],
    context: TContext,
) -> list[InputGuardrailResult]:
    if not guardrails:
        return []

    guardrail_tasks = [
        asyncio.create_task(guardrail.run(model, input, context=context))
        for guardrail in guardrails
    ]

    guardrail_results = []

    for done in asyncio.as_completed(guardrail_tasks):
        result = await done
        if result.output.tripwire_triggered:
            # Cancel all guardrail tasks if a tripwire is triggered.
            for t in guardrail_tasks:
                t.cancel()
            # TODO: (ovallis) Add logging here
            raise InputGuardrailTripwireTriggered(result)
        else:
            guardrail_results.append(result)

    return guardrail_results


async def run_output_guardrails(
    model: AsyncOpenAI,
    guardrails: list[OutputGuardrail[TContext]],
    model_output: Any,
    context: TContext,
) -> list[OutputGuardrailResult]:
    if not guardrails:
        return []

    guardrail_tasks = [
        asyncio.create_task(guardrail.run(model, model_output, context=context))
        for guardrail in guardrails
    ]

    guardrail_results = []

    for done in asyncio.as_completed(guardrail_tasks):
        result = await done
        if result.output.tripwire_triggered:
            # Cancel all guardrail tasks if a tripwire is triggered.
            for t in guardrail_tasks:
                t.cancel()
            # TODO: (ovallis) Add logging here
            raise OutputGuardrailTripwireTriggered(result)
        else:
            guardrail_results.append(result)

    return guardrail_results
