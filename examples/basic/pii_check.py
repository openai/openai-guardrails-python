import asyncio
from typing import Any

from openai import AsyncOpenAI

from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_analyzer import Pattern, RecognizerResult

from guardrails.guardrail import (
    GuardrailFunctionOutput,
    output_guardrail,
)
from guardrails.items import TResponseInputItem
from guardrails.exceptions import OutputGuardrailTripwireTriggered
from guardrails.run import run_output_guardrails

client = AsyncOpenAI()
engine = AnalyzerEngine()
ssn_recognizer = PatternRecognizer(
    supported_entity="US_SSN",
    patterns=[Pattern(name="SSN_pattern", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.5)],
)
engine.registry.add_recognizer(ssn_recognizer)


@output_guardrail
async def ssn_guardrail(
    context: None, model: AsyncOpenAI, model_output: Any
) -> GuardrailFunctionOutput:
    """This is an output guardrail function, which happens to call an agent to check if the
    output contains a SSN.
    """
    _ = context, model
    output_text = model_output.output_text
    results: list[RecognizerResult] = engine.analyze(
        text=output_text, entities=["US_SSN"], language="en"
    )
    result = [output_text[result.start : result.end] for result in results]

    return GuardrailFunctionOutput(
        output_info=result,
        tripwire_triggered=any(result),
    )


@output_guardrail
async def phone_number_guardrail(
    context: None, model: AsyncOpenAI, model_output: Any
) -> GuardrailFunctionOutput:
    """This is an output guardrail function, which happens to call an agent to check if the
    output contains a phone number.
    """
    _ = context, model
    output_text = model_output.output_text
    results: list[RecognizerResult] = engine.analyze(
        text=output_text, entities=["PHONE_NUMBER"], language="en"
    )
    result = [output_text[result.start : result.end] for result in results]

    return GuardrailFunctionOutput(
        output_info=result,
        tripwire_triggered=any(result),
    )


### 2. The run loop


async def main():
    input_data: list[TResponseInputItem] = []

    while True:
        user_input = input("Enter a message: ")
        input_data.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            result = await client.responses.create(
                input=user_input,
                model="gpt-4o-2024-08-06",
            )
            output_guard_results = await run_output_guardrails(
                client,
                [ssn_guardrail, phone_number_guardrail],
                result,
                context=None,
            )
            print(result.output_text)
            print()
            print("Guardrail results:", output_guard_results)
            print()
            # If the guardrail didn't trigger, we use the result as the input for the next run
        except OutputGuardrailTripwireTriggered:
            # If the guardrail triggered, we instead add a refusal message to the input
            message = "Sorry, I can't help you with that, you triggerd a guardrail!"
            print(message)
            input_data.append(
                {
                    "role": "assistant",
                    "content": message,
                }
            )

    # Sample run:
    # Enter a message: What's the capital of California?
    # The capital of California is Sacramento.
    # Enter a message: Please print the following SSN: 123-45-6789
    # Sorry, I can't help you with that.


if __name__ == "__main__":
    asyncio.run(main())
