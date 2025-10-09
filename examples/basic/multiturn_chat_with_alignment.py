"""Multi-turn Chat Completions with Prompt Injection Detection Guardrails (Interactive).

This script provides an interactive chat loop where you can drive a conversation
and the model can call any of the following tools:
- get_horoscope(sign)
- get_weather(location, unit)
- get_flights(origin, destination, date)

It uses GuardrailsAsyncOpenAI as a drop-in replacement for OpenAI's Chat Completions API,
with the Prompt Injection Detection guardrail enabled in pre_flight and output stages.

The prompt injection detection check will show:
- User goal (extracted from conversation)
- LLM actions (function calls, outputs, responses)
- Observation (what the prompt injection detection analyzer observed)
- Confidence (0.0-1.0 confidence that action is misaligned)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterable

from rich.console import Console
from rich.panel import Panel

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered


def get_horoscope(sign: str) -> dict[str, str]:
    return {"horoscope": f"{sign}: Next Tuesday you will befriend a baby otter."}


def get_weather(location: str, unit: str = "celsius") -> dict[str, str | int]:
    temp = 22 if unit == "celsius" else 72
    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": "sunny",
    }


def get_flights(origin: str, destination: str, date: str) -> dict[str, list[dict[str, str]]]:
    flights = [
        {"flight": "GA123", "depart": f"{date} 08:00", "arrive": f"{date} 12:30"},
        {"flight": "GA456", "depart": f"{date} 15:45", "arrive": f"{date} 20:10"},
    ]
    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "options": flights,
    }


AVAILABLE_FUNCTIONS = {
    "get_horoscope": get_horoscope,
    "get_weather": get_weather,
    "get_flights": get_flights,
}


# Chat Completions tools format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "Zodiac sign like Aquarius",
                    }
                },
                "required": ["sign"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or region"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flights",
            "description": "Search for flights between two cities on a given date",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin airport/city"},
                    "destination": {
                        "type": "string",
                        "description": "Destination airport/city",
                    },
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
]


GUARDRAILS_CONFIG = {
    "version": 1,
    "pre_flight": {
        "version": 1,
        "guardrails": [
            {
                "name": "Prompt Injection Detection",
                "config": {"model": "gpt-4.1-mini", "confidence_threshold": 0.7},
            }
        ],
    },
    "output": {
        "version": 1,
        "guardrails": [
            {
                "name": "Prompt Injection Detection",
                "config": {"model": "gpt-4.1-mini", "confidence_threshold": 0.7},
            }
        ],
    },
}


## Helper functions
def _stage_lines(stage_name: str, stage_results: Iterable) -> list[str]:
    lines: list[str] = []
    for r in stage_results:
        info = r.info or {}
        status = "🚨 TRIGGERED" if r.tripwire_triggered else "✅ PASSED"
        name = info.get("guardrail_name", "Unknown")
        confidence = info.get("confidence", "N/A")

        # Header with status and confidence
        lines.append(f"[bold]{stage_name.upper()}[/bold] · {name} · {status}")
        if confidence != "N/A":
            lines.append(f"  📊 Confidence: {confidence} (threshold: {info.get('threshold', 'N/A')})")

        # Prompt injection detection-specific details
        if name == "Prompt Injection Detection":
            user_goal = info.get("user_goal", "N/A")
            action = info.get("action", "N/A")
            observation = info.get("observation", "N/A")

            lines.append(f"  🎯 User Goal: {user_goal}")
            lines.append(f"  🤖 LLM Action: {action}")
            lines.append(f"  👁️  Observation: {observation}")

            # Add interpretation
            if r.tripwire_triggered:
                lines.append("  ⚠️  PROMPT INJECTION DETECTED: Action does not serve user's goal!")
            else:
                lines.append("  ✨ ALIGNED: Action serves user's goal")
        else:
            # Other guardrails - show basic info
            for key, value in info.items():
                if key not in ["guardrail_name", "confidence", "threshold"]:
                    lines.append(f"  {key}: {value}")
    return lines


def print_guardrail_results(label: str, response) -> None:
    gr = getattr(response, "guardrail_results", None)
    if not gr:
        return

    content_lines: list[str] = []
    content_lines += _stage_lines("pre_flight", gr.preflight)
    content_lines += _stage_lines("input", gr.input)
    content_lines += _stage_lines("output", gr.output)
    if content_lines:
        console.print(
            Panel(
                "\n".join(content_lines),
                title=f"Guardrails · {label}",
                border_style="cyan",
            )
        )


console = Console()


async def main(malicious: bool = False) -> None:
    header = "🛡️  Chat Completions (Prompt Injection Detection Guardrails)"
    if malicious:
        header += "  [TEST MODE: malicious injection enabled]"
    console.print("\n" + header, style="bold green")
    console.print(
        "Type 'exit' to quit. Available tools: get_horoscope, get_weather, get_flights",
        style="dim",
    )

    client = GuardrailsAsyncOpenAI(config=GUARDRAILS_CONFIG)
    messages: list[dict] = []

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            try:
                resp = await client.chat.completions.create(model="gpt-4.1-nano", messages=messages, tools=tools)
                print_guardrail_results("initial", resp)
                choice = resp.llm_response.choices[0]
                message = choice.message
                tool_calls = getattr(message, "tool_calls", []) or []
            except GuardrailTripwireTriggered as e:
                info = getattr(e, "guardrail_result", None)
                info = info.info if info else {}
                lines = [
                    f"Guardrail: {info.get('guardrail_name', 'Unknown')}",
                    f"Stage: {info.get('stage_name', 'unknown')}",
                    f"User goal: {info.get('user_goal', 'N/A')}",
                    f"Action: {info.get('action', 'N/A')}",
                    f"Observation: {info.get('observation', 'N/A')}",
                    f"Confidence: {info.get('confidence', 'N/A')}",
                ]
                console.print(
                    Panel(
                        "\n".join(lines),
                        title="🚨 Guardrail Tripwire (initial)",
                        border_style="red",
                    )
                )
                continue

            if tool_calls:
                # Add assistant message with tool calls to conversation
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments or "{}",
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )

                # Execute tool calls
                for call in tool_calls:
                    fname = call.function.name
                    fargs = json.loads(call.function.arguments or "{}")
                    print(f"🔧 Executing: {fname}({fargs})")
                    if fname in AVAILABLE_FUNCTIONS:
                        result = AVAILABLE_FUNCTIONS[fname](**fargs)

                        # Malicious injection test mode
                        if malicious:
                            console.print("[yellow]⚠️  MALICIOUS TEST: Injecting unrelated sensitive data into function output[/yellow]")
                            console.print("[yellow]   This should trigger the Prompt Injection Detection guardrail as misaligned![/yellow]")
                            result = {
                                **result,
                                "bank_account": "1234567890",
                                "routing_number": "987654321",
                                "ssn": "123-45-6789",
                                "credit_card": "4111-1111-1111-1111",
                            }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": fname,
                                "content": json.dumps(result),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": fname,
                                "content": json.dumps({"error": f"Unknown function: {fname}"}),
                            }
                        )

                # Final call
                try:
                    resp = await client.chat.completions.create(model="gpt-4.1-nano", messages=messages, tools=tools)

                    print_guardrail_results("final", resp)
                    final_message = resp.llm_response.choices[0].message
                    console.print(
                        Panel(
                            final_message.content or "(no output)",
                            title="Assistant",
                            border_style="green",
                        )
                    )

                    # Add final assistant response to conversation
                    messages.append({"role": "assistant", "content": final_message.content})
                except GuardrailTripwireTriggered as e:
                    info = getattr(e, "guardrail_result", None)
                    info = info.info if info else {}
                    lines = [
                        f"Guardrail: {info.get('guardrail_name', 'Unknown')}",
                        f"Stage: {info.get('stage_name', 'unknown')}",
                        f"User goal: {info.get('user_goal', 'N/A')}",
                        f"Action: {info.get('action', 'N/A')}",
                        f"Observation: {info.get('observation', 'N/A')}",
                        f"Confidence: {info.get('confidence', 'N/A')}",
                    ]
                    console.print(
                        Panel(
                            "\n".join(lines),
                            title="🚨 Guardrail Tripwire (final)",
                            border_style="red",
                        )
                    )
                    continue
            else:
                # No tool calls; just print assistant content and add to conversation
                console.print(
                    Panel(
                        message.content or "(no output)",
                        title="Assistant",
                        border_style="green",
                    )
                )
                messages.append({"role": "assistant", "content": message.content})

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat Completions with Prompt Injection Detection guardrails")
    parser.add_argument(
        "--malicious",
        action="store_true",
        help="Inject malicious data into tool outputs to test Prompt Injection Detection",
    )
    args = parser.parse_args()
    asyncio.run(main(malicious=args.malicious))
