from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Generic, Union, overload

from openai import AsyncOpenAI
from typing_extensions import TypeVar

from .exceptions import UserError
from .items import TResponseInputItem
from .run_context import TContext
from .util._types import MaybeAwaitable


@dataclass
class GuardrailFunctionOutput:
    """The output of a guardrail function."""

    output_info: Any
    """
    Optional information about the guardrail's output. For example, the guardrail could include
    information about the checks it performed and granular results.
    """

    tripwire_triggered: bool
    """
    Whether the tripwire was triggered. If triggered, the model's execution will be halted.
    """


@dataclass
class InputGuardrailResult:
    """The result of a guardrail run."""

    guardrail: InputGuardrail[Any]
    """
    The guardrail that was run.
    """

    output: GuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class OutputGuardrailResult:
    """The result of a guardrail run."""

    guardrail: OutputGuardrail[Any]
    """
    The guardrail that was run.
    """

    model_output: Any
    """
    The output of the model that was checked by the guardrail.
    """

    model: AsyncOpenAI
    """
    The model that was checked by the guardrail.
    """

    output: GuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class InputGuardrail(Generic[TContext]):
    """Input guardrails are checks that run in parallel to the model's execution.
    They can be used to do things like:
    - Check if input messages are off-topic

    You can use the `@input_guardrail()` decorator to turn a function into an `InputGuardrail`, or
    create an `InputGuardrail` manually.

    Guardrails return a `GuardrailResult`. If `result.tripwire_triggered` is `True`, the model
    execution will immediately stop and a `InputGuardrailTripwireTriggered` exception will be raised
    """

    guardrail_function: Callable[
        [TContext, AsyncOpenAI, str | list[TResponseInputItem]],
        MaybeAwaitable[GuardrailFunctionOutput],
    ]
    """A function that receives the model input and the context, and returns a
     `GuardrailResult`. The result marks whether the tripwire was triggered, and can optionally
     include information about the guardrail's output.
    """

    name: str | None = None
    """The name of the guardrail, used for tracing. If not provided, we'll use the guardrail
    function's name.
    """

    def get_name(self) -> str:
        if self.name:
            return self.name

        return self.guardrail_function.__name__

    async def run(
        self,
        model: AsyncOpenAI,
        input: str | list[TResponseInputItem],
        *,
        context: TContext,
    ) -> InputGuardrailResult:
        if not callable(self.guardrail_function):
            raise UserError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )

        output = self.guardrail_function(context, model, input)
        if inspect.isawaitable(output):
            output = await output

        return InputGuardrailResult(
            guardrail=self,
            output=output,
        )


@dataclass
class OutputGuardrail(Generic[TContext]):
    """Output guardrails are checks that run on the final output of a model.
    They can be used to do check if the output passes certain validation criteria

    You can use the `@output_guardrail()` decorator to turn a function into an `OutputGuardrail`,
    or create an `OutputGuardrail` manually.

    Guardrails return a `GuardrailResult`. If `result.tripwire_triggered` is `True`, a
    `OutputGuardrailTripwireTriggered` exception will be raised.
    """

    guardrail_function: Callable[
        [TContext, AsyncOpenAI, Any],
        MaybeAwaitable[GuardrailFunctionOutput],
    ]
    """A function that receives the final model, its output, and the context, and returns a
     `GuardrailResult`. The result marks whether the tripwire was triggered, and can optionally
     include information about the guardrail's output.
    """

    name: str | None = None
    """The name of the guardrail, used for tracing. If not provided, we'll use the guardrail
    function's name.
    """

    def get_name(self) -> str:
        if self.name:
            return self.name

        return self.guardrail_function.__name__

    async def run(
        self,
        model: AsyncOpenAI,
        model_output: Any,
        *,
        context: TContext,
    ) -> OutputGuardrailResult:
        if not callable(self.guardrail_function):
            raise UserError(
                f"Guardrail function must be callable, got {self.guardrail_function}"
            )

        output = self.guardrail_function(context, model, model_output)
        if inspect.isawaitable(output):
            output = await output

        return OutputGuardrailResult(
            guardrail=self,
            model=model,
            model_output=model_output,
            output=output,
        )


TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)

# For InputGuardrail
_InputGuardrailFuncSync = Callable[
    [
        TContext_co,
        AsyncOpenAI,
        Union[str, list[TResponseInputItem]],
    ],
    GuardrailFunctionOutput,
]
_InputGuardrailFuncAsync = Callable[
    [
        TContext_co,
        AsyncOpenAI,
        Union[str, list[TResponseInputItem]],
    ],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def input_guardrail(
    func: _InputGuardrailFuncSync[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    func: _InputGuardrailFuncAsync[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    *,
    name: str | None = None,
) -> Callable[
    [_InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co]],
    InputGuardrail[TContext_co],
]: ...


def input_guardrail(
    func: _InputGuardrailFuncSync[TContext_co]
    | _InputGuardrailFuncAsync[TContext_co]
    | None = None,
    *,
    name: str | None = None,
) -> (
    InputGuardrail[TContext_co]
    | Callable[
        [_InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co]],
        InputGuardrail[TContext_co],
    ]
):
    """
    Decorator that transforms a sync or async function into an `InputGuardrail`.
    It can be used directly (no parentheses) or with keyword args, e.g.:

        @input_guardrail
        def my_sync_guardrail(...): ...

        @input_guardrail(name="guardrail_name")
        async def my_async_guardrail(...): ...
    """

    def decorator(
        f: _InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co],
    ) -> InputGuardrail[TContext_co]:
        return InputGuardrail(guardrail_function=f, name=name)

    if func is not None:
        # Decorator was used without parentheses
        return decorator(func)

    # Decorator used with keyword arguments
    return decorator


_OutputGuardrailFuncSync = Callable[
    [TContext_co, AsyncOpenAI, Any],
    GuardrailFunctionOutput,
]
_OutputGuardrailFuncAsync = Callable[
    [TContext_co, AsyncOpenAI, Any],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def output_guardrail(
    func: _OutputGuardrailFuncSync[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    func: _OutputGuardrailFuncAsync[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    *,
    name: str | None = None,
) -> Callable[
    [_OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co]],
    OutputGuardrail[TContext_co],
]: ...


def output_guardrail(
    func: _OutputGuardrailFuncSync[TContext_co]
    | _OutputGuardrailFuncAsync[TContext_co]
    | None = None,
    *,
    name: str | None = None,
) -> (
    OutputGuardrail[TContext_co]
    | Callable[
        [
            _OutputGuardrailFuncSync[TContext_co]
            | _OutputGuardrailFuncAsync[TContext_co]
        ],
        OutputGuardrail[TContext_co],
    ]
):
    """
    Decorator that transforms a sync or async function into an `OutputGuardrail`.
    It can be used directly (no parentheses) or with keyword args, e.g.:

        @output_guardrail
        def my_sync_guardrail(...): ...

        @output_guardrail(name="guardrail_name")
        async def my_async_guardrail(...): ...
    """

    def decorator(
        f: _OutputGuardrailFuncSync[TContext_co]
        | _OutputGuardrailFuncAsync[TContext_co],
    ) -> OutputGuardrail[TContext_co]:
        return OutputGuardrail(guardrail_function=f, name=name)

    if func is not None:
        # Decorator was used without parentheses
        return decorator(func)

    # Decorator used with keyword arguments
    return decorator
