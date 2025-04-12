from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .guardrail import InputGuardrailResult, OutputGuardrailResult


class GuardrailException(Exception):
    """Base class for all guardrail exceptions."""


class UserError(GuardrailException):
    """Exception raised when the user makes an error using the SDK."""

    message: str

    def __init__(self, message: str):
        self.message = message


class ModelBehaviorError(GuardrailException):
    """Exception raised when the model does something unexpected, e.g. calling a tool that doesn't
    exist, or providing malformed JSON.
    """

    message: str

    def __init__(self, message: str):
        self.message = message


class InputGuardrailTripwireTriggered(GuardrailException):
    """Exception raised when a guardrail tripwire is triggered."""

    guardrail_result: "InputGuardrailResult"
    """The result data of the guardrail that was triggered."""

    def __init__(self, guardrail_result: "InputGuardrailResult"):
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )


class OutputGuardrailTripwireTriggered(GuardrailException):
    """Exception raised when a guardrail tripwire is triggered."""

    guardrail_result: "OutputGuardrailResult"
    """The result data of the guardrail that was triggered."""

    def __init__(self, guardrail_result: "OutputGuardrailResult"):
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )
