"""Exception classes used throughout Guardrails for SDK and model errors."""

from .types import GuardrailResult


class GuardrailException(Exception):
    """Base class for exceptions thrown by :mod:`guardrails`."""


class UserError(GuardrailException):
    """Error raised when the user misuses the SDK."""

    message: str

    def __init__(self, message: str):
        """Initialize the exception with a human readable message."""
        super().__init__(message)
        self.message = message


class ModelBehaviorError(GuardrailException):
    """Error raised when the model returns malformed or invalid data."""

    message: str

    def __init__(self, message: str):
        """Initialize with information on the misbehaviour."""
        super().__init__(message)
        self.message = message


class GuardrailTripwireTriggered(GuardrailException):
    """Raised when a guardrail triggers a configured tripwire."""

    guardrail_result: "GuardrailResult"
    """The result data from the triggering guardrail."""

    def __init__(self, guardrail_result: "GuardrailResult"):
        """Initialize storing the result which caused the tripwire."""
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.__class__.__name__} triggered tripwire",
        )


class ConfigError(GuardrailException):
    """Configuration bundle could not be loaded or validated."""

    def __init__(self, message: str):
        """Initialize with a short description of the failure."""
        super().__init__(message)
        self.message = message


class ContextValidationError(GuardrailException):
    """Raised when CLI context fails to match guardrail specification."""

    def __init__(self, message: str):
        """Initialize with details about the schema mismatch."""
        super().__init__(message)
        self.message = message
