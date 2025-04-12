from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseStreamEvent,
)

TResponse = Response
"""A type alias for the Response type from the OpenAI SDK."""

TResponseInputItem = ResponseInputItemParam
"""A type alias for the ResponseInputItemParam type from the OpenAI SDK."""

TResponseOutputItem = ResponseOutputItem
"""A type alias for the ResponseOutputItem type from the OpenAI SDK."""

TResponseStreamEvent = ResponseStreamEvent
"""A type alias for the ResponseStreamEvent type from the OpenAI SDK."""
