"""Helpers for loading configuration bundles and running guardrails.

This module is the bridge between configuration and runtime execution for
guardrail validation. It provides pure helpers for loading config bundles,
instantiating guardrails from registry specs, and orchestrating parallel
execution of guardrail checks. All logic is pure and side-effect-free
except for I/O during config reading.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final, Generic, ParamSpec, TypeAlias, TypeVar, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from ._openai_utils import prepare_openai_kwargs
from .exceptions import ConfigError, GuardrailTripwireTriggered
from .registry import GuardrailRegistry, default_spec_registry
from .spec import GuardrailSpec
from .types import GuardrailResult, MaybeAwaitableResult, TCfg, TContext, TIn
from .utils.context import validate_guardrail_context

logger = logging.getLogger(__name__)

P = ParamSpec("P")


@dataclass(frozen=True, slots=True)
class ConfiguredGuardrail(Generic[TContext, TIn, TCfg]):
    """A configured, executable guardrail.

    This class binds a `GuardrailSpec` definition to a validated configuration
    object. The resulting instance is used to run guardrail logic in production
    pipelines. It supports both sync and async check functions.

    Attributes:
        definition (GuardrailSpec[TContext, TIn, TCfg]): The immutable guardrail specification.
        config (TCfg): Validated user configuration for this instance.
    """

    definition: GuardrailSpec[TContext, TIn, TCfg]
    config: TCfg

    async def _ensure_async(
        self,
        fn: Callable[P, MaybeAwaitableResult],
        *a: P.args,
        **kw: P.kwargs,
    ) -> GuardrailResult:
        """Ensure a guardrail function is executed asynchronously.

        If the function is sync, runs it in a thread for compatibility with async flows.
        If already async, simply awaits it. Used internally to normalize execution style.

        Args:
            fn: Guardrail check function (sync or async).
            *a: Positional arguments for the check function.
            **kw: Keyword arguments for the check function.

        Returns:
            GuardrailResult: The result of the check function.
        """
        result = fn(*a, **kw)
        if inspect.isawaitable(result):
            return await result
        return cast("GuardrailResult", await asyncio.to_thread(lambda: result))

    async def run(self, ctx: TContext, data: TIn) -> GuardrailResult:
        """Run the guardrail's check function with the provided context and data.

        Main entry point for executing guardrails. Supports both sync and async
        functions, ensuring results are always awaited.

        Args:
            ctx (TContext): Runtime context for the guardrail.
            data (TIn): Input value to be checked.

        Returns:
            GuardrailResult: The outcome of the guardrail logic.
        """
        return await self._ensure_async(self.definition.check_fn, ctx, data, self.config)


class GuardrailConfig(BaseModel):
    """Configuration for a single guardrail instance.

    Used for serializing, deserializing, and validating guardrail configs as part
    of a bundle.

    Attributes:
        name (str): The registry name used to look up the guardrail spec.
        config (dict[str, Any]): Raw user configuration for this guardrail.
    """

    name: str
    config: dict[str, Any]


class ConfigBundle(BaseModel):
    """Versioned collection of configured guardrails.

    Represents a serializable "bundle" of guardrails to be run as a unit.
    Suitable for JSON storage and loading.

    Attributes:
        guardrails (list[GuardrailConfig]): The configured guardrails.
        version (int): Format version for forward/backward compatibility.
        stage_name (str): User-defined name for the pipeline stage this bundle is for.
            This can be any string that helps identify which part of your pipeline
            triggered the guardrail (e.g., "user_input_validation", "content_generation",
            "pre_processing", etc.). It will be included in GuardrailResult info for
            easy identification.
        config (dict[str, Any]): Execution configuration for this bundle.
            Optional fields include:
            - concurrency (int): Maximum number of guardrails to run in parallel (default: 10)
            - suppress_tripwire (bool): If True, don't raise exceptions on tripwires (default: False)
    """

    guardrails: list[GuardrailConfig]
    version: int = 1
    stage_name: str = "unnamed"
    config: dict[str, Any] = {}

    model_config = ConfigDict(frozen=True, extra="forbid")


class PipelineBundles(BaseModel):
    """Three-stage collection of validated ConfigBundles for an LLM pipeline.

    This class groups together guardrail configurations for the three main
    pipeline stages:

    - pre_flight: Checks that run before the LLM request is issued.
    - input: Checks on the user's prompt (may run concurrently with LLM call).
    - output: Checks on incremental LLM output.

    At least one stage must be provided, but all stages are optional.

    Attributes:
        pre_flight (ConfigBundle | None): Guardrails to run before the LLM request.
        input (ConfigBundle | None): Guardrails to run on user input.
        output (ConfigBundle | None): Guardrails to run on generated output.
        version (int): Schema version for the envelope itself. Defaults to 1.

    Example:
        ```python
        # All stages
        pipeline = PipelineBundles(
            pre_flight=load_config_bundle(PRE_FLIGHT),
            input=load_config_bundle(INPUT_BUNDLE),
            output=load_config_bundle(OUTPUT_BUNDLE),
        )

        # Just output stage
        pipeline = PipelineBundles(
            output=load_config_bundle(OUTPUT_BUNDLE),
        )

        # Print active stages
        for stage in pipeline.stages():
            print(stage.version)
        ```
    """

    pre_flight: ConfigBundle | None = None
    input: ConfigBundle | None = None
    output: ConfigBundle | None = None
    version: int = 1

    model_config = ConfigDict(frozen=True, extra="forbid")

    _STAGE_ORDER: Final[tuple[str, ...]] = ("pre_flight", "input", "output")

    def model_post_init(self, __context: Any) -> None:
        """Validate that at least one stage is provided."""
        if not any(getattr(self, stage) is not None for stage in self._STAGE_ORDER):
            raise ValueError("At least one stage (pre_flight, input, or output) must be provided")

    def stages(self) -> tuple[ConfigBundle, ...]:
        """Return non-None bundles in execution order (pre_flight → input → output)."""
        return tuple(bundle for name in self._STAGE_ORDER if (bundle := getattr(self, name)) is not None)


@dataclass(frozen=True, slots=True)
class JsonString:
    """Explicit wrapper to mark a string as a raw JSON config.

    Used to distinguish JSON string inputs from other config sources (path/dict).

    Attributes:
        content (str): The raw JSON string.
    """

    content: str


JsonPrimitive = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

ConfigSource: TypeAlias = ConfigBundle | Path | JsonValue | JsonString
PipelineSource: TypeAlias = PipelineBundles | Path | JsonValue | JsonString

T = TypeVar("T", bound=BaseModel)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _validate_from_dict(data: dict[str, Any], model: type[T]) -> T:
    return model.model_validate(data)


def _validate_from_json(text: str, model: type[T]) -> T:
    return model.model_validate_json(text)


def _load_bundle(source: ConfigSource | PipelineSource, model: type[T]) -> T:
    """Generic loader for `ConfigBundle` and `PipelineBundles`."""
    match source:
        case _ if isinstance(source, model):
            return cast(T, source)
        case dict():
            logger.debug("Validating %s from dict", model.__name__)
            return _validate_from_dict(source, model)
        case Path() as path:
            logger.debug("Validating %s from %s", model.__name__, path)
            return _validate_from_json(_read_text(path), model)
        case JsonString(content=text):
            logger.debug("Validating %s from JSON string", model.__name__)
            return _validate_from_json(text, model)
        case _:
            raise ConfigError(f"Unsupported source type for {model.__name__}: {type(source).__name__}. Wrap raw JSON strings with `JsonString`.")


def load_config_bundle(source: ConfigSource) -> ConfigBundle:
    """Load a ConfigBundle from a path, dict, JSON string, or already-parsed object.

    Supported sources:
      - ConfigBundle: Already validated bundle.
      - dict: Raw data, parsed as ConfigBundle.
      - Path: JSON file on disk.
      - JsonString: Raw JSON string.

    Example usage:
        ```python
        bundle = load_config_bundle(JsonString('{"guardrails": [...]}'))
        bundle = load_config_bundle({"guardrails": [...]})
        bundle = load_config_bundle(Path("./config.json"))
        ```

    Args:
        source (ConfigSource): Config bundle input.

    Raises:
        ConfigError: If loading fails.
        ValidationError: If model validation fails.
        FileNotFoundError: If file doesn't exist.

    Returns:
        ConfigBundle: The loaded and validated configuration bundle.
    """
    return _load_bundle(source, ConfigBundle)


def load_pipeline_bundles(source: PipelineSource) -> PipelineBundles:
    """Load a PipelineBundles from a path, dict, JSON string, or already-parsed object.

    Supported sources:
      - PipelineBundles: Already validated pipeline bundles.
      - dict: Raw data, parsed as PipelineBundles.
      - Path: JSON file on disk.
      - JsonString: Raw JSON string.

    Example usage:
        ```python
        bundle = load_pipeline_bundles(JsonString('{"pre_flight": {...}, ...}'))
        bundle = load_pipeline_bundles({"pre_flight": {...}, ...})
        bundle = load_pipeline_bundles(Path("./pipeline.json"))
        ```

    Args:
        source (PipelineSource): Pipeline bundles input.

    Raises:
        ConfigError: If loading fails.
        ValidationError: If model validation fails.
        FileNotFoundError: If file doesn't exist.

    Returns:
        PipelineBundles: The loaded and validated pipeline configuration bundle.
    """
    return _load_bundle(source, PipelineBundles)


def instantiate_guardrails(
    bundle: ConfigBundle,
    registry: GuardrailRegistry | None = None,
) -> list[ConfiguredGuardrail[Any, Any, Any]]:
    """Instantiate all configured guardrails in a bundle as executable objects.

    This function validates each guardrail configuration, retrieves the spec
    from the registry, and returns a list of fully configured guardrails.

    Args:
        bundle (ConfigBundle): The validated configuration bundle.
        registry (GuardrailRegistry, optional): Registry mapping names to specs.
            If not provided, defaults to `default_spec_registry`.

    Raises:
        ConfigError: If any individual guardrail config is invalid.

    Returns:
        list[ConfiguredGuardrail[Any, Any, Any]]: All configured/runnable guardrail objects.
    """
    registry = registry or default_spec_registry
    logger.debug("Instantiating guardrails using registry %s", registry)
    out: list[ConfiguredGuardrail[Any, Any, Any]] = []
    for item in bundle.guardrails:
        logger.debug("Configuring guardrail '%s'", item.name)
        grd_spec = registry.get(item.name)
        # Validate guardrail-specific config and wrap errors
        try:
            cfg = grd_spec.config_schema.model_validate(item.config)
        except Exception as e:
            logger.error("Guardrail '%s' config validation failed: %s", item.name, e)
            raise ConfigError(
                f"Configuration for guardrail '{item.name}' invalid: {e}",
            ) from e
        out.append(grd_spec.instantiate(config=cfg))
    logger.info("Instantiated %d guardrails", len(out))
    return out


async def run_guardrails(
    ctx: TContext,
    data: TIn,
    media_type: str,
    guardrails: Iterable[ConfiguredGuardrail[TContext, TIn, Any]],
    *,
    concurrency: int = 10,
    result_handler: (Callable[[GuardrailResult], Coroutine[None, None, None]] | None) = None,
    suppress_tripwire: bool = False,
    stage_name: str | None = None,
    raise_guardrail_errors: bool = False,
) -> list[GuardrailResult]:
    """Run a set of configured guardrails concurrently and collect their results.

    Validates context requirements for each guardrail, filters guardrails by the specified media type,
    and runs each check function concurrently, up to the specified concurrency limit. Results for all
    executed guardrails are collected and returned in order. If any guardrail triggers a tripwire,
    the function will raise a `GuardrailTripwireTriggered` exception unless tripwire suppression is enabled.

    An optional asynchronous result handler can be provided to perform side effects (e.g., logging,
    custom result processing) for each guardrail result as it becomes available.

    Args:
        ctx (TContext): Context object passed to all guardrail checks. Must satisfy all required
            fields specified by each guardrail's context schema.
        data (TIn): The input to be validated by the guardrails.
        media_type (str): MIME type used to filter which guardrails to execute. (e.g. "text/plain")
        guardrails (Iterable[ConfiguredGuardrail[TContext, TIn, Any]]): Iterable of configured
            guardrails to run.
        concurrency (int, optional): Maximum number of guardrails to run in parallel.
            Defaults to 10.
        result_handler (Callable[[GuardrailResult], Awaitable[None]], optional): Asynchronous
            callback function to be invoked for each guardrail result as it is produced.
            Defaults to None.
        suppress_tripwire (bool, optional): If True, tripwire-triggered results are included in
            the returned list and no exception is raised. If False (default), the function aborts
            and raises `GuardrailTripwireTriggered` on the first tripwire.
        stage_name (str | None, optional): Name of the pipeline stage (e.g., "pre_flight", "input", "output").
            If provided, this will be included in the GuardrailResult info. Defaults to None.
        raise_guardrail_errors (bool, optional): If True, raise exceptions when guardrails fail to execute.
            If False (default), treat guardrail execution errors as safe and continue execution.

    Returns:
        list[GuardrailResult]: List of results for all executed guardrails. If tripwire suppression
            is disabled (default) and a tripwire is triggered, only results up to the first tripwire
            may be included.

    Raises:
        GuardrailTripwireTriggered: Raised if a guardrail tripwire is triggered and
            `suppress_tripwire` is False.
        ContextValidationError: Raised if the provided context does not meet requirements for any
            guardrail being executed.

    Example:
        ```python
        results = await run_guardrails(
            ctx=my_ctx,
            data="example input",
            media_type="text/plain",
            guardrails=my_guardrails,
            concurrency=4,
            suppress_tripwire=True,
            stage_name="input",
        )
        ```
    """  # noqa: E501
    selected = [g for g in guardrails if g.definition.media_type == media_type]
    logger.debug(
        "Running %d guardrails for media-type '%s' with concurrency=%d",
        len(selected),
        media_type,
        concurrency,
    )
    if not selected:
        return []

    for g in selected:
        validate_guardrail_context(g, ctx)

    semaphore = asyncio.Semaphore(concurrency)
    results: list[GuardrailResult] = []
    tripwire_results: list[GuardrailResult] = []

    async def _run_one(
        g: ConfiguredGuardrail[TContext, TIn, Any],
    ) -> GuardrailResult:
        async with semaphore:
            logger.debug("Running guardrail '%s'", g.definition.name)
            try:
                result = await g.run(ctx, data)

                # Always add stage_name to the result info while preserving all fields
                result = GuardrailResult(
                    tripwire_triggered=result.tripwire_triggered,
                    execution_failed=result.execution_failed,
                    original_exception=result.original_exception,
                    info={**result.info, "stage_name": stage_name or "unnamed"},
                )

            except Exception as exc:
                logger.error("Guardrail '%s' failed to execute: %s", g.definition.name, exc)

                if raise_guardrail_errors:
                    # Re-raise the exception to stop execution
                    raise exc
                else:
                    # Create a safe result indicating execution error
                    result = GuardrailResult(
                        tripwire_triggered=False,  # Don't trigger tripwire on execution errors
                        execution_failed=True,
                        original_exception=exc,
                        info={
                            "checked_text": str(data),
                            "stage_name": stage_name or "unnamed",
                            "guardrail_name": g.definition.name,
                            "error": str(exc),
                        },
                    )

            # Invoke user-provided handler for each result
            if result_handler:
                try:
                    await result_handler(result)
                except Exception as handler_err:
                    logger.error(
                        "Error in result_handler for guardrail '%s': %s",
                        g.definition.name,
                        handler_err,
                    )
            if result.tripwire_triggered:
                logger.warning("Tripwire triggered by '%s'", g.definition.name)
            return result

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_run_one(g)) for g in selected]
    results.extend(task.result() for task in tasks)

    # Check for guardrail execution failures and re-raise if configured
    if raise_guardrail_errors:
        execution_failures = [r for r in results if r.execution_failed]

        if execution_failures:
            # Re-raise the first execution failure
            failure = execution_failures[0]
            logger.debug("Re-raising guardrail execution error due to raise_guardrail_errors=True")
            raise failure.original_exception

    tripwire_results = [r for r in results if r.tripwire_triggered]
    if tripwire_results and not suppress_tripwire:
        raise GuardrailTripwireTriggered(tripwire_results[0])

    logger.info("Completed guardrail run; %d results returned", len(results))
    return results


@cache
def _get_default_ctx():
    @dataclass
    class DefaultCtx:
        guardrail_llm: AsyncOpenAI

    return DefaultCtx(guardrail_llm=AsyncOpenAI(**prepare_openai_kwargs({})))


async def check_plain_text(
    text: str,
    bundle_path: ConfigSource = Path("guardrails.json"),
    registry: GuardrailRegistry | None = None,
    ctx: Any = None,
    **kwargs: Any,
) -> list[GuardrailResult]:
    """Validate plain text input against all configured guardrails for the 'text/plain' media type.

    This function loads a guardrail configuration bundle, instantiates all guardrails for the
    specified registry, and runs each guardrail against the provided text input. It is the
    recommended entry point for validating plain text with one or more guardrails in an async context.

    If no context object (`ctx`) is provided, a minimal default context will be constructed with
    attributes required by the guardrails' context schema (for example, an OpenAI client and any
    required fields with safe default values). For advanced use cases, you can supply your own
    context object.

    Keyword arguments are forwarded to `run_guardrails`, allowing you to control concurrency,
    provide a result handler, or suppress tripwire exceptions.

    Args:
        text (str): The plain text input to validate.
        bundle_path (ConfigSource, optional): Guardrail configuration bundle. This can be a file path,
            dict, JSON string, or `ConfigBundle` instance. Defaults to `"guardrails.json"`.
        registry (GuardrailRegistry, optional): Guardrail registry to use for instantiation.
            If not provided, the default registry is used.
        ctx (Any, optional): Application context object passed to each guardrail.
            If None (default), a minimal default context will be used.
        **kwargs: Additional keyword arguments forwarded to `run_guardrails`. Common options include:

            - concurrency (int): Maximum number of guardrails to run concurrently.
            - result_handler (Callable[[GuardrailResult], Awaitable[None]]): Async function invoked for each result.
            - suppress_tripwire (bool): If True, do not raise an exception on tripwire; instead, return all results.

    Returns:
        list[GuardrailResult]: Results from all executed guardrails.

    Raises:
        ConfigError: If the configuration bundle cannot be loaded or is invalid.
        ContextValidationError: If the context does not meet required fields for the guardrails.
        GuardrailTripwireTriggered: If a guardrail tripwire is triggered and `suppress_tripwire` is False.

    Example:
        ```python
        from guardrails import check_plain_text

        results = await check_plain_text(
            "some text",
            bundle_path="my_guardrails.json",
            concurrency=4,
            suppress_tripwire=True,
        )
        ```
    """  # noqa: E501
    if ctx is None:
        ctx = _get_default_ctx()
    bundle = load_config_bundle(bundle_path)
    guardrails: list[ConfiguredGuardrail[Any, str, Any]] = instantiate_guardrails(bundle, registry=registry)
    return await run_guardrails(ctx, text, "text/plain", guardrails, stage_name=bundle.stage_name, **kwargs)
