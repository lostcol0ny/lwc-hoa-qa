"""OpenTelemetry tracing for the pipeline, viewable in Langfuse.

Each question becomes one trace: a root ``ask`` span, a child per stage (gate,
cache, sweep, draft, verify), and under those one span per model request. The
spans are plain OpenTelemetry sent over OTLP/HTTP, so any OTLP backend can
receive them; the ``langfuse.*`` attributes are the ones Langfuse reads to
show inputs, outputs, model usage and cost
(langfuse.com/integrations/native/opentelemetry, read 2026-10-10).

Off unless ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set; when
off every call here is a no-op. Two levels of detail when on:

- **Numbers and codes** (always): scores, token counts, cost, timings,
  outcomes, chunk ids.
- **Text** (only with ``TRACE_CONTENT=1``): the question, the prompts, the
  passages sent to the answer model, the claims, the answer. This stores
  residents' questions with a third party, which the request log never does
  (docs/qa-core.md, "Logging and privacy"), so it is a separate switch.
"""

import asyncio
import base64
import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)

DEFAULT_LANGFUSE_URL = "https://cloud.langfuse.com"
OTLP_TRACES_PATH = "/api/public/otel/v1/traces"
FLUSH_TIMEOUT_MS = 2_000
# The Deploy workflow writes the deployed commit here (the function bundle
# includes it, like corpus.json), so traces can be compared across deploys.
RELEASE_FILE = Path("release.txt")
_RELEASE = re.compile(r"[0-9A-Za-z._-]{1,64}")

# Tags the caller adds to every trace started inside ``tagged(...)``.
_extra_tags: ContextVar[tuple[str, ...]] = ContextVar("extra_tags", default=())

# Langfuse observation types (langfuse.observation.type).
Kind = Literal["span", "generation", "chain", "retriever", "guardrail", "evaluator"]


@dataclass
class _State:
    provider: TracerProvider | None = None
    content: bool = False
    environment: str = "development"
    release: str | None = None


_state = _State()


def enabled() -> bool:
    return _state.provider is not None


def content_enabled() -> bool:
    return _state.provider is not None and _state.content


def use(
    provider: TracerProvider | None,
    *,
    content: bool = False,
    environment: str = "development",
    release: str | None = None,
) -> None:
    """Install (or, with None, remove) the provider. Tests call this directly."""
    _state.provider = provider
    _state.content = content
    _state.environment = environment
    _state.release = release


def langfuse_target(env: Mapping[str, str]) -> tuple[str, dict[str, str]] | None:
    """The OTLP endpoint and headers for Langfuse, or None if not configured."""
    public = env.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret = env.get("LANGFUSE_SECRET_KEY", "").strip()
    if not public or not secret:
        return None
    base = (
        env.get("LANGFUSE_BASE_URL", "").strip()
        or env.get("LANGFUSE_HOST", "").strip()
        or DEFAULT_LANGFUSE_URL
    ).rstrip("/")
    if not base.startswith("https://"):
        logger.error("LANGFUSE_BASE_URL must be an https:// URL; tracing stays off")
        return None
    auth = base64.b64encode(f"{public}:{secret}".encode()).decode("ascii")
    return base + OTLP_TRACES_PATH, {
        "Authorization": f"Basic {auth}",
        # Makes new data show up in Langfuse without the batch delay.
        "x-langfuse-ingestion-version": "4",
    }


def configure(env: Mapping[str, str]) -> bool:
    """Start exporting to Langfuse if its keys are set; return whether it did.

    Safe to call more than once: a second call keeps the first provider.
    """
    if enabled():
        return True
    target = langfuse_target(env)
    if target is None:
        return False
    # Imported here so an app with tracing off never loads the exporter.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    endpoint, headers = target
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=5)
    provider = TracerProvider(resource=Resource.create({"service.name": "hoa-qa"}))
    provider.add_span_processor(BatchSpanProcessor(exporter, max_queue_size=4096))
    use(
        provider,
        content=env.get("TRACE_CONTENT", "").strip() == "1",
        environment=(
            env.get("TRACE_ENVIRONMENT", "").strip()
            or env.get("VERCEL_ENV", "").strip()
            or "development"
        ),
        release=release(env),
    )
    logger.info(
        "tracing on environment=%s content=%s", _state.environment, _state.content
    )
    return True


def release(env: Mapping[str, str], path: Path = RELEASE_FILE) -> str | None:
    """Which build this is: ``RELEASE``, else release.txt, else Vercel's commit.

    Anything that is not a short plain identifier is ignored, so a stray file
    can't put arbitrary text on every trace.
    """
    candidates = [env.get("RELEASE", "")]
    try:
        candidates.append(path.read_text(encoding="utf-8"))
    except OSError:
        pass
    candidates.append(env.get("VERCEL_GIT_COMMIT_SHA", ""))
    for candidate in candidates:
        value = candidate.strip()
        if _RELEASE.fullmatch(value):
            return value
    return None


@contextmanager
def tagged(*tags: str) -> Iterator[None]:
    """Add ``tags`` to every trace started inside the block (and its tasks)."""
    token = _extra_tags.set((*_extra_tags.get(), *tags))
    try:
        yield
    finally:
        _extra_tags.reset(token)


def flush(timeout_ms: int = FLUSH_TIMEOUT_MS) -> None:
    """Send what is queued. Never raises: tracing must not fail a request."""
    provider = _state.provider
    if provider is None:
        return
    try:
        provider.force_flush(timeout_ms)
    except Exception as exc:
        logger.error("trace flush failed error_type=%s", type(exc).__name__)


async def flush_async(timeout_ms: int = FLUSH_TIMEOUT_MS) -> None:
    """``flush`` off the event loop. A serverless function is frozen once it
    responds, so the web app calls this before returning each answer."""
    if enabled():
        await asyncio.to_thread(flush, timeout_ms)


def _attr(value: Any) -> str | bool | int | float:
    if isinstance(value, str | bool | int | float):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


class Observation:
    """One span. Every method is a no-op when tracing is off."""

    def __init__(self, span: Span | None) -> None:
        self._span = span

    def set(self, **values: Any) -> None:
        """Numbers and codes: always recorded. Never pass text here."""
        if self._span is None:
            return
        for key, value in values.items():
            if value is not None:
                self._span.set_attribute(
                    f"langfuse.observation.metadata.{key}", _attr(value)
                )

    def input(self, value: Any) -> None:
        """Text going into this step: recorded only with TRACE_CONTENT=1."""
        if self._span is not None and _state.content:
            self._span.set_attribute("langfuse.observation.input", _attr(value))

    def output(self, value: Any, *, text: bool = True) -> None:
        """What came out. ``text=False`` marks a value that holds no text
        (scores, ids), which is recorded either way."""
        if self._span is not None and (_state.content or not text):
            self._span.set_attribute("langfuse.observation.output", _attr(value))

    def model(
        self,
        name: str,
        *,
        input_tokens: int,
        output_tokens: int = 0,
        input_usd: float = 0.0,
        output_usd: float = 0.0,
    ) -> None:
        """Model, token usage and cost, in the attributes Langfuse reads."""
        if self._span is None:
            return
        self._span.set_attribute("gen_ai.request.model", name)
        self._span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        self._span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        self._span.set_attribute(
            "langfuse.observation.cost_details",
            json.dumps(
                {
                    "input": input_usd,
                    "output": output_usd,
                    "total": input_usd + output_usd,
                }
            ),
        )

    def trace(self, *, name: str, tags: Sequence[str], **metadata: Any) -> None:
        """Trace-level fields, set on the root span.

        ``tags`` are a small fixed vocabulary to filter and group by;
        ``metadata`` is for values with many possibilities (an entry id, a
        model name). Both are numbers and codes, never text.
        """
        if self._span is None:
            return
        self._span.set_attribute("langfuse.trace.name", name)
        self._span.set_attribute(
            "langfuse.trace.tags", list(dict.fromkeys([*tags, *_extra_tags.get()]))
        )
        for key, value in metadata.items():
            if value is not None:
                self._span.set_attribute(f"langfuse.trace.metadata.{key}", _attr(value))
        if _state.release:
            self._span.set_attribute("langfuse.release", _state.release)

    def error(self, error_type: str) -> None:
        """Mark the step failed, by exception type only (messages can quote
        request bodies)."""
        if self._span is not None:
            self._span.set_status(Status(StatusCode.ERROR, error_type))


_OFF = Observation(None)


@contextmanager
def span(name: str, kind: Kind = "span") -> Iterator[Observation]:
    """Open a span as a child of the current one. Yields a no-op when off."""
    provider = _state.provider
    if provider is None:
        yield _OFF
        return
    tracer = provider.get_tracer("hoa_qa")
    # Exceptions are recorded by type only (see Observation.error), so the
    # SDK's own recording, which stores the message and traceback, is off.
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as raw:
        raw.set_attribute("langfuse.observation.type", kind)
        raw.set_attribute("langfuse.environment", _state.environment)
        observation = Observation(raw)
        try:
            yield observation
        except BaseException as exc:
            observation.error(type(exc).__name__)
            raise
