"""
Synapse Core Generators and Lazy Streaming Engine.

Provides:
- GeneratorState: Lifecycle state enum (CREATED, SUSPENDED, RUNNING, CLOSED)
- SynapseGenerator: Full Python generator protocol wrapper with bi-directional send/throw/close
- StreamPipeline: Fluent lazy streaming pipeline (map, filter, take, skip, batch, collect, reduce)
- TokenStream: Specialized lazy stream for LLM / AI token streaming with chunk buffering,
               delta emission, and cross-chunk stop sequence handling.
"""

from __future__ import annotations

from enum import Enum
import functools
import inspect
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")  # Yield type
S = TypeVar("S")  # Send type
R = TypeVar("R")  # Return type
U = TypeVar("U")  # Mapped / reduced type

_SENTINEL = object()


class GeneratorState(str, Enum):
    """Lifecycle states for a SynapseGenerator."""
    CREATED = "CREATED"
    SUSPENDED = "SUSPENDED"
    RUNNING = "RUNNING"
    CLOSED = "CLOSED"


class SynapseGenerator(Generic[T, S, R]):
    """
    Synapse bi-directional generator and iterator protocol implementation.

    Supports:
    - __iter__() -> Iterator[T]
    - __next__() -> T
    - send(value: S) -> T
    - throw(typ: Type[BaseException], val=None, tb=None) -> T
    - close() -> None
    - state tracking (CREATED, SUSPENDED, RUNNING, CLOSED)
    - return value capture (self.value)
    - fluent conversion to StreamPipeline (.to_pipeline())
    """

    def __init__(
        self,
        gen: Optional[Union[Generator[T, S, R], Iterable[T], Callable[..., Any]]] = None,
    ):
        self._state: GeneratorState = GeneratorState.CREATED
        self._value: Optional[R] = None

        if gen is None:
            if hasattr(self, "generate") and callable(self.generate):
                res = self.generate()
                self._gen = res if hasattr(res, "__next__") else iter(res)
            else:
                self._gen = iter(())
        elif inspect.isgeneratorfunction(gen) or (
            callable(gen) and not hasattr(gen, "__next__") and not hasattr(gen, "__iter__")
        ):
            res = gen()
            self._gen = res if hasattr(res, "__next__") else iter(res)
        elif hasattr(gen, "__next__"):
            self._gen = gen  # type: ignore
        else:
            self._gen = iter(gen)  # type: ignore

    @property
    def state(self) -> GeneratorState:
        """Returns the current lifecycle state of the generator."""
        return self._state

    @property
    def value(self) -> Optional[R]:
        """Returns the return value of the generator if completed, else None."""
        return self._value

    @property
    def gi_running(self) -> bool:
        """Returns True if generator is actively executing."""
        return self._state == GeneratorState.RUNNING

    @property
    def gi_frame(self) -> Any:
        """Underlying execution frame if available."""
        return getattr(self._gen, "gi_frame", None)

    @property
    def gi_code(self) -> Any:
        """Underlying code object if available."""
        return getattr(self._gen, "gi_code", None)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        return self.send(None)  # type: ignore

    def send(self, value: S) -> T:
        """
        Send a value into the generator.
        Returns the next value yielded by the generator, or raises StopIteration.
        """
        if self._state == GeneratorState.CLOSED:
            raise StopIteration()

        if self._state == GeneratorState.RUNNING:
            raise ValueError("generator already executing")

        if self._state == GeneratorState.CREATED and value is not None:
            raise TypeError("can't send non-None value to a just-started generator")

        self._state = GeneratorState.RUNNING
        try:
            if hasattr(self._gen, "send"):
                val = self._gen.send(value)
            else:
                val = next(self._gen)
            self._state = GeneratorState.SUSPENDED
            return val
        except StopIteration as e:
            self._state = GeneratorState.CLOSED
            self._value = getattr(e, "value", None)
            raise
        except BaseException:
            self._state = GeneratorState.CLOSED
            raise

    def throw(
        self,
        typ: Union[Type[BaseException], BaseException],
        val: Any = None,
        tb: Any = None,
    ) -> T:
        """
        Raise an exception inside the generator at the point where it was paused.
        Returns next yielded value, or raises StopIteration / re-raises exception.
        """
        if self._state == GeneratorState.CLOSED:
            if isinstance(typ, BaseException):
                exc = typ
            elif isinstance(typ, type) and issubclass(typ, BaseException):
                if isinstance(val, BaseException):
                    exc = val
                elif val is not None:
                    exc = typ(val)
                else:
                    exc = typ()
            else:
                exc = RuntimeError(f"Invalid exception: {typ}")
            if tb is not None:
                exc = exc.with_traceback(tb)
            raise exc

        if self._state == GeneratorState.RUNNING:
            raise ValueError("generator already executing")

        self._state = GeneratorState.RUNNING
        try:
            if isinstance(typ, BaseException):
                exc = typ
            elif isinstance(typ, type) and issubclass(typ, BaseException):
                if isinstance(val, BaseException):
                    exc = val
                elif val is not None:
                    exc = typ(val)
                else:
                    exc = typ()
            else:
                exc = RuntimeError(f"Invalid exception: {typ}")
            if tb is not None:
                exc = exc.with_traceback(tb)

            if hasattr(self._gen, "throw"):
                try:
                    val_yielded = self._gen.throw(exc)
                except TypeError:
                    val_yielded = self._gen.throw(typ, val, tb)
                self._state = GeneratorState.SUSPENDED
                return val_yielded
            else:
                raise exc
        except StopIteration as e:
            self._state = GeneratorState.CLOSED
            self._value = getattr(e, "value", None)
            raise
        except BaseException:
            self._state = GeneratorState.CLOSED
            raise

    def close(self) -> None:
        """
        Closes the generator cleanly, raising GeneratorExit inside if suspended.
        """
        if self._state == GeneratorState.CLOSED:
            return
        if self._state == GeneratorState.RUNNING:
            raise ValueError("generator already executing")

        self._state = GeneratorState.RUNNING
        try:
            if hasattr(self._gen, "close"):
                self._gen.close()
        finally:
            self._state = GeneratorState.CLOSED

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def to_pipeline(self) -> StreamPipeline[T]:
        """Convert this generator into a fluent StreamPipeline."""
        return StreamPipeline(self)

    @classmethod
    def from_generator_func(
        cls, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> SynapseGenerator[Any, Any, Any]:
        """
        Create a SynapseGenerator from a generator function.
        Calls func(*args, **kwargs) and wraps the result in a SynapseGenerator.
        """
        if callable(func):
            return cls(func(*args, **kwargs))
        return cls(func)


def from_generator_func(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> SynapseGenerator[Any, Any, Any]:
    """Module-level helper to create a SynapseGenerator from a generator function."""
    return SynapseGenerator.from_generator_func(func, *args, **kwargs)


def synapse_generator(
    func: Callable[..., Any],
) -> Callable[..., SynapseGenerator[Any, Any, Any]]:
    """Decorator to mark a function as returning a SynapseGenerator."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> SynapseGenerator[Any, Any, Any]:
        return SynapseGenerator(func(*args, **kwargs))

    return wrapper


class _LazyIterable(Generic[T]):
    """An iterable backed by a generator factory so it can be re-iterated if source allows."""

    def __init__(self, factory: Callable[[], Iterator[T]]):
        self._factory = factory

    def __iter__(self) -> Iterator[T]:
        return self._factory()


class StreamPipeline(Generic[T]):
    """
    Fluent lazy streaming pipeline for high-performance data processing.

    Operations are evaluated lazily on demand upon iteration or terminal consumption.
    Pipelines derived from reusable sources (e.g. lists, collections) can be re-iterated.
    """

    def __init__(self, source: Iterable[T]):
        self._source: Iterable[T] = source

    def __iter__(self) -> Iterator[T]:
        return iter(self._source)

    def map(self, fn: Callable[[T], U]) -> StreamPipeline[U]:
        """Lazily map each element using fn."""
        def _factory() -> Iterator[U]:
            for item in self:
                yield fn(item)

        return StreamPipeline(_LazyIterable(_factory))

    def filter(self, predicate: Callable[[T], bool]) -> StreamPipeline[T]:
        """Lazily filter elements matching predicate."""
        def _factory() -> Iterator[T]:
            for item in self:
                if predicate(item):
                    yield item

        return StreamPipeline(_LazyIterable(_factory))

    def take(self, n: int) -> StreamPipeline[T]:
        """Lazily take up to n elements from stream."""
        def _factory() -> Iterator[T]:
            if n <= 0:
                return
            count = 0
            for item in self:
                yield item
                count += 1
                if count >= n:
                    break

        return StreamPipeline(_LazyIterable(_factory))

    def skip(self, n: int) -> StreamPipeline[T]:
        """Lazily skip the first n elements from stream."""
        def _factory() -> Iterator[T]:
            if n <= 0:
                for item in self:
                    yield item
                return
            it = iter(self)
            count = 0
            while count < n:
                try:
                    next(it)
                    count += 1
                except StopIteration:
                    return
            for item in it:
                yield item

        return StreamPipeline(_LazyIterable(_factory))

    def batch(self, size: int) -> StreamPipeline[List[T]]:
        """Lazily group items into batches of specified size."""
        if size <= 0:
            raise ValueError(f"Batch size must be greater than 0, got {size}")

        def _factory() -> Iterator[List[T]]:
            current: List[T] = []
            for item in self:
                current.append(item)
                if len(current) == size:
                    yield current
                    current = []
            if current:
                yield current

        return StreamPipeline(_LazyIterable(_factory))

    def collect(self) -> List[T]:
        """Terminal operation: consume stream into a list."""
        return list(self)

    @overload
    def reduce(self, fn: Callable[[T, T], T]) -> T: ...
    @overload
    def reduce(self, fn: Callable[[U, T], U], initial: U) -> U: ...

    def reduce(self, fn: Callable[[Any, T], Any], initial: Any = _SENTINEL) -> Any:
        """Terminal operation: reduce stream to a single value using fn."""
        if initial is _SENTINEL:
            return functools.reduce(fn, self)
        return functools.reduce(fn, self, initial)

    def for_each(self, fn: Callable[[T], Any]) -> None:
        """Terminal operation: execute fn on each element."""
        for item in self:
            fn(item)

    def first(self, default: Any = None) -> Optional[T]:
        """Terminal operation: get the first element or default."""
        for item in self:
            return item
        return default

    def count(self) -> int:
        """Terminal operation: count total elements in stream."""
        c = 0
        for _ in self:
            c += 1
        return c

    def to_generator(self) -> SynapseGenerator[T, None, None]:
        """Convert pipeline to a SynapseGenerator."""
        return SynapseGenerator(self)

    def chain(self, *others: Iterable[T]) -> StreamPipeline[T]:
        """Lazily chain this stream with other iterables."""
        def _factory() -> Iterator[T]:
            for item in self:
                yield item
            for other in others:
                for item in other:
                    yield item

        return StreamPipeline(_LazyIterable(_factory))

    def flatten(self) -> StreamPipeline[Any]:
        """Lazily flatten nested iterables."""
        def _factory() -> Iterator[Any]:
            for item in self:
                if isinstance(item, Iterable) and not isinstance(item, (str, bytes, bytearray)):
                    for sub in item:
                        yield sub
                else:
                    yield item

        return StreamPipeline(_LazyIterable(_factory))

    @classmethod
    def from_iterable(cls, iterable: Iterable[T]) -> StreamPipeline[T]:
        """Create a StreamPipeline from any iterable."""
        return cls(iterable)

    @classmethod
    def from_generator(cls, gen: Any) -> StreamPipeline[T]:
        """Create a StreamPipeline from a generator or SynapseGenerator."""
        return cls(gen)


def _extract_delta(chunk: Any) -> str:
    """
    Extract string delta from various LLM / AI streaming chunk formats.
    Supports OpenAI, Anthropic, Ollama, HuggingFace, vLLM, raw strings, and byte payloads.
    Non-text metadata chunks (e.g. usage, empty choices, finish reasons) return empty strings.
    """
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, (bytes, bytearray)):
        return chunk.decode("utf-8", errors="replace")

    # Dict formats
    if isinstance(chunk, dict):
        # OpenAI style choices
        choices = chunk.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            c0 = choices[0]
            if isinstance(c0, dict):
                delta = c0.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if content is not None:
                        return str(content)
                    text = delta.get("text")
                    if text is not None:
                        return str(text)
                elif isinstance(delta, str):
                    return delta
                text = c0.get("text")
                if text is not None:
                    return str(text)
            elif hasattr(c0, "delta"):
                d = getattr(c0, "delta")
                if hasattr(d, "content") and d.content is not None:
                    return str(d.content)
                if hasattr(d, "text") and d.text is not None:
                    return str(d.text)
            elif hasattr(c0, "text") and c0.text is not None:
                return str(c0.text)

        # Anthropic / direct delta dict
        if "delta" in chunk:
            d = chunk["delta"]
            if isinstance(d, dict):
                content = d.get("content")
                if content is not None:
                    return str(content)
                text = d.get("text")
                if text is not None:
                    return str(text)
            elif hasattr(d, "content") and d.content is not None:
                return str(d.content)
            elif hasattr(d, "text") and d.text is not None:
                return str(d.text)
            elif isinstance(d, str):
                return d

        # Ollama style message content
        if "message" in chunk and isinstance(chunk["message"], dict):
            content = chunk["message"].get("content")
            if content is not None:
                return str(content)

        # Direct content or text keys
        if "content" in chunk and chunk["content"] is not None:
            return str(chunk["content"])
        if "text" in chunk and chunk["text"] is not None:
            return str(chunk["text"])
        if "response" in chunk and chunk["response"] is not None:
            return str(chunk["response"])
        if "token" in chunk and isinstance(chunk["token"], dict) and "text" in chunk["token"]:
            return str(chunk["token"]["text"])

        # Dict without textual fields is a metadata chunk (usage, finish_reason, ping, etc.)
        return ""

    # Object formats (OpenAI / Anthropic Pydantic models, custom SDK objects)
    if hasattr(chunk, "choices"):
        choices = getattr(chunk, "choices", None)
        if choices and len(choices) > 0:
            c0 = choices[0]
            delta = getattr(c0, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                if content is not None:
                    return str(content)
                text = getattr(delta, "text", None)
                if text is not None:
                    return str(text)
            text = getattr(c0, "text", None)
            if text is not None:
                return str(text)
        return ""

    if hasattr(chunk, "delta"):
        d = getattr(chunk, "delta")
        if hasattr(d, "content") and d.content is not None:
            return str(d.content)
        if hasattr(d, "text") and d.text is not None:
            return str(d.text)
        if isinstance(d, str):
            return d
        return ""

    if hasattr(chunk, "message"):
        msg = getattr(chunk, "message")
        if hasattr(msg, "content") and msg.content is not None:
            return str(msg.content)

    if hasattr(chunk, "text"):
        val = getattr(chunk, "text")
        if val is not None:
            return str(val)

    if hasattr(chunk, "content"):
        val = getattr(chunk, "content")
        if val is not None:
            return str(val)

    if hasattr(chunk, "response"):
        val = getattr(chunk, "response")
        if val is not None:
            return str(val)

    # Primitives
    if isinstance(chunk, (int, float, bool)):
        return str(chunk)

    return ""


class TokenStream(StreamPipeline[str]):
    """
    Specialized lazy stream for LLM / AI token streaming.

    Features:
    - Chunk buffering: buffer tokens until threshold chunk_size
    - Delta yielding: emits safe text deltas lazily
    - Stop sequence handling: detects stop sequences across chunk boundaries
      and terminates immediately without leaking the stop sequence.
    - Accumulated text access via .text, .current_text, and .collect_text()
    - Inspection flags: .is_stopped, .matched_stop
    """

    def __init__(
        self,
        source: Iterable[Any],
        stop: Optional[Union[str, List[str]]] = None,
        stop_sequences: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = None,
    ):
        stops: List[str] = []
        if stop is not None:
            if isinstance(stop, str):
                stops.append(stop)
            else:
                stops.extend(stop)
        if stop_sequences is not None:
            if isinstance(stop_sequences, str):
                stops.append(stop_sequences)
            else:
                stops.extend(stop_sequences)

        seen = set()
        clean_stops: List[str] = []
        for s in stops:
            s_str = str(s)
            if s_str and s_str not in seen:
                seen.add(s_str)
                clean_stops.append(s_str)

        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")

        self._raw_source: Iterable[Any] = source
        self.stop_sequences: List[str] = clean_stops
        self.chunk_size: Optional[int] = chunk_size

        self._is_stopped: bool = False
        self._matched_stop: Optional[str] = None
        self._accumulated_text: List[str] = []
        self._exhausted: bool = False

        super().__init__(self._generate_tokens())

    @property
    def is_stopped(self) -> bool:
        """True if the token stream was terminated by a stop sequence."""
        return self._is_stopped

    @property
    def matched_stop(self) -> Optional[str]:
        """The stop sequence that caused termination, or None."""
        return self._matched_stop

    @property
    def current_text(self) -> str:
        """Returns the text accumulated so far without consuming remaining stream."""
        return "".join(self._accumulated_text)

    @property
    def text(self) -> str:
        """Returns the full text collected so far, or consumes the stream if not yet consumed."""
        if not self._exhausted:
            self.collect()
        return "".join(self._accumulated_text)

    def collect_text(self) -> str:
        """Consumes the stream and returns the concatenated string."""
        if not self._exhausted:
            self.collect()
        return "".join(self._accumulated_text)

    def collect(self) -> List[str]:
        """Terminal operation: consume stream into a list of delta strings."""
        if not self._exhausted:
            for _ in self:
                pass
        return list(self._accumulated_text)

    def with_stop(self, stop: Union[str, List[str]]) -> TokenStream:
        """Return a new TokenStream with additional or updated stop sequence(s)."""
        combined = list(self.stop_sequences)
        if isinstance(stop, str):
            combined.append(stop)
        else:
            combined.extend(stop)
        return TokenStream(
            self._raw_source,
            stop_sequences=combined,
            chunk_size=self.chunk_size,
        )

    def with_chunk_size(self, size: int) -> TokenStream:
        """Return a new TokenStream with specified chunk buffering size."""
        return TokenStream(
            self._raw_source,
            stop_sequences=self.stop_sequences,
            chunk_size=size,
        )

    def _generate_tokens(self) -> Iterator[str]:
        buffer = ""
        stops = self.stop_sequences
        chunk_buffer = ""
        use_chunk_buffer = self.chunk_size is not None and self.chunk_size > 1
        chunk_sz = self.chunk_size or 1

        try:
            for raw_chunk in self._raw_source:
                delta = _extract_delta(raw_chunk)
                if not delta:
                    continue
                buffer += delta

                if stops:
                    # Check for exact stop matches in current buffer
                    earliest_idx = -1
                    earliest_stop = None
                    for s in stops:
                        idx = buffer.find(s)
                        if idx != -1:
                            if earliest_idx == -1 or idx < earliest_idx:
                                earliest_idx = idx
                                earliest_stop = s

                    if earliest_stop is not None:
                        # Found stop sequence: emit preceding content cleanly
                        emit_part = buffer[:earliest_idx]
                        buffer = ""
                        self._is_stopped = True
                        self._matched_stop = earliest_stop

                        if use_chunk_buffer:
                            emit_part = chunk_buffer + emit_part
                            chunk_buffer = ""
                            total_len = len(emit_part)
                            num_full = total_len // chunk_sz
                            for i in range(num_full):
                                part = emit_part[i * chunk_sz : (i + 1) * chunk_sz]
                                self._accumulated_text.append(part)
                                yield part
                            remainder = emit_part[num_full * chunk_sz :]
                            if remainder:
                                self._accumulated_text.append(remainder)
                                yield remainder
                        else:
                            if emit_part:
                                self._accumulated_text.append(emit_part)
                                yield emit_part
                        return

                    # No complete match yet: check if any suffix of buffer is a prefix of any stop
                    longest_overlap = 0
                    for s in stops:
                        max_check = min(len(buffer), len(s) - 1)
                        for k in range(max_check, 0, -1):
                            if buffer.endswith(s[:k]):
                                if k > longest_overlap:
                                    longest_overlap = k
                                break

                    safe_len = len(buffer) - longest_overlap
                    if safe_len > 0:
                        to_emit = buffer[:safe_len]
                        buffer = buffer[safe_len:]
                        if use_chunk_buffer:
                            chunk_buffer += to_emit
                            total_len = len(chunk_buffer)
                            num_full = total_len // chunk_sz
                            for i in range(num_full):
                                part = chunk_buffer[i * chunk_sz : (i + 1) * chunk_sz]
                                self._accumulated_text.append(part)
                                yield part
                            chunk_buffer = chunk_buffer[num_full * chunk_sz :]
                        else:
                            self._accumulated_text.append(to_emit)
                            yield to_emit
                else:
                    # No stop sequences configured
                    if use_chunk_buffer:
                        chunk_buffer += buffer
                        buffer = ""
                        total_len = len(chunk_buffer)
                        num_full = total_len // chunk_sz
                        for i in range(num_full):
                            part = chunk_buffer[i * chunk_sz : (i + 1) * chunk_sz]
                            self._accumulated_text.append(part)
                            yield part
                        chunk_buffer = chunk_buffer[num_full * chunk_sz :]
                    else:
                        self._accumulated_text.append(buffer)
                        yield buffer
                        buffer = ""

            # Stream finished naturally: flush remaining buffers
            remaining = buffer
            if use_chunk_buffer:
                remaining = chunk_buffer + remaining
                total_len = len(remaining)
                num_full = total_len // chunk_sz
                for i in range(num_full):
                    part = remaining[i * chunk_sz : (i + 1) * chunk_sz]
                    self._accumulated_text.append(part)
                    yield part
                tail = remaining[num_full * chunk_sz :]
                if tail:
                    self._accumulated_text.append(tail)
                    yield tail
            else:
                if remaining:
                    self._accumulated_text.append(remaining)
                    yield remaining

        finally:
            self._exhausted = True


__all__ = [
    "GeneratorState",
    "SynapseGenerator",
    "StreamPipeline",
    "TokenStream",
    "from_generator_func",
    "synapse_generator",
]
