"""
Unit tests for Faz 5: Jeneratörler & Tembel Veri Akışı Motoru (SynapseGenerator, StreamPipeline, TokenStream).
"""

from __future__ import annotations

import itertools
import pytest
from typing import Generator, List

from synapse.core.generators import (
    GeneratorState,
    StreamPipeline,
    SynapseGenerator,
    TokenStream,
    from_generator_func,
    synapse_generator,
)


class CustomTestError(Exception):
    """Custom exception for throw testing."""
    pass


# ==============================================================================
# 1. Basic Generation and StopIteration Tests
# ==============================================================================

def test_basic_generation_and_stop_iteration():
    """Test standard generator iteration with __next__ and StopIteration."""
    def number_gen() -> Generator[int, None, str]:
        yield 1
        yield 2
        yield 3
        return "completed"

    gen = SynapseGenerator(number_gen())
    assert gen.state == GeneratorState.CREATED
    assert gen.value is None

    assert next(gen) == 1
    assert gen.state == GeneratorState.SUSPENDED

    assert next(gen) == 2
    assert gen.state == GeneratorState.SUSPENDED

    assert next(gen) == 3
    assert gen.state == GeneratorState.SUSPENDED

    with pytest.raises(StopIteration) as exc_info:
        next(gen)

    assert exc_info.value.value == "completed"
    assert gen.state == GeneratorState.CLOSED
    assert gen.value == "completed"

    # Subsequent next() continues to raise StopIteration
    with pytest.raises(StopIteration):
        next(gen)


def test_generator_for_loop_and_empty():
    """Test SynapseGenerator in a standard Python for-loop and with empty iterables."""
    def empty_gen():
        if False:
            yield

    gen = SynapseGenerator(empty_gen())
    items = list(gen)
    assert items == []
    assert gen.state == GeneratorState.CLOSED

    # Test with standard iterable wrapping
    gen_list = SynapseGenerator([10, 20, 30])
    assert list(gen_list) == [10, 20, 30]
    assert gen_list.state == GeneratorState.CLOSED


# ==============================================================================
# 2. State Transitions & Inspection Tests
# ==============================================================================

def test_generator_state_transitions():
    """Verify CREATED -> SUSPENDED -> CLOSED state transitions."""
    def simple_gen():
        yield "a"
        yield "b"

    sg = SynapseGenerator(simple_gen())
    assert sg.state == GeneratorState.CREATED
    assert not sg.gi_running

    item1 = sg.send(None)
    assert item1 == "a"
    assert sg.state == GeneratorState.SUSPENDED

    item2 = sg.send(None)
    assert item2 == "b"
    assert sg.state == GeneratorState.SUSPENDED

    with pytest.raises(StopIteration):
        sg.send(None)

    assert sg.state == GeneratorState.CLOSED


# ==============================================================================
# 3. send() Protocol Tests
# ==============================================================================

def test_send_protocol():
    """Test bi-directional send() communication into suspended generator."""
    def echo_gen() -> Generator[str, str, str]:
        val = yield "ready"
        while val != "quit":
            val = yield f"echo:{val}"
        return "finished"

    sg = SynapseGenerator(echo_gen())

    # Sending non-None to a just-started generator must raise TypeError
    with pytest.raises(TypeError, match="can't send non-None value"):
        sg.send("hello")

    # Start generator with None
    initial = sg.send(None)
    assert initial == "ready"
    assert sg.state == GeneratorState.SUSPENDED

    res1 = sg.send("synapse")
    assert res1 == "echo:synapse"
    assert sg.state == GeneratorState.SUSPENDED

    res2 = sg.send("stream")
    assert res2 == "echo:stream"
    assert sg.state == GeneratorState.SUSPENDED

    with pytest.raises(StopIteration) as exc_info:
        sg.send("quit")

    assert exc_info.value.value == "finished"
    assert sg.state == GeneratorState.CLOSED
    assert sg.value == "finished"


# ==============================================================================
# 4. throw() Protocol Tests
# ==============================================================================

def test_throw_protocol_caught():
    """Test generator catching an exception thrown via throw() and continuing."""
    def resilient_gen():
        try:
            yield "first"
        except CustomTestError:
            yield "recovered_from_error"
        yield "second"

    sg = SynapseGenerator(resilient_gen())
    assert next(sg) == "first"
    assert sg.state == GeneratorState.SUSPENDED

    # Throw exception: generator catches it and yields recovery value
    res = sg.throw(CustomTestError("test exception"))
    assert res == "recovered_from_error"
    assert sg.state == GeneratorState.SUSPENDED

    # Generator can continue normally after catching
    assert next(sg) == "second"
    assert sg.state == GeneratorState.SUSPENDED

    with pytest.raises(StopIteration):
        next(sg)
    assert sg.state == GeneratorState.CLOSED


def test_throw_protocol_uncaught():
    """Test unhandled exception via throw() sets state to CLOSED and propagates."""
    def fragile_gen():
        yield "only_item"

    sg = SynapseGenerator(fragile_gen())
    assert next(sg) == "only_item"

    with pytest.raises(CustomTestError, match="unhandled"):
        sg.throw(CustomTestError("unhandled"))

    assert sg.state == GeneratorState.CLOSED

    # Calling next or send on closed raises StopIteration
    with pytest.raises(StopIteration):
        next(sg)


# ==============================================================================
# 5. close() Protocol Tests
# ==============================================================================

def test_close_and_subsequent_iteration():
    """Test close() cleanly terminates generator and makes next() raise StopIteration."""
    cleanup_called = False

    def monitored_gen():
        nonlocal cleanup_called
        try:
            yield 1
            yield 2
        finally:
            cleanup_called = True

    sg = SynapseGenerator(monitored_gen())
    assert next(sg) == 1
    assert sg.state == GeneratorState.SUSPENDED
    assert not cleanup_called

    sg.close()
    assert cleanup_called
    assert sg.state == GeneratorState.CLOSED

    # Subsequent iterations must immediately raise StopIteration
    with pytest.raises(StopIteration):
        next(sg)

    # Calling close on already closed generator is a safe no-op
    sg.close()
    assert sg.state == GeneratorState.CLOSED


# ==============================================================================
# 6. Factory and Decorator Helpers
# ==============================================================================

def test_from_generator_func_factory_and_decorator():
    """Test from_generator_func factory, arguments handling, and decorator."""
    def gen_func_no_args():
        yield 100
        yield 200

    # 1. 0-arg function directly to generator
    sg1 = SynapseGenerator.from_generator_func(gen_func_no_args)
    assert isinstance(sg1, SynapseGenerator)
    assert list(sg1) == [100, 200]

    # 2. Module level helper with args
    def gen_func_with_args(start: int, count: int):
        for i in range(count):
            yield start + i

    sg2 = from_generator_func(gen_func_with_args, 10, 3)
    assert isinstance(sg2, SynapseGenerator)
    assert list(sg2) == [10, 11, 12]

    # 3. synapse_generator decorator
    @synapse_generator
    def decorated_gen(x: int):
        yield x * 1
        yield x * 2

    sg3 = decorated_gen(5)
    assert isinstance(sg3, SynapseGenerator)
    assert list(sg3) == [5, 10]


# ==============================================================================
# 7. StreamPipeline Operations: map, filter, take, skip, batch, collect, reduce
# ==============================================================================

def test_stream_pipeline_map_and_filter():
    """Test lazy map and filter operations on StreamPipeline."""
    evaluated = []

    def source():
        for i in range(10):
            evaluated.append(i)
            yield i

    pipe = (
        StreamPipeline(source())
        .filter(lambda x: x % 2 == 0)
        .map(lambda x: x * 10)
    )

    # Laziness check: nothing evaluated before collect()
    assert evaluated == []

    result = pipe.collect()
    assert result == [0, 20, 40, 60, 80]
    assert len(evaluated) == 10


def test_stream_pipeline_take_and_skip():
    """Test lazy take and skip operations."""
    pipe = StreamPipeline(range(10)).skip(3).take(4)
    assert pipe.collect() == [3, 4, 5, 6]

    # Boundary and edge cases
    assert StreamPipeline(range(10)).take(0).collect() == []
    assert StreamPipeline(range(10)).take(-5).collect() == []
    assert StreamPipeline(range(5)).skip(10).collect() == []
    assert StreamPipeline(range(5)).skip(0).collect() == [0, 1, 2, 3, 4]
    assert StreamPipeline(range(5)).skip(-2).collect() == [0, 1, 2, 3, 4]


def test_stream_pipeline_batch():
    """Test batching elements with exact and partial chunk divisions."""
    # Uneven batches
    b1 = StreamPipeline(range(7)).batch(3).collect()
    assert b1 == [[0, 1, 2], [3, 4, 5], [6]]

    # Even batches
    b2 = StreamPipeline(range(6)).batch(2).collect()
    assert b2 == [[0, 1], [2, 3], [4, 5]]

    # Empty stream
    b3 = StreamPipeline([]).batch(3).collect()
    assert b3 == []

    # Invalid batch size
    with pytest.raises(ValueError, match="greater than 0"):
        StreamPipeline(range(5)).batch(0)


def test_stream_pipeline_reduce():
    """Test reduce with and without initial accumulator."""
    # With initial value
    sum_plus_10 = StreamPipeline([1, 2, 3, 4]).reduce(lambda acc, x: acc + x, 10)
    assert sum_plus_10 == 20

    # Without initial value
    prod = StreamPipeline([1, 2, 3, 4]).reduce(lambda acc, x: acc * x)
    assert prod == 24

    # Empty with initial value
    empty_sum = StreamPipeline([]).reduce(lambda acc, x: acc + x, 0)
    assert empty_sum == 0

    # Empty without initial raises TypeError
    with pytest.raises(TypeError):
        StreamPipeline([]).reduce(lambda acc, x: acc + x)


def test_stream_pipeline_utilities():
    """Test helper utilities: first, count, for_each, chain, flatten."""
    p = StreamPipeline([10, 20, 30])
    assert p.first() == 10
    assert StreamPipeline([]).first("default") == "default"
    assert StreamPipeline([1, 2, 3]).count() == 3

    seen = []
    StreamPipeline([1, 2]).for_each(seen.append)
    assert seen == [1, 2]

    # Chain
    chained = StreamPipeline([1, 2]).chain([3, 4], [5]).collect()
    assert chained == [1, 2, 3, 4, 5]

    # Flatten
    nested = StreamPipeline([[1, 2], [3, [4, 5]]]).flatten().collect()
    assert nested == [1, 2, 3, [4, 5]]


# ==============================================================================
# 8. Memory Efficiency with Infinite Generator
# ==============================================================================

def test_memory_efficiency_infinite_generator():
    """Ensure infinite generator evaluated with .take(5) does not hang or exhaust memory."""
    def infinite_counter():
        count = 0
        while True:
            yield count
            count += 1

    # Pipeline chaining infinite stream with filter, map, and take(5)
    pipe = (
        StreamPipeline(infinite_counter())
        .filter(lambda x: x % 2 == 0)
        .map(lambda x: f"num_{x}")
        .take(5)
    )

    result = pipe.collect()
    assert result == ["num_0", "num_2", "num_4", "num_6", "num_8"]


# ==============================================================================
# 9. TokenStream Tests
# ==============================================================================

def test_token_stream_basic_and_text_emission():
    """Test TokenStream delta emission, text property, and collect_text()."""
    tokens = ["Synapse", " ", "AI", " is", " ultra", " fast."]
    stream = TokenStream(tokens)

    # Test iterating deltas
    deltas = list(stream)
    assert deltas == tokens
    assert stream.text == "Synapse AI is ultra fast."
    assert stream.collect_text() == "Synapse AI is ultra fast."
    assert stream.is_stopped is False
    assert stream.matched_stop is None


def test_token_stream_stop_sequence_single_chunk():
    """Test stop sequence cutoff within a single token chunk."""
    tokens = [
        "Thinking process...\n",
        "Result: 42\n<|STOP|> Additional leaked secret",
        "Should not be processed",
    ]
    stream = TokenStream(tokens, stop="<|STOP|>")

    deltas = stream.collect()
    assert stream.is_stopped is True
    assert stream.matched_stop == "<|STOP|>"
    assert stream.text == "Thinking process...\nResult: 42\n"
    assert "leaked secret" not in stream.text
    assert "Should not be processed" not in stream.text


def test_token_stream_stop_sequence_split_across_chunks():
    """Test stop sequence fragmented across multiple consecutive token chunks."""
    # Stop sequence "<|im_end|>" split across 3 chunks: ["<|im", "_e", "nd|>"]
    tokens = [
        "Answer: ",
        "Paris",
        " is the",
        " capital.<|im",
        "_e",
        "nd|>",
        " extra garbage content",
    ]
    stream = TokenStream(tokens, stop_sequences=["<|im_end|>"])

    deltas = stream.collect()
    assert stream.is_stopped is True
    assert stream.matched_stop == "<|im_end|>"
    assert stream.text == "Answer: Paris is the capital."
    assert "<|im" not in stream.text
    assert "extra garbage" not in stream.text


def test_token_stream_stop_sequence_false_alarm():
    """Test that a partial stop prefix that does not complete is safely flushed."""
    # Stop sequence is "###END"
    # Chunk contains "###" but followed by "START" instead of "END"
    tokens = [
        "Header: ",
        "###",
        "START",
        " of block.",
    ]
    stream = TokenStream(tokens, stop="###END")

    deltas = stream.collect()
    assert stream.is_stopped is False
    assert stream.matched_stop is None
    assert stream.text == "Header: ###START of block."


def test_token_stream_multiple_stop_sequences():
    """Test multiple stop sequences with earliest match precedence."""
    tokens = [
        "Line 1\n",
        "Line 2\n",
        "### Human: Next prompt",
        "Line 3\n",
    ]
    stream = TokenStream(tokens, stop=["\n\n", "### Human:"])

    deltas = stream.collect()
    assert stream.is_stopped is True
    assert stream.matched_stop == "### Human:"
    assert stream.text == "Line 1\nLine 2\n"


def test_token_stream_chunk_buffering():
    """Test token stream chunk buffering with chunk_size."""
    char_tokens = ["a", "b", "c", "d", "e", "f", "g"]
    stream = TokenStream(char_tokens, chunk_size=3)

    deltas = stream.collect()
    # Chunks of 3: "abc", "def", and remainder "g"
    assert deltas == ["abc", "def", "g"]
    assert stream.text == "abcdefg"

    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        TokenStream(["a"], chunk_size=0)


def test_token_stream_openai_dict_format():
    """Test extraction of text deltas from OpenAI API style chunk dictionaries."""
    chunks = [
        {"choices": [{"delta": {"content": "Deep"}}]},
        {"choices": [{"delta": {"content": " learning"}}]},
        {"choices": [{"delta": {"content": " with"}}]},
        {"choices": [{"delta": {"content": " Synapse."}}]},
        {"choices": [{"delta": {"content": "[STOP] ignored"}}]},
    ]
    stream = TokenStream(chunks, stop="[STOP]")
    assert stream.collect_text() == "Deep learning with Synapse."
    assert stream.is_stopped is True
    assert stream.matched_stop == "[STOP]"


def test_token_stream_fluent_pipeline_integration():
    """Test TokenStream inherited StreamPipeline methods."""
    tokens = ["apple", "banana", "cherry", "date"]
    stream = TokenStream(tokens)

    # Use pipeline map and filter
    uppercased = stream.filter(lambda t: len(t) > 5).map(str.upper).collect()
    assert uppercased == ["BANANA", "CHERRY"]


# ==============================================================================
# 10. Generator to Pipeline Interop
# ==============================================================================

def test_generator_to_pipeline_fluent_chain():
    """Test seamless conversion between SynapseGenerator and StreamPipeline."""
    def number_stream():
        for i in range(1, 20):
            yield i

    sg = SynapseGenerator(number_stream())
    res = (
        sg.to_pipeline()
        .filter(lambda x: x % 2 != 0)  # 1, 3, 5, 7, 9, ...
        .map(lambda x: x * 2)          # 2, 6, 10, 14, 18, ...
        .take(4)
        .batch(2)
        .collect()
    )
    assert res == [[2, 6], [10, 14]]

    # Converting back to generator
    pipeline = StreamPipeline(["x", "y", "z"])
    gen = pipeline.to_generator()
    assert isinstance(gen, SynapseGenerator)
    assert next(gen) == "x"
    assert gen.state == GeneratorState.SUSPENDED
    assert list(gen) == ["y", "z"]
    assert gen.state == GeneratorState.CLOSED


# ==============================================================================
# 11. Additional Edge Case Tests
# ==============================================================================

def test_synapse_generator_subclass_with_generate():
    """Test subclassing SynapseGenerator and defining a custom generate() method."""
    class CustomGenerator(SynapseGenerator[int, None, None]):
        def generate(self):
            yield 42
            yield 84

    cg = CustomGenerator()
    assert cg.state == GeneratorState.CREATED
    assert list(cg) == [42, 84]
    assert cg.state == GeneratorState.CLOSED


def test_throw_on_closed_generator():
    """Test calling throw on an already closed generator raises the exception directly."""
    sg = SynapseGenerator([1])
    assert next(sg) == 1
    with pytest.raises(StopIteration):
        next(sg)
    assert sg.state == GeneratorState.CLOSED

    # Calling throw on closed generator raises the exception
    with pytest.raises(ValueError, match="closed test"):
        sg.throw(ValueError("closed test"))


def test_token_stream_stop_at_start():
    """Test when stop sequence is encountered immediately at the beginning."""
    tokens = ["<END>", "after end"]
    stream = TokenStream(tokens, stop="<END>")
    deltas = stream.collect()
    assert deltas == []
    assert stream.collect_text() == ""
    assert stream.is_stopped is True
    assert stream.matched_stop == "<END>"


def test_token_stream_with_stop_builder():
    """Test fluent builder methods with_stop and with_chunk_size."""
    tokens = ["hello", " world", " STOP"]
    stream = TokenStream(tokens).with_stop("STOP").with_chunk_size(4)
    text = stream.collect_text()
    assert text == "hello world "
    assert stream.is_stopped is True
    assert stream.matched_stop == "STOP"


def test_stream_pipeline_reusability():
    """Test that StreamPipeline over reusable iterables can be traversed multiple times."""
    pipeline = (
        StreamPipeline([1, 2, 3, 4, 5])
        .filter(lambda x: x % 2 != 0)
        .map(lambda x: x * 10)
    )
    first_run = pipeline.collect()
    second_run = pipeline.collect()
    assert first_run == [10, 30, 50]
    assert second_run == [10, 30, 50]

    # Re-iteration with batch and take
    p2 = StreamPipeline(range(10)).take(6).batch(2)
    assert p2.collect() == [[0, 1], [2, 3], [4, 5]]
    assert p2.collect() == [[0, 1], [2, 3], [4, 5]]


def test_close_on_running_generator_raises_error():
    """Test calling close on actively running generator raises ValueError('generator already executing')."""
    sg = None

    def self_closing():
        nonlocal sg
        sg.close()
        yield 1

    sg = SynapseGenerator(self_closing())
    with pytest.raises(ValueError, match="generator already executing"):
        next(sg)


def test_subsequent_stopiteration_clean_value():
    """Verify first StopIteration carries return value, while subsequent ones do not leak it."""
    def returning_gen():
        yield 1
        return "my_result"

    sg = SynapseGenerator(returning_gen())
    assert next(sg) == 1

    with pytest.raises(StopIteration) as first_stop:
        next(sg)
    assert first_stop.value.value == "my_result"
    assert sg.value == "my_result"

    with pytest.raises(StopIteration) as second_stop:
        next(sg)
    assert second_stop.value.value is None
    # Generator value attribute is preserved
    assert sg.value == "my_result"


def test_token_stream_openai_usage_metadata_ignored():
    """Verify metadata chunks (e.g. usage, finish reason, empty choices) are ignored and not leaked."""
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    stream = TokenStream(chunks)
    deltas = stream.collect()
    assert deltas == ["Hello", " world"]
    assert stream.text == "Hello world"
    assert "usage" not in stream.text
    assert "prompt_tokens" not in stream.text


def test_token_stream_anthropic_format_delta():
    """Verify Anthropic style stream chunks (dict with type: text_delta and SDK object)."""
    # Dict format
    dict_chunks = [
        {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Anthropic"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " stream"}},
        {"type": "message_stop"},
    ]
    stream_dict = TokenStream(dict_chunks)
    assert stream_dict.collect_text() == "Anthropic stream"

    # Object format
    class TextDelta:
        def __init__(self, text: str):
            self.text = text
            self.type = "text_delta"

    class StreamEvent:
        def __init__(self, delta: Any):
            self.delta = delta

    obj_chunks = [
        StreamEvent(TextDelta("Fast ")),
        StreamEvent(TextDelta("inference")),
    ]
    stream_obj = TokenStream(obj_chunks)
    assert stream_obj.collect_text() == "Fast inference"


def test_token_stream_ollama_and_bytes_format():
    """Verify Ollama format (message.content / response) and raw bytes decoding."""
    ollama_chunks = [
        {"message": {"role": "assistant", "content": "Ollama "}},
        {"response": "works!"},
    ]
    assert TokenStream(ollama_chunks).collect_text() == "Ollama works!"

    # Bytes streaming
    byte_chunks = [b"Byte ", b"stream ", b"tokens"]
    assert TokenStream(byte_chunks).collect_text() == "Byte stream tokens"


def test_token_stream_current_text_inspection():
    """Verify current_text inspects accumulated progress without prematurely consuming stream."""
    tokens = ["chunk1", "chunk2", "chunk3", "chunk4"]
    stream = TokenStream(tokens)
    it = iter(stream)

    assert stream.current_text == ""
    assert next(it) == "chunk1"
    assert stream.current_text == "chunk1"
    assert next(it) == "chunk2"
    assert stream.current_text == "chunk1chunk2"
    # Consuming the rest
    remaining = list(it)
    assert remaining == ["chunk3", "chunk4"]
    assert stream.current_text == "chunk1chunk2chunk3chunk4"
    assert stream.text == "chunk1chunk2chunk3chunk4"


def test_token_stream_overlapping_stops_stress():
    """Stress test overlapping stop sequences across various chunk splits."""
    stream1 = TokenStream(["b", "an", "an", "a", " extra"], stop=["banana", "anana"])
    assert stream1.collect_text() == ""
    assert stream1.is_stopped is True
    assert stream1.matched_stop == "banana"

    stream2 = TokenStream(["b", "an", "an", "a", " extra"], stop=["anana", "banana"])
    assert stream2.collect_text() == ""
    assert stream2.is_stopped is True
    assert stream2.matched_stop == "banana"

    # When "anana" occurs without "b"
    stream3 = TokenStream(["an", "an", "a", " rest"], stop=["banana", "anana"])
    assert stream3.collect_text() == ""
    assert stream3.is_stopped is True
    assert stream3.matched_stop == "anana"


def test_token_stream_large_chunk_slicing_performance():
    """Verify that chunk_size buffering with large chunk deltas evaluates in linear time."""
    large_payload = "A" * 100_000
    stream = TokenStream([large_payload], chunk_size=50)
    deltas = stream.collect()
    assert len(deltas) == 2000
    assert all(len(d) == 50 for d in deltas)
    assert stream.text == large_payload


