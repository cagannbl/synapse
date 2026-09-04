import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine
from synapse.ai.providers import MockProvider, set_llm_provider, reset_llm_provider


@pytest.fixture(autouse=True)
def clean_llm_provider():
    yield
    reset_llm_provider()


def run_source(source: str) -> VirtualMachine:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(code)
    return vm


def test_ai_prompt_sentiment_classification():
    mock = MockProvider()
    set_llm_provider(mock)

    source = """
prompt analyze_sentiment(review):
    system: "Classify review sentiment."
    user: review

let pos_res = analyze_sentiment("Synapse language is fast and intuitive!")
let neg_res = analyze_sentiment("I found a terrible bug and hate it.")
"""
    vm = run_source(source)
    assert vm.globals["pos_res"] == "POSITIVE"
    assert vm.globals["neg_res"] == "NEGATIVE"


def test_ai_agent_runtime():
    mock = MockProvider(default_response="Task completed successfully.")
    set_llm_provider(mock)

    source = """
agent CodeBot:
    model: "mock-model"
    instructions: "You fix syntax errors and analyze ASTs."

let result = CodeBot.run("Check my neural network layers.")
"""
    vm = run_source(source)
    assert "Task completed" in vm.globals["result"]
    assert vm.globals["CodeBot"].name == "CodeBot"


def test_prompt_template_substring_safety():
    mock = MockProvider()
    set_llm_provider(mock)

    # Parametre adı 'text', ancak sistem ve kullanıcı promptunda 'context' geçiyor
    source = """
prompt analyze_context(text):
    system: "You analyze context and subtext."
    user: f"Here is the context: '{text}'"

let res = analyze_context("hello world")
"""
    vm = run_source(source)
    # mock.last_prompt kontrol et: 'context' kelimesi bozulmamış olmalı
    assert "context" in mock.last_prompt
    assert "hello world" in mock.last_prompt
    # Asla 'conhello world' gibi bozulma olmamalı
    assert "conhello" not in mock.last_prompt


def test_ai_agent_react_tool_execution():
    import json
    # Mock LLM ilk adımda tool çağrısı yapacak, ikinci adımda nihai cevabı verecek
    mock = MockProvider(responses=[
        '```json\n{"tool": "multiply", "args": [6, 7]}\n```',
        'Final Answer: Result of calculation is 42'
    ])
    set_llm_provider(mock)

    source = """
tool multiply(a, b):
    return a * b

agent CalculatorBot:
    model: "mock-model"
    tools: [multiply]
    instructions: "You use tools to calculate expressions."

let final_res = CalculatorBot.run("Multiply 6 by 7")
"""
    vm = run_source(source)
    assert "Result of calculation is 42" in vm.globals["final_res"]
    assert "multiply" in vm.globals["CalculatorBot"].tools
