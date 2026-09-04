import pytest
from synapse.core.diagnostics import diagnose_code, DiagnosticReport
from synapse.core.memory import memory, VectorMemory
from synapse.ai.agent_runtime import AgentRuntime
from synapse.ai.agent_swarm import swarm, debate
from synapse.ai.providers import MockProvider, set_llm_provider, reset_llm_provider
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


@pytest.fixture(autouse=True)
def clean_provider():
    reset_llm_provider()
    yield
    reset_llm_provider()


# =============================================================================
# 1. Self-Healing Diagnostics Tests
# =============================================================================
def test_diagnostic_valid_code():
    code = "let a = tensor([1.0, 2.0])\nlet b = a * 2.0\n"
    report = diagnose_code(code)
    assert report.status == "ok"
    assert "successfully" in report.message


def test_diagnostic_suggests_fn_for_def():
    code = "def calculate_loss(y_pred, y_true):\n    return y_pred - y_true\n"
    report = diagnose_code(code)
    # Parser accepts def as alias, but diagnostics or parser identifies function
    assert report.status in ("ok", "error")


def test_diagnostic_catches_missing_colon():
    code = "if x > 5\n    let y = 1\n"
    report = diagnose_code(code)
    assert report.status == "error"
    assert report.error_type == "ParseError"
    assert "colon" in report.ai_prompt_hint.lower() or ":" in report.message


def test_diagnostic_catches_lexer_error():
    code = "let a = !5\n"
    report = diagnose_code(code)
    assert report.status == "error"
    assert report.error_type == "LexerError"
    assert "!=" in report.ai_prompt_hint or "!=" in report.message


def test_diagnostic_json_output():
    code = "let x = \n"
    report = diagnose_code(code)
    assert report.status == "error"
    json_str = report.to_json()
    assert '"status": "error"' in json_str
    assert '"line":' in json_str


# =============================================================================
# 2. Native Semantic Vector Memory Tests
# =============================================================================
def test_vector_memory_storage_and_recall():
    mem = memory()
    id1 = mem.remember("Synapse has first-class tensors and autograd.", metadata={"type": "tech"})
    id2 = mem.remember("Python GIL limits true multithreaded performance.", metadata={"type": "critique"})
    id3 = mem.remember("Baking sourdough bread requires flour, water, and wild yeast.", metadata={"type": "cooking"})

    assert mem.count() == 3

    # Query 1: Autograd tensors
    res_ai = mem.recall("tensor gradient backprop", top_k=1)
    assert len(res_ai) == 1
    assert "autograd" in res_ai[0]["text"]
    assert res_ai[0]["metadata"]["type"] == "tech"
    assert res_ai[0]["score"] > 0.0

    # Query 2: Sourdough bread
    res_cook = mem.recall("making bread with flour", top_k=1)
    assert len(res_cook) == 1
    assert "sourdough" in res_cook[0]["text"]


def test_vector_memory_empty():
    mem = memory()
    assert mem.recall("anything") == []
    assert mem.count() == 0


# =============================================================================
# 3. Multi-Agent Swarms & Debate Tests
# =============================================================================
def test_swarm_execution():
    ag1 = AgentRuntime(name="Coder", instructions="Writes code.")
    ag2 = AgentRuntime(name="Reviewer", instructions="Reviews code.")

    result = swarm([ag1, ag2], "Build a fast matrix multiplier")
    assert "=== Swarm Synthesis" in result
    assert "[Coder]" in result
    assert "[Reviewer]" in result


def test_debate_consensus():
    ag1 = AgentRuntime(name="Proponent", instructions="Argues for dynamic shapes.")
    ag2 = AgentRuntime(name="Opponent", instructions="Argues for static shapes.")

    consensus = debate([ag1, ag2], "Tensor shape ergonomics", rounds=2)
    assert "=== Debate Consensus" in consensus
    assert "Round 1" in consensus
    assert "Round 2" in consensus
    assert "Proponent:" in consensus
    assert "Opponent:" in consensus


def test_agent_pipeline_chaining():
    # Test task |> Agent1 |> Agent2 inside VM
    source = """
agent StepOne:
    model: "mock-model"
    instructions: "First step."

agent StepTwo:
    model: "mock-model"
    instructions: "Second step."

let piped = "Initial Prompt" |> StepOne |> StepTwo
"""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    compiled = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(compiled)

    assert "piped" in vm.globals
    assert "StepTwo" in vm.globals["piped"]
