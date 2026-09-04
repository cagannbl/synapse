import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine
from synapse.ai.providers import reset_llm_provider


@pytest.fixture(autouse=True)
def clean_llm():
    reset_llm_provider()
    yield
    reset_llm_provider()


def test_researcher_critic_pipeline_with_memory():
    source = """
agent Researcher:
    model: "mock-model"
    instructions: "You research AI advancements and reasoning models."

agent Critic:
    model: "mock-model"
    instructions: "You audit and review research findings for accuracy."

let topic = "Advancements in reasoning models and native vector memory"
let findings = topic |> Researcher |> Critic

let brain = memory()
brain.remember(findings, metadata={"status": "approved", "source": "pipeline"})

let results = brain.recall("reasoning models vector memory", top_k=1)
"""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(code)

    assert "findings" in vm.globals
    assert "Critic" in vm.globals["findings"]
    assert "Researcher" in vm.globals["findings"]
    assert vm.globals["brain"].count() == 1
    assert len(vm.globals["results"]) == 1
    assert vm.globals["results"][0]["metadata"]["status"] == "approved"
    assert vm.globals["results"][0]["score"] > 0.0
