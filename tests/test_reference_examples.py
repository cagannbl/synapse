"""
Synapse Reference Architecture Showcases Verification Test Suite
================================================================
Verifies syntax, AST parsing, bytecode compilation, transpilation, and VM execution
across all 3 Industrial Reference Architectures:
1. Enterprise RAG Service (examples/enterprise_rag_service/main.syn)
2. HPC Streaming Pipeline (examples/hpc_streaming_pipeline/main.syn)
3. Edge WebAssembly Inference (examples/edge_wasm_inference/main.syn)
"""

import os
import pytest

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast_nodes import Program, AgentDef, FunctionDef, VarDeclStmt
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine
from synapse.codegen.wasm_compiler import WasmCompiler
from synapse.core.diagnostics import diagnose_code
from synapse.tools.linter import SynapseLinter
from synapse.orm import Database
from synapse.core.memory import VectorMemory
from synapse.core.dataframe import DataFrame


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAG_PATH = os.path.join(BASE_DIR, "examples", "enterprise_rag_service", "main.syn")
HPC_PATH = os.path.join(BASE_DIR, "examples", "hpc_streaming_pipeline", "main.syn")
WASM_PATH = os.path.join(BASE_DIR, "examples", "edge_wasm_inference", "main.syn")


# =============================================================================
# 1. Enterprise RAG Service Verification Tests
# =============================================================================

def test_enterprise_rag_ast_parsing():
    """Verify syntax and AST structure of enterprise_rag_service/main.syn."""
    assert os.path.isfile(RAG_PATH), f"File not found: {RAG_PATH}"
    with open(RAG_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()

    assert isinstance(ast, Program)
    assert len(ast.statements) > 10

    # Verify Agent definition
    agent_defs = [s for s in ast.statements if isinstance(s, AgentDef)]
    assert len(agent_defs) == 1
    assert agent_defs[0].name == "EnterpriseRAGAgent"

    # Verify Route Handler functions
    fn_defs = {s.name: s for s in ast.statements if isinstance(s, FunctionDef)}
    assert "handle_health" in fn_defs
    assert "handle_documents" in fn_defs
    assert "handle_search" in fn_defs
    assert "handle_stream" in fn_defs
    assert "generate_rag_response_tokens" in fn_defs


def test_enterprise_rag_vm_execution():
    """Verify end-to-end VM execution of enterprise_rag_service/main.syn."""
    with open(RAG_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler(name="enterprise_rag_service").compile(ast)

    vm = VirtualMachine()
    vm.execute(code)

    # 1. Verify Relational Database State
    assert "db" in vm.globals
    db_inst = vm.globals["db"]
    assert isinstance(db_inst, Database)
    rows = db_inst.query("SELECT * FROM knowledge_catalog")
    assert len(rows) == 4

    # 2. Verify Vector Memory State
    assert "rag_brain" in vm.globals
    brain = vm.globals["rag_brain"]
    assert isinstance(brain, VectorMemory)
    assert brain.count() == 4

    # 3. Verify Route Handlers
    routes = vm.globals["routes"]
    assert "/api/health" in routes
    assert "/api/documents" in routes
    assert "/api/search" in routes
    assert "/api/stream" in routes

    # Invoke handle_health
    health_resp = vm.globals["handle_health"](vm)
    assert health_resp["status"] == "online"
    assert health_resp["safetensors_loaded"] is True
    assert health_resp["indexed_documents"] == 4

    # 4. Verify SafeTensors file creation
    assert os.path.isfile("enterprise_rag_weights.safetensors")


# =============================================================================
# 2. HPC Streaming Pipeline Verification Tests
# =============================================================================

def test_hpc_streaming_pipeline_ast_parsing():
    """Verify syntax and AST structure of hpc_streaming_pipeline/main.syn."""
    assert os.path.isfile(HPC_PATH), f"File not found: {HPC_PATH}"
    with open(HPC_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()

    assert isinstance(ast, Program)
    assert len(ast.statements) > 10

    # Verify routines
    fn_defs = {s.name: s for s in ast.statements if isinstance(s, FunctionDef)}
    assert "producer_routine" in fn_defs
    assert "detector_routine" in fn_defs


def test_hpc_streaming_pipeline_vm_execution():
    """Verify end-to-end VM execution of hpc_streaming_pipeline/main.syn."""
    with open(HPC_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler(name="hpc_streaming_pipeline").compile(ast)

    vm = VirtualMachine()
    vm.execute(code)

    # 1. Verify Arrow IPC file generated on disk
    assert os.path.isfile("market_telemetry.arrow")

    # 2. Verify LazyFrame pushdown results
    assert "anomaly_ticks" in vm.globals
    anomalies_df = vm.globals["anomaly_ticks"]
    assert isinstance(anomalies_df, DataFrame)
    assert len(anomalies_df) == 3
    assert set(anomalies_df.columns) == {"tick_id", "symbol", "spread", "volume"}

    # 3. Verify CSP channel worker execution
    assert vm.globals["prod_status"] == "PRODUCER_DONE"
    assert vm.globals["det_stats"]["processed"] == 5
    assert vm.globals["det_stats"]["anomalies"] == 2

    # 4. Verify Alert sink received high-spread alerts
    collected = vm.globals["collected_alerts"]
    assert len(collected) == 2
    for alert in collected:
        assert alert["sp"] >= 1.0


# =============================================================================
# 3. Edge WebAssembly Inference Verification Tests
# =============================================================================

def test_edge_wasm_inference_ast_parsing():
    """Verify syntax and AST structure of edge_wasm_inference/main.syn."""
    assert os.path.isfile(WASM_PATH), f"File not found: {WASM_PATH}"
    with open(WASM_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()

    assert isinstance(ast, Program)
    assert len(ast.statements) > 10

    # Verify functions and signatures
    fn_defs = {s.name: s for s in ast.statements if isinstance(s, FunctionDef)}
    assert "relu_activation" in fn_defs
    assert "quantize_feature" in fn_defs
    assert "dequantize_feature" in fn_defs
    assert "compute_linear_neuron" in fn_defs
    assert "predict_anomaly" in fn_defs

    assert fn_defs["relu_activation"].return_type == "f64"
    assert fn_defs["predict_anomaly"].return_type == "int"


def test_edge_wasm_inference_transpilation_and_wasm_compiler():
    """Verify C transpilation, function extraction, and HTML harness for WASM."""
    with open(WASM_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    wc = WasmCompiler()

    # 1. Transpilation to C99
    c_code = wc.transpile(source, filename=WASM_PATH)
    assert "synapse_runtime.h" in c_code
    assert "double relu_activation(double x)" in c_code
    assert "int quantize_feature(double val, double scale)" in c_code
    assert "int predict_anomaly(double score, double threshold)" in c_code
    assert "int main(" in c_code

    # 2. Function Signature Extraction for Emscripten Exports
    fn_names, sigs = WasmCompiler.extract_functions(source)
    assert "relu_activation" in fn_names
    assert "quantize_feature" in fn_names
    assert "compute_linear_neuron" in fn_names
    assert "predict_anomaly" in fn_names
    assert sigs["predict_anomaly"]["return_type"] == "int"

    # 3. HTML Companion Test Harness Generation
    html = wc.generate_html_harness("main.js", function_signatures=sigs, module_name="SynapseEdgeModule")
    assert "<!DOCTYPE html>" in html
    assert "SynapseEdgeModule" in html
    assert "predict_anomaly" in html

    # 4. VM Numerical Correctness
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler(name="edge_wasm_inference").compile(ast)
    vm = VirtualMachine()
    vm.execute(code)

    assert vm.globals["q_code"] == 98
    assert vm.globals["recovered_val"] == pytest.approx(2.45)
    assert len(vm.globals["completed_tasks"]) == 4


# =============================================================================
# 4. Cross-Showcase Diagnostics & README Verification Tests
# =============================================================================

def test_reference_examples_diagnostics_and_lint_clean():
    """Verify that all 3 reference examples have zero syntax errors and clean diagnostics."""
    linter = SynapseLinter()
    for file_path in [RAG_PATH, HPC_PATH, WASM_PATH]:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        diag = diagnose_code(source, file_path)
        assert diag.status == "ok", f"Diagnostic error in {file_path}: {diag.message}"

        lint_reports = linter.lint_file(file_path)
        errors = [r for r in lint_reports if r.severity == "error"]
        assert len(errors) == 0, f"Lint errors in {file_path}: {[e.message for e in errors]}"


def test_reference_examples_readme_documentation_completeness():
    """Verify that all 3 showcase examples have complete README.md files with diagrams."""
    rag_readme = os.path.join(BASE_DIR, "examples", "enterprise_rag_service", "README.md")
    hpc_readme = os.path.join(BASE_DIR, "examples", "hpc_streaming_pipeline", "README.md")
    wasm_readme = os.path.join(BASE_DIR, "examples", "edge_wasm_inference", "README.md")

    for r_path in [rag_readme, hpc_readme, wasm_readme]:
        assert os.path.isfile(r_path), f"README missing at {r_path}"
        with open(r_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "```mermaid" in content, f"Mermaid architecture diagram missing in {r_path}"
        assert "synapse run" in content or "synapse build" in content
        assert len(content) > 500
