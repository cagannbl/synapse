"""
Tests for Synapse Automated Documentation Generator (synapse-docgen).
Phase 5: API reference building, AST docstring extraction, tensor contracts, and coverage audits.
"""
import os
import sys
import tempfile
import pytest
from pathlib import Path

from synapse.tools.docgen import (
    SynapseDocGenerator,
    ModuleDoc,
    FunctionDoc,
    ParamDoc,
    DocCoverageReport,
    main as docgen_main,
)


SAMPLE_SYN_SOURCE = '''"""
Linear Algebra & Deep Learning Core Utility Module.
Provides high-performance matrix operations and neural network building blocks.
"""

# Mathematical constant PI
const PI: float = 3.141592653589793

enum ActivationType:
    ReLU
    GELU
    Sigmoid

struct HyperParameters:
    learning_rate: float
    batch_size: int
    dropout: float

fn matmul_forward(A: Tensor[32, 64], B: Tensor[64, 128]) -> Tensor[32, 128]:
    """
    Computes batched matrix multiplication with SIMD acceleration.
    Args:
        A: Left input activation tensor [32, 64]
        B: Weight parameter tensor [64, 128]
    Returns:
        Tensor[32, 128]: Projected tensor
    """
    return A @ B

fn relu_activation(x: Tensor[32, 128]) -> Tensor[32, 128]:
    # In-place ReLU activation function
    return x

fn undocumented_helper(a: int, b: int) -> int:
    return a + b

prompt CodeReviewer(code_snippet: str) -> str:
    system: "You are an expert compiler engineer."
    user: code_snippet

agent MathSpecialist:
    name: "Euler"
    model: "synapse-deep-v1"
'''


@pytest.fixture
def temp_syn_file():
    with tempfile.NamedTemporaryFile(suffix=".syn", mode="w", encoding="utf-8", delete=False) as f:
        f.write(SAMPLE_SYN_SOURCE)
        path = f.name
    yield Path(path)
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_syn_dir():
    temp_dir = tempfile.TemporaryDirectory()
    dir_path = Path(temp_dir.name)

    # Sub-file 1
    (dir_path / "math_ops.syn").write_text(SAMPLE_SYN_SOURCE, encoding="utf-8")
    
    # Sub-file 2 in a nested directory
    sub_dir = dir_path / "models"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "transformer.syn").write_text('''"""
Transformer Attention Block.
"""
fn attention(Q: Tensor[B, S, D], K: Tensor[B, S, D], V: Tensor[B, S, D]) -> Tensor[B, S, D]:
    """Computes scaled dot-product attention."""
    return Q
''', encoding="utf-8")

    yield dir_path
    temp_dir.cleanup()


# =============================================================================
# 1. Single File AST Extraction Tests
# =============================================================================

def test_docgen_extract_module_doc(temp_syn_file):
    """Verify that module-level docstring and file path are parsed."""
    gen = SynapseDocGenerator(file_path=temp_syn_file)
    modules = gen.parse()
    assert len(modules) == 1

    mod = modules[0]
    assert mod.filepath == str(temp_syn_file)
    assert "Linear Algebra & Deep Learning Core" in mod.module_docstring


def test_docgen_extract_functions_and_signatures(temp_syn_file):
    """Verify extraction of function signatures and parameters."""
    gen = SynapseDocGenerator(file_path=temp_syn_file)
    modules = gen.parse()
    mod = modules[0]

    fn_names = [f.name for f in mod.functions]
    assert "matmul_forward" in fn_names
    assert "relu_activation" in fn_names
    assert "undocumented_helper" in fn_names

    matmul_fn = next(f for f in mod.functions if f.name == "matmul_forward")
    assert "Computes batched matrix multiplication" in matmul_fn.docstring
    assert len(matmul_fn.params) == 2
    assert matmul_fn.params[0].name == "A"
    assert "Tensor[32, 64]" in str(matmul_fn.params[0].type_annot)


def test_docgen_extract_tensor_contracts(temp_syn_file):
    """Verify that tensor shape contracts are correctly recognized on inputs and returns."""
    gen = SynapseDocGenerator(file_path=temp_syn_file)
    modules = gen.parse()
    mod = modules[0]

    matmul_fn = next(f for f in mod.functions if f.name == "matmul_forward")
    assert matmul_fn.is_tensor_return is True
    assert "Tensor[32, 128]" in str(matmul_fn.return_tensor_contract)

    param_a = matmul_fn.params[0]
    assert param_a.is_tensor is True
    assert param_a.tensor_dims == (32, 64)


def test_docgen_extract_enums_structs_prompts_agents(temp_syn_file):
    """Verify extraction of enums, structs, prompts, and agents."""
    gen = SynapseDocGenerator(file_path=temp_syn_file)
    modules = gen.parse()
    mod = modules[0]

    assert len(mod.enums) == 1
    assert mod.enums[0].name == "ActivationType"
    assert "ReLU" in mod.enums[0].variants

    assert len(mod.structs) == 1
    assert mod.structs[0].name == "HyperParameters"
    field_names = [f.name for f in mod.structs[0].fields]
    assert "learning_rate" in field_names
    assert "batch_size" in field_names

    assert len(mod.prompts) == 1
    assert mod.prompts[0].name == "CodeReviewer"

    assert len(mod.agents) == 1
    assert mod.agents[0].name == "MathSpecialist"


# =============================================================================
# 2. Markdown & HTML Output Generation Tests
# =============================================================================

def test_docgen_generate_markdown(temp_syn_file):
    """Verify Markdown generation contains table of contents, headers, and code blocks."""
    gen = SynapseDocGenerator(file_path=temp_syn_file)
    md = gen.generate_markdown()

    assert "matmul_forward" in md
    assert "Tensor[32, 64]" in md
    assert "ActivationType" in md
    assert "HyperParameters" in md


def test_docgen_generate_html(temp_syn_file):
    """Verify responsive HTML generation contains styled sections."""
    gen = SynapseDocGenerator(file_path=temp_syn_file)
    html_out = gen.generate_html()

    assert "<!DOCTYPE html>" in html_out
    assert "matmul_forward" in html_out
    assert "Tensor[32, 128]" in html_out
    assert "HyperParameters" in html_out


# =============================================================================
# 3. Directory Recursive Scanning & Coverage Report Tests
# =============================================================================

def test_docgen_recursive_directory_scan(temp_syn_dir):
    """Verify that multiple files across nested directories are scanned."""
    gen = SynapseDocGenerator(dir_path=temp_syn_dir, recursive=True)
    modules = gen.parse()

    assert len(modules) == 2
    mod_files = [Path(m.filepath).name for m in modules]
    assert "math_ops.syn" in mod_files
    assert "transformer.syn" in mod_files


def test_docgen_coverage_report(temp_syn_file):
    """Verify documentation and type contract coverage auditing."""
    gen = SynapseDocGenerator(file_path=temp_syn_file)
    cov: DocCoverageReport = gen.get_coverage_report()

    assert cov.total_functions == 3
    assert cov.documented_functions >= 1
    assert cov.typed_functions == 3
    assert cov.docstring_coverage > 0.0

    summary = cov.format_summary()
    assert "Synapse Documentation & Type Audit Report" in summary


# =============================================================================
# 4. CLI Entry Point Integration Tests
# =============================================================================

def test_docgen_cli_run(temp_syn_file, tmp_path):
    """Verify running docgen via main() creates output file."""
    out_md = tmp_path / "output_test.md"
    ret = docgen_main([str(temp_syn_file), "-o", str(out_md)])
    assert ret == 0
    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "matmul_forward" in content
