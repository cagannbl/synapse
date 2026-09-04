import os
import subprocess
import sys
import pytest

from synapse.compiler.options import (
    CompilationProfile,
    CompilationMode,
    CompilerOptions,
    StandaloneViolationError,
    E_INTEROP_STANDALONE_VIOLATION,
    check_profile_compliance,
    heal_and_execute,
)
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser, ParseError
from synapse.parser import parse_source
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def run_cli(*args):
    """Executes the Synapse CLI command via subprocess and returns CompletedProcess."""
    cmd = [sys.executable, "-m", "synapse.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


# ============================================================================
# 1. Enums & Options Tests
# ============================================================================

def test_compilation_profile_enum():
    """Verify CompilationProfile enum values, string equality, and parsing."""
    assert CompilationProfile.STANDALONE == "standalone"
    assert CompilationProfile.HYBRID == "hybrid"
    assert CompilationProfile.from_string("standalone") == CompilationProfile.STANDALONE
    assert CompilationProfile.from_string("hybrid") == CompilationProfile.HYBRID
    assert CompilationProfile.from_string("STD") == CompilationProfile.STANDALONE

    with pytest.raises(ValueError, match="Unknown CompilationProfile"):
        CompilationProfile.from_string("invalid_profile")


def test_compilation_mode_enum():
    """Verify CompilationMode enum values, string equality, and parsing."""
    assert CompilationMode.STRICT == "strict"
    assert CompilationMode.TOLERANT == "tolerant"
    assert CompilationMode.from_string("strict") == CompilationMode.STRICT
    assert CompilationMode.from_string("tolerant") == CompilationMode.TOLERANT
    assert CompilationMode.from_string("AI_TOLERANT") == CompilationMode.TOLERANT

    with pytest.raises(ValueError, match="Unknown CompilationMode"):
        CompilationMode.from_string("invalid_mode")


def test_compiler_options_defaults_and_conversion():
    """Verify CompilerOptions defaults and string-to-enum conversions."""
    default_opts = CompilerOptions()
    assert default_opts.profile == CompilationProfile.STANDALONE
    assert default_opts.mode == CompilationMode.STRICT

    custom_opts = CompilerOptions(profile="hybrid", mode="tolerant")
    assert custom_opts.profile == CompilationProfile.HYBRID
    assert custom_opts.mode == CompilationMode.TOLERANT


# ============================================================================
# 2. Profile Compliance Check Tests
# ============================================================================

def test_check_profile_compliance_standalone_blocks_direct_python_import():
    """Ensure check_profile_compliance raises StandaloneViolationError on top-level Python import."""
    code = "import py.math\nlet x = 10\n"
    ast = parse_source(code, filename="test_standalone.syn")

    with pytest.raises(StandaloneViolationError) as exc_info:
        check_profile_compliance(ast, CompilationProfile.STANDALONE)

    err = exc_info.value
    assert "py.math" in str(err)
    assert err.error_code == "E_INTEROP_STANDALONE_VIOLATION"
    assert err.module == "math"
    assert err.line == 1


def test_check_profile_compliance_standalone_blocks_nested_python_import():
    """Ensure check_profile_compliance raises StandaloneViolationError inside functions/blocks."""
    code = """fn compute(a: int) -> int:
    if a > 0:
        import py.os
        return 1
    return 0
"""
    ast = parse_source(code, filename="test_nested.syn")

    with pytest.raises(StandaloneViolationError) as exc_info:
        check_profile_compliance(ast, CompilationProfile.STANDALONE)

    err = exc_info.value
    assert "py.os" in str(err)
    assert err.error_code == "E_INTEROP_STANDALONE_VIOLATION"
    assert err.module == "os"
    assert err.line == 3


def test_check_profile_compliance_hybrid_permits_python_import():
    """Ensure check_profile_compliance allows Python imports when profile is HYBRID."""
    code = "import py.math\nimport py.numpy as np\nlet pi = py.math.pi\n"
    ast = parse_source(code, filename="test_hybrid.syn")

    assert check_profile_compliance(ast, CompilationProfile.HYBRID) is True
    assert check_profile_compliance(ast, "hybrid") is True


def test_compiler_with_standalone_profile_rejects_interop():
    """Ensure Compiler.compile() raises StandaloneViolationError under STANDALONE profile."""
    code = "import py.math\nlet r = py.math.sqrt(4.0)\n"
    ast = parse_source(code, filename="test_compile_fail.syn")

    compiler = Compiler(
        name="test_fail",
        options=CompilerOptions(profile=CompilationProfile.STANDALONE),
    )
    with pytest.raises(StandaloneViolationError):
        compiler.compile(ast)


def test_compiler_with_hybrid_profile_executes_successfully():
    """Ensure Compiler with HYBRID profile compiles and VM executes Python interop."""
    code = "import py.math\nlet val = py.math.sqrt(25.0)\nprint(val)\n"
    ast = parse_source(code, filename="test_compile_success.syn")

    compiler = Compiler(
        name="test_success",
        options=CompilerOptions(profile=CompilationProfile.HYBRID),
    )
    code_obj = compiler.compile(ast)
    vm = VirtualMachine()
    vm.set_custom_print()
    vm.execute(code_obj)
    assert any("5.0" in out for out in vm.output_buffer)


# ============================================================================
# 3. Determinism & Strict vs Tolerant Mode Tests
# ============================================================================

def test_strict_mode_rejects_python_syntax_drift():
    """In STRICT mode, syntax drift (e.g. missing block colon) is directly rejected."""
    drift_code = "fn calculate(a, b)\n    return a + b\n"

    # Parsing directly without self-healing raises ParseError
    with pytest.raises(ParseError):
        tokens = Lexer(drift_code).tokenize()
        Parser(tokens).parse()


def test_tolerant_mode_auto_heals_syntax_drift():
    """In TOLERANT mode, syntax drift is repaired by fix_ai_drift before compilation."""
    from synapse.core.diagnostics import fix_ai_drift

    drift_code = "fn calculate(a, b)\n    return a + b\nlet result = calculate(10, 20)\nprint(result)\n"
    healed_code, diff, changes = fix_ai_drift(drift_code)

    assert "fn calculate(a, b):" in healed_code
    assert len(changes) > 0
    assert "+fn calculate(a, b):" in diff

    # Now healed code parses and executes cleanly
    ast = parse_source(healed_code, filename="healed.syn")
    code_obj = Compiler(name="healed").compile(ast)
    vm = VirtualMachine()
    vm.set_custom_print()
    vm.execute(code_obj)
    assert any("30" in out for out in vm.output_buffer)


# ============================================================================
# 4. CLI Commands Tests: run, build, fix
# ============================================================================

def test_cli_run_standalone_vs_hybrid_flags(tmp_path):
    """Test synapse run --profile standalone rejects and --profile hybrid allows py imports."""
    syn_file = str(tmp_path / "interop_test.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("import py.math\nlet val = py.math.floor(4.9)\nprint(val)\n")

    # 1. Explicit standalone -> must fail
    res_standalone = run_cli("run", syn_file, "--profile", "standalone")
    assert res_standalone.returncode != 0
    assert "Profile Error" in res_standalone.stderr or "E_INTEROP_STANDALONE_VIOLATION" in res_standalone.stderr

    # 2. Explicit hybrid -> must succeed
    res_hybrid = run_cli("run", syn_file, "--profile", "hybrid")
    assert res_hybrid.returncode == 0
    assert "4" in res_hybrid.stdout


def test_cli_run_default_backward_compatibility(tmp_path):
    """Ensure omitting --profile gracefully falls back to hybrid when import py is used."""
    syn_file = str(tmp_path / "compat_test.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("import py.math\nlet x = py.math.ceil(2.1)\nprint(x)\n")

    # Without explicit --profile, existing tests & workflows work seamlessly
    res = run_cli("run", syn_file)
    assert res.returncode == 0
    assert "3" in res.stdout


def test_cli_run_mode_strict_vs_tolerant(tmp_path):
    """Test synapse run --mode strict rejects drift while --mode tolerant auto-heals."""
    drift_file = str(tmp_path / "drift_test.syn")
    with open(drift_file, "w", encoding="utf-8") as f:
        f.write("fn add_nums(x, y)\n    return x + y\nlet res = add_nums(15, 25)\nprint(res)\n")

    # 1. Strict mode (default) -> fails on missing colon
    res_strict = run_cli("run", drift_file, "--mode", "strict")
    assert res_strict.returncode != 0
    assert "Syntax Error" in res_strict.stderr

    # 2. Tolerant mode -> heals missing colon and succeeds
    res_tolerant = run_cli("run", drift_file, "--mode", "tolerant")
    assert res_tolerant.returncode == 0
    assert "40" in res_tolerant.stdout


def test_cli_build_profile_isolation(tmp_path):
    """Test synapse build --c-only enforces profile isolation."""
    syn_file = str(tmp_path / "build_interop.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("import py.math\nfn run():\n    return 42\n")

    # Standalone build -> blocked
    res_standalone = run_cli("build", syn_file, "--c-only", "--profile", "standalone")
    assert res_standalone.returncode != 0
    assert "Profile Error" in res_standalone.stderr or "E_INTEROP_STANDALONE_VIOLATION" in res_standalone.stderr

    # Hybrid build -> succeeds
    res_hybrid = run_cli("build", syn_file, "--c-only", "--profile", "hybrid")
    assert res_hybrid.returncode == 0
    assert "C source emitted:" in res_hybrid.stdout


def test_cli_fix_command_repairs_file_in_place(tmp_path):
    """Test synapse fix <file.syn> repairs syntax drift in-place."""
    target_file = str(tmp_path / "code_to_fix.syn")
    with open(target_file, "w", encoding="utf-8") as f:
        f.write("def double(n):\n    return n * 2\n")

    res = run_cli("fix", target_file)
    assert res.returncode == 0
    assert "Fixed syntax and style drifts" in res.stdout

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "fn double(n):" in content
    assert "def double" not in content


def test_cli_fix_command_diff_mode_preserves_file(tmp_path):
    """Test synapse fix <file.syn> --diff displays unified diff without altering file."""
    target_file = str(tmp_path / "diff_only.syn")
    original_content = "def multiply(a, b):\n    return a * b\n"
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(original_content)

    res = run_cli("fix", target_file, "--diff")
    assert res.returncode == 0
    assert "+fn multiply" in res.stdout or "-def multiply" in res.stdout

    # File on disk must remain unwritten
    with open(target_file, "r", encoding="utf-8") as f:
        after_content = f.read()
    assert after_content == original_content


def test_heal_and_execute_backward_compatibility():
    """Verify heal_and_execute API remains functional across packages."""
    from synapse.core.diagnostics import heal_and_execute as diag_heal
    from synapse.compiler import heal_and_execute as comp_heal

    assert diag_heal is not None
    assert comp_heal is not None

    test_code = "def sample():\n    return 99\nlet r = sample()\nprint(r)\n"
    vm = VirtualMachine()
    vm.set_custom_print()
    res, report = comp_heal(test_code, vm=vm, filename="compat_heal.syn")

    assert report.status == "ok"
    assert report.auto_fixed_code is not None
    assert "fn sample():" in report.auto_fixed_code
    assert any("99" in out for out in vm.output_buffer)
