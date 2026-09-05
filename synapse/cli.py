import sys
import os
import argparse
from typing import Optional
from synapse import __version__
from synapse.lexer.lexer import Lexer, LexerError
from synapse.parser.parser import Parser, ParseError
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine, VMRuntimeError


def print_banner():
    print(fr"""
  ___                                     
 / __|_  _ _ _  __ _ _ __  ___ ___        
 \__ \ || | ' \/ _` | '_ \(_-</ -_)       
 |___/\_, |_||_\__,_| .__/__/\___| v{__version__}
      |__/          |_|                   
 Synapse AI-Native Programming Language REPL
 Type 'exit()' or press Ctrl+C / Ctrl+Z to quit.
""")


def _is_flag_explicit(flag: str) -> bool:
    """Checks whether a command-line flag was explicitly passed in sys.argv."""
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv)


def run_file(
    filepath: str,
    json_diagnostics: bool = False,
    ai_tolerant: bool = False,
    no_cache: bool = False,
    profile: Optional[str] = None,
    mode: Optional[str] = None,
    explicit_profile: bool = False,
    compat: Optional[str] = None,
    vm_mode: bool = False,
    output_format: str = "human",
    is_agent: bool = False,
):
    use_agentic_json = (output_format == "json") or is_agent
    from synapse.core.diagnostics import Diagnostic, DiagnosticReport, diagnose_code

    if not os.path.isfile(filepath):
        if use_agentic_json:
            diag = Diagnostic(
                file=filepath,
                line=1,
                column=1,
                severity="error",
                code="SYN-E001",
                message=f"File not found '{filepath}'",
            )
            print(DiagnosticReport(status="error", diagnostics=[diag]).to_json(indent=2))
        elif json_diagnostics:
            import json
            print(json.dumps({"status": "error", "error_type": "FileNotFound", "message": f"File not found '{filepath}'"}))
        else:
            print(f"Error: File not found '{filepath}'", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    is_python = (compat == "python") or filepath.endswith(".py")
    if is_python:
        from synapse.tools.migrator import migrate_code
        source = migrate_code(source)
        if not explicit_profile:
            profile = "hybrid"

    is_tolerant = ai_tolerant or (mode == "tolerant")
    if is_tolerant:
        from synapse.core.diagnostics import fix_ai_drift
        source, _, _ = fix_ai_drift(source, filename=filepath)

    try:
        from synapse.parser import parse_source
        ast = parse_source(source, filename=filepath, use_cache=not no_cache)

        from synapse.compiler.options import (
            CompilationProfile,
            StandaloneViolationError,
            check_profile_compliance,
            _find_all_import_stmts,
        )

        selected_profile: CompilationProfile
        if explicit_profile and profile:
            selected_profile = CompilationProfile.from_string(profile)
        elif profile == "hybrid":
            selected_profile = CompilationProfile.HYBRID
        elif profile == "standalone" and explicit_profile:
            selected_profile = CompilationProfile.STANDALONE
        else:
            # Backward-compatible automatic fallback if profile not explicitly specified:
            imports = _find_all_import_stmts(ast)
            has_py = any(
                getattr(imp, "is_python", False) or (getattr(imp, "module_path", None) and imp.module_path[0] == "py")
                for imp in imports
            )
            selected_profile = CompilationProfile.HYBRID if has_py else CompilationProfile.STANDALONE

        check_profile_compliance(ast, selected_profile)

        # C99 AOT Birincil Motor: Standalone modunda C99 olarak derleyip çalıştır
        if not vm_mode and selected_profile == CompilationProfile.STANDALONE and not is_python and not use_agentic_json:
            try:
                from synapse.codegen.native_compiler import NativeCompiler
                native_comp = NativeCompiler()
                comp_info = native_comp.find_c_compiler()
                if comp_info:
                    ok, retcode, c_out, c_err = native_comp.compile_and_run(filepath, no_cache=no_cache)
                    if ok:
                        if c_out:
                            sys.stdout.write(c_out)
                        if c_err:
                            sys.stderr.write(c_err)
                        sys.exit(retcode)
            except Exception:
                pass

        code = Compiler(name=os.path.basename(filepath), profile=selected_profile).compile(ast)
        vm = VirtualMachine()
        if use_agentic_json:
            import io
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                vm.execute(code)
            print(DiagnosticReport(status="ok", diagnostics=[]).to_json(indent=2))
            sys.exit(0)
        else:
            vm.execute(code)
    except Exception as e:
        from synapse.compiler.options import StandaloneViolationError
        if use_agentic_json:
            report = diagnose_code(source, filepath=filepath, error=e)
            if report.status == "ok":
                diag = Diagnostic(
                    file=filepath,
                    line=1,
                    column=1,
                    severity="error",
                    code="SYN-E001",
                    message=str(e),
                )
                report.status = "error"
                report.diagnostics.append(diag)
                report.error_type = type(e).__name__
                report.message = str(e)
                report.ai_prompt_hint = f"Runtime execution failed: {e}"
            print(report.to_json(indent=2))
            sys.exit(1)
        elif json_diagnostics:
            report = diagnose_code(source, filepath)
            if report.status == "ok":
                report.status = "error"
                report.error_type = type(e).__name__
                report.message = str(e)
                report.ai_prompt_hint = f"Runtime execution failed: {e}"
            print(report.to_json())
            sys.exit(1)
        else:
            if isinstance(e, StandaloneViolationError):
                print(f"\n[Profile Error] {e}", file=sys.stderr)
            elif isinstance(e, (LexerError, ParseError)):
                print(f"\n[Syntax Error] {e}", file=sys.stderr)
            elif isinstance(e, VMRuntimeError):
                print(f"\n[Runtime Error] {e}", file=sys.stderr)
            else:
                print(f"\n[Fatal Error] {e}", file=sys.stderr)
            sys.exit(1)


def fix_file(filepath: str, output: Optional[str] = None, show_diff: bool = False):
    if not os.path.isfile(filepath):
        print(f"Error: File not found '{filepath}'", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    from synapse.core.diagnostics import fix_ai_drift
    fixed_code, diff, changes = fix_ai_drift(source, filename=filepath)

    if show_diff:
        if diff:
            print(diff)
        else:
            print(f"No changes needed for '{filepath}'. Code is already clean.")
        return

    dest = output or filepath
    with open(dest, "w", encoding="utf-8") as f:
        f.write(fixed_code)

    if changes:
        print(f"Fixed syntax and style drifts in '{filepath}' -> '{dest}' ({len(changes)} changes applied):")
        for c in changes:
            print(f"  * {c}")
    else:
        print(f"'{filepath}' is already compliant with Synapse standards. No changes made.")


def fmt_file(filepath: str, output: Optional[str] = None, show_diff: bool = False):
    if not os.path.isfile(filepath):
        print(f"Error: File not found '{filepath}'", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    from synapse.core.diagnostics import fix_ai_drift
    fixed_code, diff, changes = fix_ai_drift(source, filename=filepath)

    if show_diff:
        if diff:
            print(diff)
        else:
            print(f"No changes needed for '{filepath}'. Code is already clean.")
        return

    dest = output or filepath
    with open(dest, "w", encoding="utf-8") as f:
        f.write(fixed_code)

    if changes:
        print(f"Formatted and normalized '{filepath}' -> '{dest}' ({len(changes)} changes applied):")
        for c in changes:
            print(f"  * {c}")
    else:
        print(f"'{filepath}' is already compliant with Synapse standards. No changes made.")


def migrate_cli(filepath: str, output: Optional[str] = None, show_diff: bool = False):
    from synapse.tools.migrator import PythonToSynapseMigrator
    if not os.path.isfile(filepath):
        print(f"Error: File not found '{filepath}'", file=sys.stderr)
        sys.exit(1)

    migrator = PythonToSynapseMigrator()
    dest = output or os.path.splitext(filepath)[0] + ".syn"
    migrated_code = migrator.migrate_file(filepath, dest if not show_diff else None)

    if show_diff:
        import difflib
        with open(filepath, "r", encoding="utf-8") as f:
            orig = f.readlines()
        migrated_lines = [line + "\n" for line in migrated_code.splitlines()]
        diff = "".join(difflib.unified_diff(orig, migrated_lines, fromfile=filepath, tofile=dest))
        if diff:
            print(diff)
        else:
            print(f"No changes generated for '{filepath}'.")
        return

    print(f"Migrated Python code successfully: '{filepath}' -> '{dest}'")


def kernel_cli(subcommand: str, user: bool = True, custom_dir: Optional[str] = None):
    from synapse.tools.kernel import install_kernel_spec
    if subcommand == "install":
        dest = install_kernel_spec(user=user, dest_dir=custom_dir)
        print(f"[Kernel] Synapse Jupyter Kernel spec installed successfully: {dest}")
    else:
        print("Usage: synapse kernel install [--user] [--dir <path>]")


def show_prompt_hint():
    from synapse.mcp_server import SYSTEM_PROMPT_DENSE
    print(SYSTEM_PROMPT_DENSE.strip())


def generate_docs(filepath: str, output: Optional[str] = None, as_html: bool = False, coverage: bool = False, as_dir: bool = False):
    from pathlib import Path
    from synapse.tools.docgen import SynapseDocGenerator

    target = Path(filepath)
    is_dir = as_dir or target.is_dir()

    if not is_dir and not target.is_file():
        print(f"Error: Path not found '{filepath}'", file=sys.stderr)
        sys.exit(1)

    gen = SynapseDocGenerator(
        file_path=target if not is_dir else None,
        dir_path=target if is_dir else None,
        recursive=True,
    )

    if coverage:
        cov = gen.get_coverage_report()
        print(cov.format_summary())
        return

    ext = ".html" if as_html else ".md"
    if output:
        dest = Path(output)
    else:
        if is_dir:
            dest = target / f"API_REFERENCE{ext}"
        else:
            dest = target.with_suffix(ext)

    if as_html:
        gen.save_html(dest)
    else:
        gen.save_markdown(dest)

    print(f"Documentation generated successfully: {dest}")


def run_mcp():
    from synapse.mcp_server import create_mcp_server
    server = create_mcp_server()
    server.run(transport="stdio")


def check_file(
    filepath: str,
    as_json: bool = False,
    type_check: bool = True,
    output_format: str = "human",
    is_agent: bool = False,
):
    use_agentic_json = is_agent or (output_format == "json") or as_json
    from synapse.core.diagnostics import Diagnostic, DiagnosticReport, diagnose_code

    if not os.path.isfile(filepath):
        if use_agentic_json:
            diag = Diagnostic(
                file=filepath,
                line=1,
                column=1,
                severity="error",
                code="SYN-E001",
                message=f"File not found '{filepath}'",
            )
            print(DiagnosticReport(status="error", diagnostics=[diag]).to_json(indent=2))
        else:
            print(f"Error: File not found '{filepath}'", file=sys.stderr)
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    report = diagnose_code(source, filepath)

    if report.status != "ok":
        if use_agentic_json:
            print(report.to_json(indent=2))
        else:
            print(f"Check FAILED [{report.error_type}] at line {report.line}:{report.column}")
            if report.source_line:
                print(f"  > {report.source_line}")
                print(f"    {report.pointer}")
            print(f"Message: {report.message}")
            if report.suggested_fix:
                print(f"Suggested fix: {report.suggested_fix}")
            if report.ai_prompt_hint:
                print(f"AI Hint: {report.ai_prompt_hint}")
        sys.exit(1)

    if type_check:
        from synapse.core.type_checker import check_source
        tc_result = check_source(source, filepath=filepath)
        if not tc_result.is_valid:
            if use_agentic_json:
                all_diags = []
                for err_rep in tc_result.errors:
                    if err_rep.diagnostics:
                        all_diags.extend(err_rep.diagnostics)
                    else:
                        all_diags.append(Diagnostic(
                            file=err_rep.file or filepath,
                            line=err_rep.line or 1,
                            column=err_rep.column or 1,
                            severity=err_rep.severity or "error",
                            code=err_rep.code or "SYN-E201",
                            message=err_rep.message,
                            expected=err_rep.expected,
                            actual=err_rep.actual,
                            suggested_fix=err_rep.suggested_fix,
                            diff=err_rep.diff,
                        ))
                print(DiagnosticReport(status="error", diagnostics=all_diags).to_json(indent=2))
            else:
                err = tc_result.errors[0]
                print(f"Check FAILED [{err.error_type}] at line {err.line}:{err.column}")
                if err.source_line:
                    print(f"  > {err.source_line}")
                    print(f"    {err.pointer}")
                print(f"Message: {err.message}")
                if err.suggested_fix:
                    print(f"Suggested fix: {err.suggested_fix}")
            sys.exit(1)

    # Tensor shape static contract check via shape guard
    from synapse.analyzer.shape_guard import verify_shapes_in_file
    is_shape_valid, shape_errors = verify_shapes_in_file(filepath)
    if not is_shape_valid and shape_errors:
        if use_agentic_json:
            all_diags = []
            for se in shape_errors:
                all_diags.append(Diagnostic(
                    file=filepath,
                    line=se.line or 1,
                    column=se.column or 1,
                    severity="error",
                    code=getattr(se, "code", "SYN-E202") or "SYN-E202",
                    message=se.message,
                    expected=str(se.expected) if se.expected is not None else None,
                    actual=str(se.actual) if se.actual is not None else None,
                    suggested_fix=getattr(se, "suggested_fix", None) or "Transpose tensor with .T",
                    diff=getattr(se, "diff", None),
                ))
            print(DiagnosticReport(status="error", diagnostics=all_diags).to_json(indent=2))
        else:
            se = shape_errors[0]
            print(f"Check FAILED [ShapeMismatch] at line {se.line}:{se.column}")
            print(f"Message: {se.message}")
            if getattr(se, "suggested_fix", None):
                print(f"Suggested fix: {se.suggested_fix}")
        sys.exit(1)

    if use_agentic_json:
        print(DiagnosticReport(status="ok", diagnostics=[]).to_json(indent=2))
    else:
        print(f"Check PASSED: {filepath} is valid Synapse code.")
    sys.exit(0)


def run_lint(
    paths: list[str],
    as_json: bool = False,
    strict: bool = False,
    output_format: str = "human",
    is_agent: bool = False,
):
    from synapse.core.diagnostics import Diagnostic, DiagnosticReport
    from synapse.tools.linter import SynapseLinter
    linter = SynapseLinter()
    all_diagnostics = []

    use_agentic = is_agent or (output_format == "json")

    if not paths:
        paths = ["."]

    target_files = []
    for p in paths:
        if os.path.isfile(p):
            target_files.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".syn"):
                        target_files.append(os.path.join(root, f))
        else:
            if use_agentic:
                diag = Diagnostic(
                    file=p,
                    line=1,
                    column=1,
                    severity="error",
                    code="SYN-E001",
                    message=f"Path not found '{p}'",
                )
                print(DiagnosticReport(status="error", diagnostics=[diag]).to_json(indent=2))
            elif as_json:
                import json
                print(json.dumps([{"code": "SYN999", "message": f"Path not found '{p}'", "line": 1, "column": 1, "severity": "error", "filename": p}], indent=2))
            else:
                print(f"Error: Path not found '{p}'", file=sys.stderr)
            sys.exit(1)

    if not target_files:
        if use_agentic:
            print(DiagnosticReport(status="ok", diagnostics=[]).to_json(indent=2))
        elif as_json:
            print("[]")
        else:
            print("No .syn files found to lint.")
        sys.exit(0)

    for filepath in sorted(target_files):
        diags = linter.lint_file(filepath)
        all_diagnostics.extend(diags)

    if use_agentic:
        diags = []
        for d in all_diagnostics:
            diags.append(Diagnostic(
                file=d.filename,
                line=d.line,
                column=d.column,
                severity=d.severity,
                code=d.code,
                message=d.message,
            ))
        has_errors = any(d.severity == "error" for d in all_diagnostics)
        status = "error" if has_errors or (strict and all_diagnostics) else "ok"
        report = DiagnosticReport(status=status, diagnostics=diags)
        print(report.to_json(indent=2))
        if strict and all_diagnostics:
            sys.exit(1)
        sys.exit(1 if has_errors else 0)
    else:
        report = linter.format_report(all_diagnostics, as_json=as_json)
        print(report)
        has_errors = any(d.severity == "error" for d in all_diagnostics)
        if strict and all_diagnostics:
            sys.exit(1)
        sys.exit(1 if has_errors else 0)


def show_ast(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    import pprint
    pprint.pprint(ast)


def show_disassembly(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler(name=os.path.basename(filepath)).compile(ast)

    print(f"Disassembly of <{code.name}>:")
    print(f"  Constants ({len(code.constants)}): {code.constants}")
    print(f"  Names     ({len(code.names)}): {code.names}")
    print(f"  Params    ({len(code.params)}): {code.params}")
    print("  Instructions:")
    for i, (op, arg) in enumerate(code.instructions):
        arg_str = f"({arg})" if arg is not None else ""
        print(f"    {i:04d}  {op.name:<18} {arg_str}")


def _sanitize_repl_input(line: str) -> str:
    import re
    cleaned = line.strip()
    # Prompt öneklerini (synapse>, >>>, $) otomatik temizle (Self-healing input)
    cleaned = re.sub(r"^(?:(?:synapse)?\s*>\s*|>>>\s*|\$\s*)+", "", cleaned)
    return cleaned.strip()


def run_repl():
    print_banner()
    vm = VirtualMachine()
    import re

    while True:
        try:
            raw_line = input("synapse> ")
            line = _sanitize_repl_input(raw_line)
            if not line:
                continue
            if line.lower() in ("exit()", "quit()", "exit", "quit", ":q", "q()", ":quit"):
                print("Exiting Synapse REPL.")
                break

            # Çok satırlı blok tespiti (eğer ':' ile bitiyorsa)
            lines = [line]
            if line.endswith(":"):
                while True:
                    sub_line = input("...      ")
                    sub_cleaned = re.sub(r"^(?:\.\.\.\s*)+", "", sub_line)
                    if not sub_cleaned.strip() and not sub_line.startswith(" "):
                        break
                    lines.append(sub_cleaned)
                    if not sub_line.startswith(" "):
                        break

            full_code = "\n".join(lines)
            tokens = Lexer(full_code).tokenize()
            ast = Parser(tokens).parse()
            code = Compiler(name="<repl>", repl_mode=True).compile(ast)
            vm.execute(code)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Synapse REPL.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(prog="synapse", description="Synapse AI Programming Language CLI")
    parser.add_argument("-v", "--version", action="version", version=f"Synapse {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # Run
    run_parser = subparsers.add_parser("run", help="Run a Synapse (.syn) file")
    run_parser.add_argument("file", help="Path to Synapse file")
    run_parser.add_argument("--json-diagnostics", action="store_true", help="Output machine-readable JSON diagnostics on error for AI assistants")
    run_parser.add_argument("--ai-tolerant", action="store_true", help="Tolerate minor Python syntax drift and auto-normalize at runtime")
    run_parser.add_argument("--no-cache", action="store_true", help="Bypass and do not use the AST compilation disk/memory cache")
    run_parser.add_argument("--profile", choices=["standalone", "hybrid"], default="standalone", help="Execution profile: 'standalone' (pure Synapse, default) or 'hybrid' (Python interop allowed)")
    run_parser.add_argument("--mode", choices=["strict", "tolerant"], default="strict", help="Compilation mode: 'strict' (default) or 'tolerant'")
    run_parser.add_argument("--compat", choices=["synapse", "python"], default="synapse", help="Language compatibility mode: 'synapse' (default) or 'python' (.py direct execution)")
    run_parser.add_argument("--vm", action="store_true", help="Force execution using Bytecode VM instead of C99 AOT")
    run_parser.add_argument("--format", choices=["human", "json"], default="human", help="Output format: 'human' (default) or 'json'")
    run_parser.add_argument("--agent", action="store_true", help="Output structured agentic diagnostics JSON format for autonomous LLM agents")

    # Test
    test_parser = subparsers.add_parser('test', help='Run Synapse test suite (.syn test files)', description='Run Synapse test suite (.syn test files)')
    test_parser.add_argument('path', nargs='?', default='tests', help='Directory or file to run tests from')
    test_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    # Fix (Syntax and Style Drift Repair)
    fix_parser = subparsers.add_parser("fix", help="Automatically repair syntax and style drifts in Synapse source code")
    fix_parser.add_argument("file", help="Path to Synapse file")
    fix_parser.add_argument("-o", "--output", help="Output to specified file instead of in-place")
    fix_parser.add_argument("--diff", action="store_true", help="Display unified diff of changes without modifying file")

    # Format & AI Drift Repair (fmt)
    fmt_parser = subparsers.add_parser("fmt", help="Format Synapse code and fix Python/AI drift (def->fn, missing let, numpy imports)")
    fmt_parser.add_argument("file", help="Path to Synapse file")
    fmt_parser.add_argument("-o", "--output", help="Output to specified file instead of in-place")
    fmt_parser.add_argument("--diff", action="store_true", help="Display unified diff of changes without modifying file")

    # Doc (API Documentation Generator)
    doc_parser = subparsers.add_parser("doc", help="Generate Markdown or HTML API documentation from Synapse (.syn) files")
    doc_parser.add_argument("file", help="Path to Synapse file or directory")
    doc_parser.add_argument("-o", "--output", help="Output file path (default: <filename>.md or <filename>.html)")
    doc_parser.add_argument("--html", action="store_true", help="Generate modern HTML documentation instead of Markdown")
    doc_parser.add_argument("--coverage", action="store_true", help="Audit docstring and static type contract coverage")
    doc_parser.add_argument("--dir", action="store_true", help="Process directory recursively")

    # Check (Lint & Syntax Diagnosis)
    check_parser = subparsers.add_parser("check", help="Validate Synapse code syntax and produce diagnostic report")
    check_parser.add_argument("file", help="Path to Synapse file")
    check_parser.add_argument("--json", action="store_true", help="Output diagnostic report in JSON format")
    check_parser.add_argument("--format", choices=["human", "json"], default="human", help="Output format: 'human' (default) or 'json'")
    check_parser.add_argument("--agent", action="store_true", help="Output structured agentic diagnostics JSON format for autonomous LLM agents")

    # Lint (Synapse Linter & Static Analysis)
    lint_parser = subparsers.add_parser("lint", help="Lint Synapse source files for warnings and syntax issues")
    lint_parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to lint (default: current directory)")
    lint_parser.add_argument("--json", action="store_true", help="Output diagnostic report in JSON format")
    lint_parser.add_argument("--strict", action="store_true", help="Treat warnings as errors (exit non-zero on any issues)")
    lint_parser.add_argument("--format", choices=["human", "json"], default="human", help="Output format: 'human' (default) or 'json'")
    lint_parser.add_argument("--agent", action="store_true", help="Output structured agentic diagnostics JSON format for autonomous LLM agents")

    # Prompt Hint (<250 token AI specification)
    subparsers.add_parser("prompt-hint", help="Display ultra-dense token-optimized system prompt for AI code generators")

    # Native MCP Server (Model Context Protocol)
    subparsers.add_parser("mcp", help="Run native Synapse Model Context Protocol (MCP) server over stdio")

    # REPL
    subparsers.add_parser("repl", help="Start interactive Synapse REPL")

    # AST
    ast_parser = subparsers.add_parser("ast", help="Display AST for a file")
    ast_parser.add_argument("file", help="Path to Synapse file")

    # Disassembly
    dis_parser = subparsers.add_parser("dis", help="Disassemble Synapse bytecode")
    dis_parser.add_argument("file", help="Path to Synapse file")

    # Emit-C (Transpile to C)
    emit_c_parser = subparsers.add_parser("emit-c", help="Transpile Synapse file to Pure C")
    emit_c_parser.add_argument("file", help="Path to Synapse file")
    emit_c_parser.add_argument("-o", "--output", help="Output .c file path")

    # Emit-PyExt (Transpile to CPython C-Extension)
    emit_pyext_parser = subparsers.add_parser("emit-pyext", help="Transpile Synapse functions into CPython C-Extension (.c)")
    emit_pyext_parser.add_argument("file", help="Path to Synapse file")
    emit_pyext_parser.add_argument("-o", "--output", help="Output .c file path")
    emit_pyext_parser.add_argument("-m", "--module-name", help="Python module name (default: file basename)")

    # Build (Compile to Native binary or WebAssembly)
    build_parser = subparsers.add_parser("build", help="Build Synapse file to Native binary or WebAssembly")
    build_parser.add_argument("file", help="Path to Synapse file")
    build_parser.add_argument("-o", "--output", help="Output executable binary or WASM path")
    build_parser.add_argument("--target", choices=["native", "wasm"], default="native", help="Target platform: 'native' (default) or 'wasm' (WebAssembly)")
    build_parser.add_argument("--c-only", action="store_true", help="Emit C code only without invoking compiler")
    build_parser.add_argument("--html", action="store_true", help="Generate companion HTML test harness when targeting wasm")
    build_parser.add_argument("--node", action="store_true", help="Generate companion Node.js test runner when targeting wasm")
    build_parser.add_argument("--profile", choices=["standalone", "hybrid"], default="standalone", help="Compilation profile: 'standalone' (pure Synapse, default) or 'hybrid' (Python interop allowed)")
    build_parser.add_argument("--mode", choices=["strict", "tolerant"], default="strict", help="Compilation mode: 'strict' (default) or 'tolerant'")
    build_parser.add_argument("--format", choices=["human", "json"], default="human", help="Output format: 'human' (default) or 'json'")
    build_parser.add_argument("--agent", action="store_true", help="Output structured agentic diagnostics JSON format for autonomous LLM agents")

    # LSP (Language Server Protocol)
    subparsers.add_parser("lsp", help="Start Synapse LSP server over stdio for IDE integration")

    # DAP (Debug Adapter Protocol)
    subparsers.add_parser("dap", help="Start Synapse DAP debugger server over stdio for VS Code integration")

    # Pkg (Package Manager)
    pkg_parser = subparsers.add_parser("pkg", help="Synapse package manager")
    pkg_sub = pkg_parser.add_subparsers(dest="pkg_subcommand")

    pkg_init = pkg_sub.add_parser("init", help="Initialize a new Synapse project with synapse.toml")
    pkg_init.add_argument("dir", nargs="?", default=".", help="Project directory (default: current directory)")
    pkg_init.add_argument("--name", help="Project name")

    pkg_add = pkg_sub.add_parser("add", help="Add / install a package dependency")
    pkg_add.add_argument("package", help="Package name, path, or URL")
    pkg_add.add_argument("--dir", default=".", help="Project directory")

    pkg_list = pkg_sub.add_parser("list", help="List installed packages")
    pkg_list.add_argument("--dir", default=".", help="Project directory")

    pkg_rm = pkg_sub.add_parser("remove", help="Remove an installed package")
    pkg_rm.add_argument("package", help="Package name to remove")
    pkg_rm.add_argument("--dir", default=".", help="Project directory")

    pkg_lock = pkg_sub.add_parser("lock", help="Generate deterministic synapse.lock lockfile")
    pkg_lock.add_argument("--dir", default=".", help="Project directory")

    pkg_search = pkg_sub.add_parser("search", help="Search the Synapse package registry")
    pkg_search.add_argument("query", nargs="?", default="", help="Query string to search for")
    pkg_search.add_argument("--dir", default=".", help="Project directory")

    pkg_install = pkg_sub.add_parser("install", help="Install package from registry with SHA-256 verification")
    pkg_install.add_argument("package", help="Package name to install")
    pkg_install.add_argument("--version", help="Package version")
    pkg_install.add_argument("--dir", default=".", help="Project directory")

    pkg_publish = pkg_sub.add_parser("publish", help="Publish package to registry with SHA-256 archive")
    pkg_publish.add_argument("package_dir", nargs="?", default=".", help="Directory of package to publish")

    # Device & Info
    subparsers.add_parser("device", help="Display CPU/CUDA hardware status and GPU model")
    subparsers.add_parser("info", help="Display hardware info and Synapse runtime status")

    # AI Status
    subparsers.add_parser("ai-status", help="Display current AI and LLM provider configuration")

    # Clean (AST Cache Cleaner)
    clean_parser = subparsers.add_parser("clean", help="Clean compiler AST disk cache (.syn_cache)")
    clean_parser.add_argument("--dir", default=".syn_cache", help="Cache directory to clean (default: .syn_cache)")

    # Verify Shapes (Symbolic Tensor Shape Guard)
    shape_parser = subparsers.add_parser("verify-shapes", help="Verify tensor shape invariants and dimension contracts statically")
    shape_parser.add_argument("file", help="Synapse source file (.syn) to verify")
    shape_parser.add_argument("--json", action="store_true", help="Output verification diagnostics as JSON")

    # Migrate (Python to Synapse Migration)
    migrate_parser = subparsers.add_parser("migrate", help="Migrate Python source (.py) to Synapse (.syn)")
    migrate_parser.add_argument("file", help="Path to Python file (.py)")
    migrate_parser.add_argument("-o", "--output", help="Output .syn file path")
    migrate_parser.add_argument("--diff", action="store_true", help="Display unified diff of changes without writing file")

    # Kernel (Jupyter Kernel Management)
    kernel_parser = subparsers.add_parser("kernel", help="Manage Synapse Jupyter Kernel")
    kernel_sub = kernel_parser.add_subparsers(dest="kernel_subcommand")
    kernel_inst = kernel_sub.add_parser("install", help="Install Synapse Jupyter kernel specification")
    kernel_inst.add_argument("--user", action="store_true", default=True, help="Install to user directory")
    kernel_inst.add_argument("--dir", help="Custom target directory for kernelspec")

    # Demo (Interactive Instant Showcase)
    demo_parser = subparsers.add_parser("demo", help="Run interactive Synapse instant showcase demos")
    demo_parser.add_argument(
        "--preset",
        choices=["nanogpt", "matmul", "tour", "dataloader"],
        default="nanogpt",
        help="Showcase preset: 'nanogpt' (default), 'matmul', 'tour', or 'dataloader'",
    )
    demo_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without prompting for keypresses (useful in automated tests and scripts)",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_file(
            args.file,
            json_diagnostics=args.json_diagnostics,
            ai_tolerant=args.ai_tolerant,
            no_cache=getattr(args, "no_cache", False),
            profile=getattr(args, "profile", "standalone"),
            mode=getattr(args, "mode", "strict"),
            explicit_profile=_is_flag_explicit("--profile"),
            compat=getattr(args, "compat", "synapse"),
            vm_mode=getattr(args, "vm", False),
            output_format=getattr(args, "format", "human"),
            is_agent=getattr(args, "agent", False),
        )
    elif args.command == "clean":
        from synapse.compiler.cache import ASTCache
        target_dir = getattr(args, "dir", ".syn_cache") or ".syn_cache"
        count = ASTCache.clean(target_dir)
        print(f"Cleared Synapse compiler cache ({count} cached AST file{'s' if count != 1 else ''} removed from '{target_dir}').")
    elif args.command == "verify-shapes":
        from synapse.analyzer.shape_guard import verify_shapes_in_file
        if not os.path.isfile(args.file):
            print(f"Error: File not found '{args.file}'", file=sys.stderr)
            sys.exit(1)
        is_valid, errors = verify_shapes_in_file(args.file)
        if getattr(args, "json", False):
            import json
            payload = {
                "valid": is_valid,
                "file": args.file,
                "errors": [
                    {
                        "message": e.message,
                        "line": e.line,
                        "column": e.column,
                        "expected": str(e.expected) if e.expected is not None else None,
                        "actual": str(e.actual) if e.actual is not None else None,
                        "suggested_fix": e.suggested_fix,
                    }
                    for e in errors
                ],
            }
            print(json.dumps(payload, indent=2))
            sys.exit(0 if is_valid else 1)
        else:
            if is_valid:
                print(f"[Shape Guard] PASS: All tensor shape invariants and dimension contracts verified in '{args.file}'")
                sys.exit(0)
            else:
                print(f"[Shape Guard] FAILED: Found {len(errors)} shape violation{'s' if len(errors) != 1 else ''} in '{args.file}':\n", file=sys.stderr)
                for err in errors:
                    print(str(err), file=sys.stderr)
                    print("", file=sys.stderr)
                sys.exit(1)
    elif args.command == "migrate":
        migrate_cli(args.file, output=getattr(args, "output", None), show_diff=getattr(args, "diff", False))
    elif args.command == "kernel":
        kernel_cli(
            getattr(args, "kernel_subcommand", "install") or "install",
            user=getattr(args, "user", True),
            custom_dir=getattr(args, "dir", None),
        )
    elif args.command == "test":
        from synapse.testing.runner import run_tests
        code = run_tests(args.path, verbose=args.verbose)
        sys.exit(code)
    elif args.command == "fix":
        fix_file(args.file, output=getattr(args, "output", None), show_diff=getattr(args, "diff", False))
    elif args.command == "fmt":
        fmt_file(args.file, output=args.output, show_diff=args.diff)
    elif args.command == "doc":
        generate_docs(
            args.file,
            output=args.output,
            as_html=getattr(args, "html", False),
            coverage=getattr(args, "coverage", False),
            as_dir=getattr(args, "dir", False),
        )
    elif args.command == "prompt-hint":
        show_prompt_hint()
    elif args.command == "mcp":
        run_mcp()
    elif args.command == "check":
        check_file(
            args.file,
            as_json=args.json,
            output_format=getattr(args, "format", "human"),
            is_agent=getattr(args, "agent", False),
        )
    elif args.command == "lint":
        run_lint(
            args.paths,
            as_json=args.json,
            strict=getattr(args, "strict", False),
            output_format=getattr(args, "format", "human"),
            is_agent=getattr(args, "agent", False),
        )
    elif args.command == "ast":
        show_ast(args.file)
    elif args.command == "dis":
        show_disassembly(args.file)
    elif args.command == "ai-status":
        from synapse.ai.providers import get_active_provider_name, OllamaProvider
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        ollama_up = OllamaProvider().is_available()
        print("=== Synapse AI Engine Configuration ===")
        print(f"  Active LLM Provider: {get_active_provider_name()}")
        print(f"  OpenAI API Key:       {'[Configured]' if has_openai else '[Not set]'}")
        print(f"  Anthropic API Key:    {'[Configured]' if has_anthropic else '[Not set]'}")
        print(f"  Local Ollama:         {'[Running on port 11434]' if ollama_up else '[Offline / Not found]'}")
    elif args.command == "emit-c":
        from synapse.codegen.native_compiler import NativeCompiler
        nc = NativeCompiler()
        out_path = args.output or os.path.splitext(args.file)[0] + ".c"
        nc.transpile_file(args.file, out_path)
        print(f"Transpiled successfully: {args.file} -> {out_path}")
    elif args.command == "emit-pyext":
        from synapse.codegen.native_compiler import NativeCompiler
        nc = NativeCompiler()
        out_path = args.output or os.path.splitext(args.file)[0] + "_ext.c"
        nc.transpile_pyext_file(args.file, out_path, module_name=getattr(args, "module_name", None))
        print(f"Python C-Extension source emitted: {args.file} -> {out_path}")
    elif args.command == "build":
        out_fmt = getattr(args, "format", "human") or "human"
        is_agent = getattr(args, "agent", False)
        use_agentic_json = is_agent or (out_fmt == "json")
        from synapse.core.diagnostics import Diagnostic, DiagnosticReport, diagnose_code

        if not os.path.isfile(args.file):
            if use_agentic_json:
                diag = Diagnostic(
                    file=args.file,
                    line=1,
                    column=1,
                    severity="error",
                    code="SYN-E001",
                    message=f"File not found '{args.file}'",
                )
                print(DiagnosticReport(status="error", diagnostics=[diag]).to_json(indent=2))
            else:
                print(f"Error: File not found '{args.file}'", file=sys.stderr)
            sys.exit(1)

        with open(args.file, "r", encoding="utf-8") as f:
            source = f.read()

        mode = getattr(args, "mode", "strict")
        if mode == "tolerant":
            from synapse.core.diagnostics import fix_ai_drift
            source, _, _ = fix_ai_drift(source, filename=args.file)

        # Profile and compliance check
        from synapse.parser import parse_source
        from synapse.compiler.options import (
            CompilationProfile,
            StandaloneViolationError,
            check_profile_compliance,
            _find_all_import_stmts,
        )

        profile_arg = getattr(args, "profile", "standalone")
        explicit_profile = _is_flag_explicit("--profile")

        try:
            ast = parse_source(source, filename=args.file)
            if explicit_profile and profile_arg:
                selected_profile = CompilationProfile.from_string(profile_arg)
            elif profile_arg == "hybrid":
                selected_profile = CompilationProfile.HYBRID
            elif profile_arg == "standalone" and explicit_profile:
                selected_profile = CompilationProfile.STANDALONE
            else:
                imports = _find_all_import_stmts(ast)
                has_py = any(
                    getattr(imp, "is_python", False) or (getattr(imp, "module_path", None) and imp.module_path[0] == "py")
                    for imp in imports
                )
                selected_profile = CompilationProfile.HYBRID if has_py else CompilationProfile.STANDALONE

            check_profile_compliance(ast, selected_profile)
        except StandaloneViolationError as e:
            if use_agentic_json:
                diag = Diagnostic(
                    file=args.file,
                    line=1,
                    column=1,
                    severity="error",
                    code="SYN-E401",
                    message=str(e),
                )
                print(DiagnosticReport(status="error", diagnostics=[diag], file=args.file, error_type="StandaloneViolationError", message=str(e)).to_json(indent=2))
            else:
                print(f"\n[Profile Error] {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            if use_agentic_json:
                report = diagnose_code(source, filepath=args.file, error=e)
                if report.status == "ok":
                    diag = Diagnostic(
                        file=args.file,
                        line=1,
                        column=1,
                        severity="error",
                        code="SYN-E101",
                        message=str(e),
                    )
                    report = DiagnosticReport(status="error", diagnostics=[diag], file=args.file, error_type=type(e).__name__, message=str(e))
                print(report.to_json(indent=2))
            else:
                if isinstance(e, (LexerError, ParseError)):
                    print(f"\n[Syntax Error] {e}", file=sys.stderr)
                else:
                    print(f"\n[Build Error] {e}", file=sys.stderr)
            sys.exit(1)

        target = getattr(args, "target", "native") or "native"
        if target == "wasm":
            from synapse.codegen.wasm_compiler import WasmCompiler
            wc = WasmCompiler()
            c_path = os.path.splitext(args.file)[0] + ".c"
            out_wasm = args.output or (os.path.splitext(args.file)[0] + ".wasm")
            c_code = wc.transpile(source, filename=args.file)
            out_dir = os.path.dirname(os.path.abspath(c_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_code)

            if getattr(args, "c_only", False):
                if use_agentic_json:
                    print(DiagnosticReport(status="ok", diagnostics=[]).to_json(indent=2))
                else:
                    print(f"C source emitted: {c_path}")
            else:
                emit_html = getattr(args, "html", False)
                emit_node = getattr(args, "node", False)
                result = wc.compile_synapse_to_wasm(
                    args.file,
                    out_wasm,
                    emit_html=emit_html,
                    emit_node=emit_node,
                )
                if result.success:
                    if use_agentic_json:
                        print(DiagnosticReport(status="ok", diagnostics=[]).to_json(indent=2))
                    else:
                        print(f"WASM build succeeded: {result.wasm_path} and {result.js_path}")
                        if result.html_path:
                            print(f"HTML harness generated: {result.html_path}")
                        if result.node_runner_path:
                            print(f"Node runner generated: {result.node_runner_path}")
                else:
                    if use_agentic_json:
                        diag = Diagnostic(
                            file=args.file,
                            line=1,
                            column=1,
                            severity="error",
                            code="SYN-E301",
                            message=f"WASM build failed: {result.error_message}",
                        )
                        print(DiagnosticReport(status="error", diagnostics=[diag]).to_json(indent=2))
                        sys.exit(1)
                    else:
                        print(f"Build notice: {result.error_message}")
        else:
            from synapse.codegen.native_compiler import NativeCompiler
            nc = NativeCompiler()
            c_path = os.path.splitext(args.file)[0] + ".c"
            out_exe = args.output or (os.path.splitext(args.file)[0] + (".exe" if sys.platform == "win32" else ""))
            c_code = nc.transpile(source, filename=args.file)
            out_dir = os.path.dirname(os.path.abspath(c_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(c_code)

            if getattr(args, "c_only", False):
                if use_agentic_json:
                    print(DiagnosticReport(status="ok", diagnostics=[]).to_json(indent=2))
                else:
                    print(f"C source emitted: {c_path}")
            else:
                success, msg = nc.build_executable(c_path, out_exe)
                if success:
                    if use_agentic_json:
                        print(DiagnosticReport(status="ok", diagnostics=[]).to_json(indent=2))
                    else:
                        print(f"Build succeeded: {out_exe}")
                else:
                    if use_agentic_json:
                        diag = Diagnostic(
                            file=args.file,
                            line=1,
                            column=1,
                            severity="error",
                            code="SYN-E301",
                            message=f"Build failed: {msg}",
                        )
                        print(DiagnosticReport(status="error", diagnostics=[diag]).to_json(indent=2))
                        sys.exit(1)
                    else:
                        print(f"Build notice: {msg}")
    elif args.command == "lsp":
        from synapse.lsp.server import LSPServer
        LSPServer().start()
    elif args.command == "dap":
        from synapse.tools.dap_server import run_dap_server
        run_dap_server()
    elif args.command == "pkg":
        from synapse.pkg.manager import PackageManager
        pm = PackageManager()
        subcmd = getattr(args, "pkg_subcommand", None)
        target_dir = getattr(args, "dir", ".") or "."
        if subcmd == "init":
            name = getattr(args, "name", None)
            manifest_path = pm.init_project(target_dir, name=name)
            print(f"Initialized Synapse project: {manifest_path}")
        elif subcmd == "add":
            pm.install_package(target_dir, args.package)
            print(f"Added package '{args.package}' to '{target_dir}'")
        elif subcmd == "list":
            packages = pm.list_packages(target_dir)
            if not packages:
                print(f"No packages installed in '{target_dir}'.")
            else:
                print(f"Packages in '{target_dir}':")
                for pkg in packages:
                    status = "installed" if pkg.get("installed") else "missing"
                    print(f"  * {pkg['name']} v{pkg['version']} ({status})")
        elif subcmd == "remove":
            pm.uninstall_package(target_dir, args.package)
            print(f"Removed package '{args.package}' from '{target_dir}'")
        elif subcmd == "lock":
            lock_path = pm.generate_lockfile(target_dir)
            print(f"Generated deterministic lockfile: {lock_path}")
        elif subcmd == "search":
            from synapse.pkg.registry import handle_pkg_search
            code = handle_pkg_search(getattr(args, "query", ""), project_dir=target_dir)
            sys.exit(code)
        elif subcmd == "install":
            from synapse.pkg.registry import handle_pkg_install
            code = handle_pkg_install(args.package, version=getattr(args, "version", None), project_dir=target_dir)
            sys.exit(code)
        elif subcmd == "publish":
            from synapse.pkg.registry import handle_pkg_publish
            code = handle_pkg_publish(getattr(args, "package_dir", "."))
            sys.exit(code)
        else:
            pkg_parser.print_help()
    elif args.command in ("device", "info"):
        from synapse.core.cuda_backend import is_cuda_available, get_device_name, device_count
        cuda_ok = is_cuda_available()
        count = device_count()
        print("=== Synapse Hardware & Acceleration Info ===")
        print(f"  Platform:        {sys.platform}")
        print(f"  CUDA Available:  {'[YES]' if cuda_ok else '[NO] (CPU Fallback)'}")
        print(f"  Device Count:    {count}")
        if cuda_ok and count > 0:
            for idx in range(count):
                print(f"  Device #{idx}:     {get_device_name(idx)}")
        else:
            print(f"  Primary Device:  CPU (AVX2 / Vectorized)")
    elif args.command == "demo":
        from synapse.tools.demo import run_demo
        interactive = not getattr(args, "non_interactive", False)
        code = run_demo(preset=getattr(args, "preset", "nanogpt"), interactive=interactive)
        sys.exit(code)
    elif args.command == "repl" or args.command is None:
        run_repl()


if __name__ == "__main__":
    main()
