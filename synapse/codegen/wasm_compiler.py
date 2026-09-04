"""
Synapse WebAssembly (WASM) Compilation Pipeline.

This module provides WebAssembly compilation using Emscripten (emcc),
generating high-performance browser modules, companion HTML harnesses,
and Node.js test runners.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class WasmBuildResult:
    """Result of WebAssembly compilation via Emscripten."""
    success: bool
    wasm_path: Optional[str] = None
    js_path: Optional[str] = None
    html_path: Optional[str] = None
    node_runner_path: Optional[str] = None
    command: Optional[List[str]] = None
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""

    @property
    def error(self) -> Optional[str]:
        """Convenience alias for error_message."""
        return self.error_message


class WasmCompiler:
    """
    Synapse WebAssembly (WASM) compiler using Emscripten (emcc).
    Compiles Synapse C99 code into standalone .wasm and .js modules
    with browser test harnesses and Node.js runners.
    """

    def __init__(self, runtime_dir: Optional[str] = None):
        self.runtime_dir = runtime_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "runtime")
        )
        self.runtime_c = os.path.join(self.runtime_dir, "synapse_runtime.c")
        self.runtime_h = os.path.join(self.runtime_dir, "synapse_runtime.h")

    @staticmethod
    def is_emcc_available() -> bool:
        """Checks if Emscripten (emcc) is available in system PATH."""
        return shutil.which("emcc") is not None or shutil.which("emcc.bat") is not None

    def transpile(self, source_code: str, filename: str = "<source>", include_line_directives: bool = True) -> str:
        """Transpiles Synapse source code to C99 using CEmitter."""
        from synapse.lexer.lexer import Lexer
        from synapse.parser.parser import Parser
        from synapse.codegen.c_emitter import CEmitter
        tokens = Lexer(source_code).tokenize()
        ast = Parser(tokens).parse()
        emitter = CEmitter(include_line_directives=include_line_directives)
        return emitter.emit(ast, filename=filename)

    def transpile_file(self, syn_path: str, c_output_path: str, include_line_directives: bool = True) -> str:
        """Transpiles a .syn file to a .c file."""
        with open(syn_path, "r", encoding="utf-8") as f:
            source = f.read()
        c_code = self.transpile(source, filename=syn_path, include_line_directives=include_line_directives)
        out_dir = os.path.dirname(os.path.abspath(c_output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(c_output_path, "w", encoding="utf-8") as f:
            f.write(c_code)
        return c_code

    @staticmethod
    def extract_functions(source_code: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Extracts top-level function names and signatures from Synapse source code.
        Returns (exported_functions, function_signatures).
        """
        from synapse.lexer.lexer import Lexer
        from synapse.parser.parser import Parser
        from synapse.parser.ast import FunctionDef

        try:
            tokens = Lexer(source_code).tokenize()
            ast = Parser(tokens).parse()
        except Exception:
            return ([], {})

        fn_names: List[str] = []
        signatures: Dict[str, Dict[str, Any]] = {}

        for stmt in getattr(ast, "statements", []):
            if isinstance(stmt, FunctionDef):
                fn_name = stmt.name
                fn_names.append(fn_name)
                ret_annot = getattr(stmt, "return_type", "f64") or "f64"
                param_annots = [getattr(p, "type_annot", "f64") or "f64" for p in getattr(stmt, "params", [])]
                signatures[fn_name] = {
                    "return_type": ret_annot,
                    "arg_types": param_annots,
                }

        return (fn_names, signatures)

    def build_emcc_command(
        self,
        c_source_path: str,
        output_js_path: str,
        exported_functions: Optional[List[str]] = None,
        optimize_level: str = "-O3",
        export_name: str = "SynapseModule",
        extra_flags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Builds the emcc compilation command line with production WASM flags:
        - -O3 (or custom optimize_level)
        - -s WASM=1
        - -s ALLOW_MEMORY_GROWTH=1
        - -s EXPORTED_FUNCTIONS=['_main', ...]
        - -s EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap', 'getValue', 'setValue']
        - -s MODULARIZE=1
        - -s EXPORT_NAME="SynapseModule"
        """
        opt = optimize_level if optimize_level.startswith("-") else f"-{optimize_level}"

        # Ensure _main is always exported
        exports: List[str] = ["_main"]
        if exported_functions:
            for fn in exported_functions:
                cleaned = fn.strip()
                if not cleaned:
                    continue
                norm_fn = cleaned if cleaned.startswith("_") else f"_{cleaned}"
                if norm_fn not in exports:
                    exports.append(norm_fn)

        exports_repr = str(exports)
        runtime_methods_repr = "['ccall', 'cwrap', 'getValue', 'setValue']"

        cmd: List[str] = [
            "emcc",
            opt,
            "-s", "WASM=1",
            "-s", "ALLOW_MEMORY_GROWTH=1",
            "-s", f"EXPORTED_FUNCTIONS={exports_repr}",
            "-s", f"EXPORTED_RUNTIME_METHODS={runtime_methods_repr}",
            "-s", "MODULARIZE=1",
            "-s", f'EXPORT_NAME="{export_name}"',
            c_source_path,
            self.runtime_c,
            f"-I{self.runtime_dir}",
            "-lm",
            "-o",
            output_js_path,
        ]

        if extra_flags:
            cmd.extend(extra_flags)

        return cmd

    @staticmethod
    def _relpath_for_target(target_file: str, referenced_js: str) -> str:
        """Calculates a portable forward-slash relative path from target_file directory to referenced_js."""
        target_dir = os.path.dirname(os.path.abspath(target_file))
        if os.path.dirname(referenced_js) == "":
            return referenced_js.replace("\\", "/")

        ref_abs = os.path.abspath(referenced_js)
        try:
            rel = os.path.relpath(ref_abs, target_dir)
        except ValueError:
            rel = referenced_js
        return rel.replace("\\", "/")

    def compile_c_to_wasm(
        self,
        c_source_path_or_code: str,
        output_wasm_path: str,
        exported_functions: Optional[List[str]] = None,
        optimize_level: str = "-O3",
        export_name: str = "SynapseModule",
        extra_flags: Optional[List[str]] = None,
        emit_html: bool = False,
        emit_node: bool = False,
        function_signatures: Optional[Dict[str, Any]] = None,
    ) -> WasmBuildResult:
        """
        Compiles C99 source file or code string to WebAssembly (.wasm and .js).
        """
        out_abs = os.path.abspath(output_wasm_path)
        out_dir = os.path.dirname(out_abs)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        base, ext = os.path.splitext(out_abs)
        if ext.lower() == ".wasm":
            wasm_path = out_abs
            js_path = f"{base}.js"
        elif ext.lower() == ".js":
            js_path = out_abs
            wasm_path = f"{base}.wasm"
        else:
            wasm_path = f"{out_abs}.wasm"
            js_path = f"{out_abs}.js"

        temp_c_path: Optional[str] = None
        # Accurately identify if c_source_path_or_code is inline code or a file path
        is_inline_code = "\n" in c_source_path_or_code or (
            (";" in c_source_path_or_code or "{" in c_source_path_or_code)
            and any(k in c_source_path_or_code for k in ("main", "syn_", "#include", "return", "int ", "void ", "double "))
        )

        if not is_inline_code:
            if not os.path.isfile(c_source_path_or_code):
                return WasmBuildResult(
                    success=False,
                    wasm_path=wasm_path,
                    js_path=js_path,
                    error_message=f"C source file does not exist: {c_source_path_or_code}",
                )
            c_source_file = os.path.abspath(c_source_path_or_code)
        else:
            fd, temp_c_path = tempfile.mkstemp(suffix=".c", prefix="synapse_wasm_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(c_source_path_or_code)
            c_source_file = temp_c_path

        if not os.path.isfile(self.runtime_c):
            if temp_c_path and os.path.isfile(temp_c_path):
                try:
                    os.remove(temp_c_path)
                except OSError:
                    pass
            return WasmBuildResult(
                success=False,
                wasm_path=wasm_path,
                js_path=js_path,
                error_message=f"Synapse C runtime file does not exist: {self.runtime_c}",
            )

        cmd = self.build_emcc_command(
            c_source_path=c_source_file,
            output_js_path=js_path,
            exported_functions=exported_functions,
            optimize_level=optimize_level,
            export_name=export_name,
            extra_flags=extra_flags,
        )

        if not self.is_emcc_available():
            if temp_c_path and os.path.isfile(temp_c_path):
                try:
                    os.remove(temp_c_path)
                except OSError:
                    pass
            return WasmBuildResult(
                success=False,
                wasm_path=wasm_path,
                js_path=js_path,
                command=cmd,
                error_message=(
                    "Emscripten compiler 'emcc' not found in PATH.\n"
                    "Please install Emscripten SDK (emsdk) and run 'emsdk activate latest'.\n"
                    "The C source code is ready to compile to WebAssembly."
                ),
            )

        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            html_path: Optional[str] = None
            node_runner_path: Optional[str] = None
            if res.returncode == 0:
                if emit_html:
                    html_path = f"{base}.html"
                    self.write_html_harness(
                        html_path=html_path,
                        wasm_js_filename=js_path,
                        function_signatures=function_signatures,
                        module_name=export_name,
                    )

                if emit_node:
                    node_runner_path = f"{base}_runner.js"
                    self.write_node_runner(
                        node_script_path=node_runner_path,
                        wasm_js_filename=js_path,
                        module_name=export_name,
                    )

                return WasmBuildResult(
                    success=True,
                    wasm_path=wasm_path,
                    js_path=js_path,
                    html_path=html_path,
                    node_runner_path=node_runner_path,
                    command=cmd,
                    stdout=res.stdout,
                    stderr=res.stderr,
                )
            else:
                err = res.stderr or res.stdout
                return WasmBuildResult(
                    success=False,
                    wasm_path=wasm_path,
                    js_path=js_path,
                    command=cmd,
                    error_message=f"Emscripten compilation failed:\n{err}",
                    stdout=res.stdout,
                    stderr=res.stderr,
                )
        except Exception as e:
            return WasmBuildResult(
                success=False,
                wasm_path=wasm_path,
                js_path=js_path,
                command=cmd,
                error_message=f"Emscripten execution error: {e}",
            )
        finally:
            if temp_c_path and os.path.isfile(temp_c_path):
                try:
                    os.remove(temp_c_path)
                except OSError:
                    pass

    def compile_synapse_to_wasm(
        self,
        syn_path: str,
        output_wasm_path: str,
        exported_functions: Optional[List[str]] = None,
        optimize_level: str = "-O3",
        export_name: str = "SynapseModule",
        emit_html: bool = False,
        emit_node: bool = False,
        function_signatures: Optional[Dict[str, Any]] = None,
    ) -> WasmBuildResult:
        """Transpiles a .syn file to C and compiles it directly to WASM."""
        base = os.path.splitext(output_wasm_path)[0]
        c_path = f"{base}.c"

        with open(syn_path, "r", encoding="utf-8") as f:
            syn_source = f.read()

        # Auto-discover exported functions and signatures if not explicitly given
        auto_fns, auto_sigs = self.extract_functions(syn_source)
        if exported_functions is None and auto_fns:
            exported_functions = auto_fns
        if function_signatures is None and auto_sigs:
            function_signatures = auto_sigs

        self.transpile_file(syn_path, c_path)
        return self.compile_c_to_wasm(
            c_path,
            output_wasm_path,
            exported_functions=exported_functions,
            optimize_level=optimize_level,
            export_name=export_name,
            emit_html=emit_html,
            emit_node=emit_node,
            function_signatures=function_signatures,
        )

    @staticmethod
    def _to_emscripten_type(t: str) -> str:
        """Converts user or language type strings into valid Emscripten ccall type identifiers."""
        s = str(t).strip().lower()
        if s in ("void", "none", "null"):
            return "void"
        if s in ("string", "str", "char*", "const char*"):
            return "string"
        if s in ("bool", "boolean"):
            return "boolean"
        if s in ("array", "buffer", "bytes", "uint8array"):
            return "array"
        return "number"

    @staticmethod
    def _normalize_signature(sig: Any) -> Tuple[str, List[str]]:
        """Normalizes various function signature specifications into (return_type, arg_types)."""
        if not sig:
            return ("number", [])
        if isinstance(sig, dict):
            ret = sig.get("return_type") or sig.get("returns") or sig.get("return") or sig.get("ret") or "number"
            args = sig.get("arg_types") or sig.get("args") or sig.get("params") or sig.get("parameters") or []
            return (str(ret), [str(a) for a in args])
        if isinstance(sig, (tuple, list)):
            if len(sig) == 2 and isinstance(sig[1], (list, tuple)):
                return (str(sig[0]), [str(a) for a in sig[1]])
            return ("number", [str(a) for a in sig])
        if isinstance(sig, str):
            return (sig, [])
        return ("number", [])

    @staticmethod
    def generate_html_harness(
        wasm_js_filename: str,
        function_signatures: Optional[Dict[str, Any]] = None,
        module_name: str = "SynapseModule",
        title: str = "Synapse WebAssembly Test Runner",
    ) -> str:
        """
        Generates a sleek, modern HTML/JS test runner adhering to Linear / Vercel minimalist aesthetics
        for executing and testing compiled Synapse WASM modules in browsers.
        """
        signatures = function_signatures or {}
        fn_cards_html = []
        clean_js_filename = wasm_js_filename.replace("\\", "/")

        for fn_name, sig in signatures.items():
            clean_fn = fn_name.lstrip("_") or fn_name
            dom_id = re.sub(r"[^a-zA-Z0-9_]", "_", clean_fn)
            raw_ret_type, raw_arg_types = WasmCompiler._normalize_signature(sig)
            ccall_ret = WasmCompiler._to_emscripten_type(raw_ret_type)
            ccall_args = [WasmCompiler._to_emscripten_type(a) for a in raw_arg_types]

            inputs_html = []
            for idx, arg_t in enumerate(raw_arg_types):
                inputs_html.append(f"""
                <div class="input-group">
                    <label>arg{idx} ({arg_t}):</label>
                    <input type="text" id="{dom_id}_arg_{idx}" value="1.0" class="input-field" />
                </div>
                """)

            inputs_str = "\n".join(inputs_html) if inputs_html else '<p class="text-zinc-500 text-xs">No parameters</p>'
            ccall_args_json = json.dumps(ccall_args)

            fn_cards_html.append(f"""
            <div class="card" id="card_{dom_id}">
                <div class="card-header">
                    <div>
                        <span class="fn-name">{clean_fn}</span>
                        <span class="badge badge-return">-&gt; {raw_ret_type}</span>
                    </div>
                    <button class="btn btn-primary" onclick="invokeFunction('{clean_fn}', '{ccall_ret}', {ccall_args_json}, '{dom_id}')">
                        Execute
                    </button>
                </div>
                <div class="card-body">
                    {inputs_str}
                    <div class="result-row">
                        <span class="result-label">Result:</span>
                        <span id="result_{dom_id}" class="result-value">-</span>
                        <span id="timing_{dom_id}" class="timing-badge"></span>
                    </div>
                </div>
            </div>
            """)

        all_cards_str = "\n".join(fn_cards_html) if fn_cards_html else """
        <div class="empty-state">
            <p>No custom exported functions specified. Click below to execute <code>_main()</code>.</p>
        </div>
        """

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #09090b;
            --card-bg: #121215;
            --border: #27272a;
            --border-subtle: #1f1f23;
            --text: #f4f4f5;
            --text-muted: #a1a1aa;
            --text-dim: #71717a;
            --primary: #ffffff;
            --primary-text: #09090b;
            --accent: #3b82f6;
            --success: #22c55e;
            --error: #ef4444;
            --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
            --mono: "JetBrains Mono", "Cascadia Code", "Fira Code", monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg);
            color: var(--text);
            font-family: var(--font);
            font-size: 14px;
            line-height: 1.5;
            min-height: 100vh;
            padding: 32px 20px;
        }}

        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}

        .header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}

        .title-group h1 {{
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .title-group p {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 500;
            background: rgba(39, 39, 42, 0.5);
            border: 1px solid var(--border);
            color: var(--text-muted);
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #eab308;
        }}

        .status-dot.ready {{
            background: var(--success);
            box-shadow: 0 0 8px rgba(34, 197, 94, 0.3);
        }}

        .status-dot.error {{
            background: var(--error);
        }}

        .control-panel {{
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }}

        .btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            border: 1px solid transparent;
        }}

        .btn-primary {{
            background: var(--primary);
            color: var(--primary-text);
        }}

        .btn-primary:hover {{
            background: #e4e4e7;
        }}

        .btn-secondary {{
            background: #18181b;
            color: var(--text);
            border-color: var(--border);
        }}

        .btn-secondary:hover {{
            background: #27272a;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}

        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 10px;
        }}

        .fn-name {{
            font-family: var(--mono);
            font-weight: 600;
            font-size: 14px;
            color: #fafafa;
        }}

        .badge {{
            display: inline-block;
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 6px;
        }}

        .badge-return {{
            background: #1e1e24;
            color: #93c5fd;
            border: 1px solid #1e293b;
            font-family: var(--mono);
        }}

        .card-body {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .input-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .input-group label {{
            font-size: 12px;
            color: var(--text-muted);
            min-width: 100px;
            font-family: var(--mono);
        }}

        .input-field {{
            flex: 1;
            background: #09090b;
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text);
            padding: 6px 10px;
            font-size: 12px;
            font-family: var(--mono);
        }}

        .input-field:focus {{
            outline: none;
            border-color: #52525b;
        }}

        .result-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 6px;
            padding-top: 8px;
            border-top: 1px solid var(--border-subtle);
        }}

        .result-label {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        .result-value {{
            font-family: var(--mono);
            font-weight: 600;
            color: var(--success);
            font-size: 13px;
        }}

        .timing-badge {{
            font-size: 11px;
            color: var(--text-dim);
            margin-left: auto;
            font-family: var(--mono);
        }}

        .console-box {{
            background: #09090b;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}

        .console-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }}

        .console-header span {{
            font-size: 12px;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .terminal {{
            background: #050507;
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 12px;
            font-family: var(--mono);
            font-size: 12px;
            color: #d4d4d8;
            max-height: 260px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }}

        .empty-state {{
            padding: 24px;
            text-align: center;
            color: var(--text-dim);
            border: 1px dashed var(--border);
            border-radius: 8px;
            margin-bottom: 24px;
        }}

        .log-stdout {{ color: #a1a1aa; }}
        .log-stderr {{ color: #ef4444; }}
        .log-info {{ color: #60a5fa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-group">
                <h1>
                    <span>Synapse</span>
                    <span style="font-size: 14px; font-weight: 400; color: var(--text-dim);">/</span>
                    <span>WASM Harness</span>
                </h1>
                <p>Emscripten Ahead-Of-Time (AOT) WebAssembly Browser Runtime</p>
            </div>
            <div id="statusBadge" class="status-badge">
                <span id="statusDot" class="status-dot"></span>
                <span id="statusText">Initializing WASM...</span>
            </div>
        </div>

        <div class="control-panel">
            <button class="btn btn-primary" onclick="runMain()">Execute _main()</button>
            <button class="btn btn-secondary" onclick="clearConsole()">Clear Terminal</button>
        </div>

        <div class="grid">
            {all_cards_str}
        </div>

        <div class="console-box">
            <div class="console-header">
                <span>Execution Logs &amp; Terminal Output</span>
                <span id="memoryInfo" style="font-family: var(--mono); font-size: 11px;">Heap: -</span>
            </div>
            <div id="terminal" class="terminal">[Synapse WASM] Initializing environment...</div>
        </div>
    </div>

    <script src="{clean_js_filename}"></script>
    <script>
        let synapseInstance = null;

        function appendLog(msg, type = 'stdout') {{
            const terminal = document.getElementById('terminal');
            const line = document.createElement('div');
            line.className = 'log-' + type;
            const time = new Date().toISOString().substring(11, 19);
            line.textContent = `[${{time}}] ${{msg}}`;
            terminal.appendChild(line);
            terminal.scrollTop = terminal.scrollHeight;
        }}

        function clearConsole() {{
            const terminal = document.getElementById('terminal');
            terminal.innerHTML = '';
        }}

        function updateStatus(text, isReady, isError = false) {{
            const badge = document.getElementById('statusBadge');
            const dot = document.getElementById('statusDot');
            const label = document.getElementById('statusText');
            label.textContent = text;
            dot.className = 'status-dot' + (isReady ? ' ready' : (isError ? ' error' : ''));
        }}

        function updateMemoryInfo() {{
            if (synapseInstance && synapseInstance.HEAPU8) {{
                const mb = (synapseInstance.HEAPU8.buffer.byteLength / (1024 * 1024)).toFixed(2);
                document.getElementById('memoryInfo').textContent = `Heap: ${{mb}} MB`;
            }}
        }}

        window.addEventListener('DOMContentLoaded', async () => {{
            try {{
                appendLog('Loading WebAssembly module: {clean_js_filename}', 'info');
                if (typeof window['{module_name}'] === 'function') {{
                    synapseInstance = await window['{module_name}']({{
                        print: (text) => appendLog(text, 'stdout'),
                        printErr: (text) => appendLog(text, 'stderr')
                    }});
                    updateStatus('Ready (WebAssembly Loaded)', true);
                    updateMemoryInfo();
                    appendLog('WebAssembly module ready. Exported runtime methods available.', 'info');
                }} else {{
                    updateStatus('Factory {module_name} not found', false, true);
                    appendLog('Error: window.{module_name} factory was not defined by {clean_js_filename}', 'stderr');
                }}
            }} catch (err) {{
                updateStatus('Module Load Failed', false, true);
                appendLog('Module load error: ' + (err.message || err), 'stderr');
            }}
        }});

        async function runMain() {{
            if (!synapseInstance) {{
                appendLog('Error: WASM module is not yet initialized', 'stderr');
                return;
            }}
            try {{
                appendLog('Executing _main()...', 'info');
                const t0 = performance.now();
                let code = 0;
                if (typeof synapseInstance._main === 'function') {{
                    code = synapseInstance._main();
                }} else if (typeof synapseInstance.ccall === 'function') {{
                    code = synapseInstance.ccall('main', 'number', [], []);
                }}
                const t1 = performance.now();
                appendLog(`_main() finished with return code: ${{code}} in ${{(t1 - t0).toFixed(2)}}ms`, 'stdout');
                updateMemoryInfo();
            }} catch (err) {{
                appendLog('_main() runtime error: ' + (err.message || err), 'stderr');
            }}
        }}

        function invokeFunction(fnName, retType, argTypes, domId) {{
            if (!synapseInstance) {{
                appendLog('Error: WASM module is not yet initialized', 'stderr');
                return;
            }}

            const targetId = domId || fnName;
            const resultEl = document.getElementById('result_' + targetId);
            const timingEl = document.getElementById('timing_' + targetId);

            try {{
                const args = [];
                for (let i = 0; i < argTypes.length; i++) {{
                    const el = document.getElementById(targetId + '_arg_' + i);
                    const raw = el ? el.value.trim() : '0';
                    if (argTypes[i] === 'number') {{
                        const num = Number(raw);
                        args.push(isNaN(num) ? 0 : num);
                    }} else if (argTypes[i] === 'boolean') {{
                        args.push(raw === 'true' || raw === '1');
                    }} else {{
                        args.push(raw);
                    }}
                }}

                appendLog(`Calling ${{fnName}}(${{args.join(', ')}})...`, 'info');
                const t0 = performance.now();

                const ccallRetType = (retType === 'void' || retType === 'null' || !retType) ? null : retType;
                let res;
                if (typeof synapseInstance.ccall === 'function') {{
                    res = synapseInstance.ccall(fnName, ccallRetType, argTypes, args);
                }} else if (typeof synapseInstance['_' + fnName] === 'function') {{
                    res = synapseInstance['_' + fnName](...args);
                }} else {{
                    throw new Error(`Exported function ${{fnName}} not found on module`);
                }}

                if (ccallRetType === null && (res === undefined || res === null)) {{
                    res = 'void (success)';
                }}

                const t1 = performance.now();
                const elapsed = (t1 - t0).toFixed(2);

                if (resultEl) resultEl.textContent = res;
                if (timingEl) timingEl.textContent = `${{elapsed}}ms`;

                appendLog(`${{fnName}} returned: ${{res}} (${{elapsed}}ms)`, 'stdout');
                updateMemoryInfo();
            }} catch (err) {{
                if (resultEl) resultEl.textContent = 'Error';
                appendLog(`Error invoking ${{fnName}}: ` + (err.message || err), 'stderr');
            }}
        }}
    </script>
</body>
</html>
"""
        return html_template

    @staticmethod
    def generate_node_runner(wasm_js_filename: str, module_name: str = "SynapseModule") -> str:
        """
        Generates a Node.js runner script for the modularized Emscripten WASM build.
        Path separators are normalized to forward slashes to prevent escape corruption on Windows.
        """
        clean_filename = wasm_js_filename.replace("\\", "/")
        code = f"""/**
 * Synapse WebAssembly Node.js Runner
 * Auto-generated by Synapse WasmCompiler
 */
const path = require('path');
const createModule = require(path.resolve(__dirname, '{clean_filename}'));

async function run(options = {{}}) {{
    console.log('[Synapse WASM] Initializing {module_name} in Node.js...');
    const startTime = process.hrtime.bigint();

    const moduleInstance = await createModule({{
        print: (text) => console.log('[stdout]', text),
        printErr: (text) => console.error('[stderr]', text),
        ...options
    }});

    const initDurationMs = Number(process.hrtime.bigint() - startTime) / 1e6;
    console.log(`[Synapse WASM] {module_name} loaded successfully in ${{initDurationMs.toFixed(2)}}ms`);

    let exitCode = null;
    if (typeof moduleInstance._main === 'function') {{
        console.log('[Synapse WASM] Invoking _main()...');
        const execStart = process.hrtime.bigint();
        exitCode = moduleInstance._main();
        const execDurationMs = Number(process.hrtime.bigint() - execStart) / 1e6;
        console.log(`[Synapse WASM] _main() returned: ${{exitCode}} (${{execDurationMs.toFixed(2)}}ms)`);
    }}

    return {{
        module: moduleInstance,
        exitCode: exitCode
    }};
}}

if (require.main === module) {{
    run().then(({{ exitCode }}) => {{
        if (exitCode !== null && exitCode !== 0) {{
            process.exit(exitCode);
        }}
    }}).catch(err => {{
        console.error('[Synapse WASM] Execution error:', err);
        process.exit(1);
    }});
}}

module.exports = {{ run, createModule }};
"""
        return code

    @classmethod
    def write_html_harness(
        cls,
        html_path: str,
        wasm_js_filename: str,
        function_signatures: Optional[Dict[str, Any]] = None,
        module_name: str = "SynapseModule",
    ) -> str:
        """Generates and writes companion HTML test harness to disk with correct relative link."""
        rel_js = cls._relpath_for_target(html_path, wasm_js_filename)
        html_content = cls.generate_html_harness(
            rel_js,
            function_signatures=function_signatures,
            module_name=module_name,
        )
        out_dir = os.path.dirname(os.path.abspath(html_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return html_content

    @classmethod
    def write_node_runner(
        cls,
        node_script_path: str,
        wasm_js_filename: str,
        module_name: str = "SynapseModule",
    ) -> str:
        """Generates and writes companion Node.js runner to disk with correct relative link."""
        rel_js = cls._relpath_for_target(node_script_path, wasm_js_filename)
        runner_content = cls.generate_node_runner(
            rel_js,
            module_name=module_name,
        )
        out_dir = os.path.dirname(os.path.abspath(node_script_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(node_script_path, "w", encoding="utf-8") as f:
            f.write(runner_content)
        return runner_content
