"""
Synapse Native Jupyter Kernel & Interactive Notebook Engine (Phase 2)
======================================================================
Provides interactive cell-by-cell execution, persistent stateful VM sessions,
rich MIME displays (DataFrames, Tensors, Plots), and kernelspec installer
for JupyterLab, VS Code, and nteract.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from synapse.parser.parser import Parser
from synapse.lexer.lexer import Lexer
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine, VMRuntimeError
from synapse.tools.display import render_mime_bundle


class SynapseKernelEngine:
    """
    Core stateful execution engine for Synapse interactive sessions.
    Maintains persistent memory, globals, functions, and models across cells.
    """

    def __init__(self):
        self.vm = VirtualMachine()
        self.execution_count = 0
        self._history: List[str] = []

    def execute_cell(self, code_text: str) -> Dict[str, Any]:
        """
        Executes a block of Synapse code within the persistent VM session.
        Captures stdout, stderr, and produces rich MIME bundles for outputs.
        """
        self.execution_count += 1
        self._history.append(code_text)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result_val = None
        status = "ok"
        error_info: Optional[Dict[str, str]] = None

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                tokens = Lexer(code_text).tokenize()
                ast = Parser(tokens).parse()
                compiled_code = Compiler(name=f"<cell_{self.execution_count}>").compile(ast)
                result_val = self.vm.execute(compiled_code)
            except VMRuntimeError as e:
                status = "error"
                error_info = {
                    "ename": "VMRuntimeError",
                    "evalue": str(e),
                    "traceback": [f"RuntimeError: {e}"],
                }
                print(f"[Synapse Runtime Error] {e}", file=sys.stderr)
            except Exception as e:
                status = "error"
                error_info = {
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [f"{type(e).__name__}: {e}"],
                }
                print(f"[Synapse Error] {e}", file=sys.stderr)

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()

        # If result is None, check if last global variable was modified or stack output
        mime_bundle: Optional[Dict[str, Any]] = None
        if result_val is not None:
            mime_bundle = render_mime_bundle(result_val)

        return {
            "status": status,
            "execution_count": self.execution_count,
            "stdout": stdout_val,
            "stderr": stderr_val,
            "result": result_val,
            "mime_bundle": mime_bundle,
            "error": error_info,
        }

    def complete(self, code_text: str, cursor_pos: int) -> Dict[str, Any]:
        """Calculates code autocompletions at cursor position."""
        prefix = code_text[:cursor_pos]
        # Find word before cursor
        word = ""
        for ch in reversed(prefix):
            if ch.isalnum() or ch == "_":
                word = ch + word
            else:
                break

        matches = []
        # Check builtins & keywords
        keywords = [
            "fn", "let", "const", "tensor", "zeros", "ones", "randn",
            "if", "else", "while", "for", "in", "return", "import",
            "dataframe", "read_csv", "spawn", "channel", "Device"
        ]
        for kw in keywords:
            if kw.startswith(word):
                matches.append(kw)

        # Check globals in VM
        for var_name in self.vm.globals.keys():
            if var_name.startswith(word) and not var_name.startswith("__"):
                if var_name not in matches:
                    matches.append(var_name)

        return {
            "matches": sorted(matches),
            "cursor_start": cursor_pos - len(word),
            "cursor_end": cursor_pos,
            "status": "ok",
        }


def get_default_kernelspec_dir(user: bool = True) -> Path:
    """Calculates platform-standard Jupyter kernels directory."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", "~")).expanduser()
        dest = base / "jupyter" / "kernels" / "synapse"
    else:
        if user:
            base = Path.home() / ".local" / "share" / "jupyter" / "kernels" / "synapse"
        else:
            base = Path("/usr/local/share/jupyter/kernels/synapse")
        dest = base
    return dest


def install_kernel_spec(user: bool = True, dest_dir: Optional[str] = None) -> Path:
    """
    Installs the Synapse Jupyter Kernel spec file (kernel.json).
    Allows JupyterLab, VS Code, and Notebooks to discover 'Synapse AI'.
    """
    dest = Path(dest_dir) if dest_dir else get_default_kernelspec_dir(user=user)
    dest.mkdir(parents=True, exist_ok=True)

    kernel_json = {
        "argv": [
            sys.executable,
            "-m",
            "synapse.tools.kernel",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Synapse AI",
        "language": "synapse",
        "interrupt_mode": "message",
        "metadata": {
            "debugger": True,
            "synapse_version": "1.0.0",
        },
    }

    manifest_path = dest / "kernel.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(kernel_json, f, indent=2)

    return manifest_path


# Optional IPyKernel bridge if ipykernel is available
try:
    from ipykernel.kernelbase import Kernel

    class SynapseJupyterKernel(Kernel):
        implementation = "Synapse"
        implementation_version = "1.0.0"
        language = "synapse"
        language_version = "1.0.0"
        language_info = {
            "name": "synapse",
            "mimetype": "text/x-synapse",
            "file_extension": ".syn",
        }
        banner = "Synapse AI - High Performance AI-Native Programming Language"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.engine = SynapseKernelEngine()

        def do_execute(
            self, code, silent, store_history=True, user_expressions=None, allow_stdin=False
        ):
            res = self.engine.execute_cell(code)

            if not silent:
                if res["stdout"]:
                    self.send_response(self.iopub_socket, "stream", {"name": "stdout", "text": res["stdout"]})
                if res["stderr"]:
                    self.send_response(self.iopub_socket, "stream", {"name": "stderr", "text": res["stderr"]})

                if res["mime_bundle"]:
                    self.send_response(
                        self.iopub_socket,
                        "execute_result",
                        {
                            "execution_count": res["execution_count"],
                            "data": res["mime_bundle"],
                            "metadata": {},
                        },
                    )

            if res["status"] == "error":
                return {
                    "status": "error",
                    "execution_count": res["execution_count"],
                    "ename": res["error"]["ename"] if res["error"] else "Error",
                    "evalue": res["error"]["evalue"] if res["error"] else "",
                    "traceback": res["error"]["traceback"] if res["error"] else [],
                }

            return {
                "status": "ok",
                "execution_count": res["execution_count"],
                "payload": [],
                "user_expressions": {},
            }

        def do_complete(self, code, cursor_pos):
            return self.engine.complete(code, cursor_pos)

except ImportError:
    SynapseJupyterKernel = None  # type: ignore


def main():
    """CLI entrypoint for kernel installation and launching."""
    args = sys.argv[1:]
    if "install" in args:
        user = "--sys-prefix" not in args
        dest = install_kernel_spec(user=user)
        print(f"[Kernel] Synapse Jupyter Kernel spec installed successfully: {dest}")
        sys.exit(0)

    if SynapseJupyterKernel is not None:
        from ipykernel.kernelapp import IPKernelApp
        IPKernelApp.launch_instance(kernel_class=SynapseJupyterKernel)
    else:
        print(
            "Synapse Kernel Engine ready. To connect to Jupyter ZeroMQ sockets, install ipykernel:\n"
            "  pip install ipykernel\n"
            "Or use 'synapse kernel install' to register kernelspec."
        )


if __name__ == "__main__":
    main()
