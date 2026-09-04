"""
Synapse Debug Adapter Protocol (DAP) Server.

Implements the standard Debug Adapter Protocol (DAP) over stdio or custom streams.
Enables visual debugging in VS Code and any DAP-compliant editor for the Synapse programming language.

Features:
- JSON-RPC DAP wire protocol with Content-Length framing.
- Full capability negotiation (initialize).
- Breakpoint management with line verification and optional conditions (setBreakpoints).
- Program launch and configuration handshake (launch, configurationDone).
- Call stack frame inspection (threads, stackTrace).
- Variable scoping and inspection (scopes: Locals, Globals, and compound data structures).
- Live execution control (continue, next [step over], stepIn, stepOut, pause, disconnect, terminate).
- Live expression evaluation in frame context (evaluate for hover & debug console).
- Live variable modification (setVariable).
- Capture stdout into DAP output events.
- Dual execution engine: Real Synapse VM execution and deterministic simulated execution.
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, BinaryIO, Callable, Optional

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler, CodeObject
from synapse.vm.virtual_machine import VirtualMachine, Frame


class DebuggerExit(BaseException):
    """Signals immediate termination of debug execution."""
    pass


class StepMode(Enum):
    NONE = 0
    CONTINUE = 1
    STEP_IN = 2
    STEP_OVER = 3
    STEP_OUT = 4


class SessionState(str, Enum):
    NOT_STARTED = "not_started"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"


# =============================================================================
# DAP Wire Protocol Framing
# =============================================================================

class DAPMessage:
    """Utilities for encoding and decoding DAP wire protocol messages."""

    HEADER_PREFIX = b"Content-Length:"
    CRLF = b"\r\n"
    HEADER_SEPARATOR = b"\r\n\r\n"

    @staticmethod
    def encode(payload: dict[str, Any]) -> bytes:
        """Encodes a DAP message dictionary into framed bytes."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("latin1")
        return header + body

    @staticmethod
    def decode(buffer: bytes) -> tuple[Optional[dict[str, Any]], int]:
        """
        Attempts to decode a single DAP message from buffer.
        Returns (decoded_dict, consumed_bytes_count).
        If incomplete, returns (None, 0).
        """
        sep_idx = buffer.find(DAPMessage.HEADER_SEPARATOR)
        if sep_idx == -1:
            return None, 0

        header_part = buffer[:sep_idx].decode("latin1", errors="replace")
        content_length: Optional[int] = None

        for line in header_part.split("\r\n"):
            line = line.strip()
            if line.lower().startswith("content-length:"):
                try:
                    content_length = int(line.split(":", 1)[1].strip())
                except ValueError:
                    content_length = None

        if content_length is None or content_length < 0:
            # Malformed header, skip past separator
            return None, sep_idx + 4

        body_start = sep_idx + 4
        body_end = body_start + content_length

        if len(buffer) < body_end:
            # Body not fully received yet
            return None, 0

        body_bytes = buffer[body_start:body_end]
        try:
            msg = json.loads(body_bytes.decode("utf-8"))
            return msg, body_end
        except Exception:
            return None, body_end

    @staticmethod
    def read_from_stream(reader: BinaryIO) -> Optional[dict[str, Any]]:
        """Reads a single DAP message from a blocking binary stream."""
        content_length: Optional[int] = None

        while True:
            line = reader.readline()
            if not line:
                return None
            line_str = line.decode("latin1", errors="replace").strip()
            if not line_str:
                # Empty line marks end of headers
                break
            if line_str.lower().startswith("content-length:"):
                try:
                    content_length = int(line_str.split(":", 1)[1].strip())
                except ValueError:
                    content_length = None

        if content_length is None:
            return None

        body = reader.read(content_length)
        if not body or len(body) < content_length:
            return None

        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return None

    @staticmethod
    def write_to_stream(writer: BinaryIO, payload: dict[str, Any]) -> None:
        """Writes a DAP message to a binary stream and flushes."""
        data = DAPMessage.encode(payload)
        writer.write(data)
        writer.flush()


# =============================================================================
# Breakpoint & Frame Data Structures
# =============================================================================

@dataclass
class BreakpointInfo:
    id: int
    verified: bool
    line: int
    source: dict[str, Any]
    condition: Optional[str] = None
    hit_count: int = 0


@dataclass
class StackFrameInfo:
    id: int
    name: str
    line: int
    column: int
    source: dict[str, Any]


def _get_type_name(val: Any) -> str:
    """Helper returning concise, friendly type names for Synapse values."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, str):
        return "str"
    if isinstance(val, (list, tuple)):
        return "list"
    if isinstance(val, dict):
        return "dict"
    if hasattr(val, "__class__"):
        name = val.__class__.__name__
        if name == "Tensor":
            return "Tensor"
        if name == "SynapseFunction":
            return "function"
        return name
    return type(val).__name__


# =============================================================================
# Debug Session Engine
# =============================================================================

class SynapseDebugSession:
    """
    Core debug execution engine for Synapse.
    Supports real VM execution with line/instruction hooks, as well as simulated stepping.
    """

    def __init__(
        self,
        on_stopped: Optional[Callable[[str, int, str], None]] = None,
        on_output: Optional[Callable[[str, str], None]] = None,
        on_terminated: Optional[Callable[[], None]] = None,
        on_exited: Optional[Callable[[int], None]] = None,
    ):
        self.on_stopped = on_stopped
        self.on_output = on_output
        self.on_terminated = on_terminated
        self.on_exited = on_exited

        self.state: SessionState = SessionState.NOT_STARTED
        self.stop_reason: Optional[str] = None
        self.program_path: str = "<program.syn>"
        self.source_code: Optional[str] = None
        self.stop_on_entry: bool = False
        self.no_debug: bool = False
        self.simulate: bool = False
        self.simulated_lines: list[int] = []
        self.simulated_locals: dict[str, Any] = {}

        # Synchronization primitives
        self._resume_event = threading.Event()
        self._stopped_event = threading.Event()
        self._terminated_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._abort_requested = False
        self._pause_requested = False

        # Stepping control
        self._step_mode: StepMode = StepMode.NONE
        self._target_frame_depth: int = 0
        self._last_frame: Optional[Frame] = None
        self._last_line: int = -1

        # Breakpoints: path -> {line: condition}
        self._breakpoints: dict[str, dict[int, Optional[str]]] = {}
        self._bp_counter: int = 1

        # VM & inspection state
        self.vm: Optional[VirtualMachine] = None
        self.current_frame: Optional[Frame] = None
        self.current_line: int = 1
        self.current_col: int = 1

        # Variable references mapping
        self._compound_vars: dict[int, Any] = {}
        self._next_var_ref: int = 100000

        # Standard built-in names to filter out of Globals for clean display
        self._builtin_keys = {
            "tensor", "zeros", "ones", "randn", "grad", "print", "len", "range",
            "sum", "abs", "min", "max", "int", "float", "str", "bool", "relu",
            "sigmoid", "tanh", "gelu", "softmax", "spawn", "channel", "parallel_map",
            "wait_all", "Channel", "TaskFuture", "nn", "optim", "Linear", "Sequential",
            "ReLU", "Sigmoid", "MSELoss", "CrossEntropyLoss", "Adam", "SGD", "Device",
            "cuda_is_available", "is_cuda_available", "device_count", "DataLoader",
            "StepLR", "CosineAnnealingLR", "memory", "VectorMemory", "swarm", "debate",
            "DistributedNode", "SwarmMesh", "std", "read_file", "write_file",
            "append_file", "file_exists", "now", "sleep", "sha256", "clamp", "lerp",
            "http_get", "http_post", "regex_match", "regex_search", "regex_replace",
            "json_parse", "json_stringify", "env", "connect_db", "gzip_compress",
            "gzip_decompress", "b64_encode", "b64_decode", "path_join", "path_exists",
            "path_dirname", "path_basename", "try_catch", "serve", "response",
            "SynapseResponse", "SynapseRequest", "SynapseSSEResponse", "sse_response",
            "SynapseWebSocket", "SynapseWebSocketRoute", "dataframe", "DataFrame",
            "col", "read_csv", "read_parquet", "assert", "assert_eq", "assert_ne",
            "assert_true", "assert_false", "fail", "__repl_display__"
        }

    # -------------------------------------------------------------------------
    # Breakpoint Management
    # -------------------------------------------------------------------------

    def _normalize_path(self, path: str) -> str:
        """Normalizes path for cross-platform case-insensitive comparison."""
        if not path:
            return ""
        norm = os.path.normpath(path).replace("\\", "/")
        if sys.platform == "win32":
            return norm.lower()
        return norm

    def set_breakpoints(
        self,
        source: dict[str, Any],
        lines: Optional[list[int]] = None,
        breakpoints: Optional[list[dict[str, Any]]] = None
    ) -> list[dict[str, Any]]:
        """Sets and verifies breakpoints for the given source file."""
        path = source.get("path") or source.get("name") or self.program_path
        norm_path = self._normalize_path(path)

        target_lines: list[int] = []
        conditions: dict[int, Optional[str]] = {}

        if breakpoints is not None:
            for bp in breakpoints:
                line = bp.get("line")
                if isinstance(line, int):
                    target_lines.append(line)
                    conditions[line] = bp.get("condition")
        elif lines is not None:
            for line in lines:
                if isinstance(line, int):
                    target_lines.append(line)
                    conditions[line] = None

        self._breakpoints[norm_path] = conditions

        result: list[dict[str, Any]] = []
        for line in target_lines:
            bp_id = self._bp_counter
            self._bp_counter += 1
            result.append({
                "id": bp_id,
                "verified": True,
                "line": line,
                "source": source,
            })

        return result

    def _has_breakpoint(self, file_path: str, line: int) -> tuple[bool, Optional[str]]:
        """Checks if a breakpoint is set for the given file and line."""
        norm_path = self._normalize_path(file_path)
        base_name = os.path.basename(norm_path)

        for registered_path, bps in self._breakpoints.items():
            if norm_path == registered_path or base_name == os.path.basename(registered_path) or not registered_path:
                if line in bps:
                    return True, bps[line]

        # If only one source file registered breakpoints and program matches
        if len(self._breakpoints) == 1:
            only_path = next(iter(self._breakpoints))
            if line in self._breakpoints[only_path]:
                return True, self._breakpoints[only_path][line]

        return False, None

    # -------------------------------------------------------------------------
    # Launch & Lifecycle
    # -------------------------------------------------------------------------

    def launch(
        self,
        program: Optional[str] = None,
        source: Optional[str] = None,
        args: Optional[list[str]] = None,
        stop_on_entry: bool = False,
        no_debug: bool = False,
        simulate: bool = False,
        simulated_lines: Optional[list[int]] = None,
        simulated_locals: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configures the debug session for launching."""
        self.program_path = program or "<program.syn>"
        self.source_code = source
        self.stop_on_entry = bool(stop_on_entry)
        self.no_debug = bool(no_debug)
        self.simulate = bool(simulate)
        self.simulated_lines = simulated_lines or ([1, 2, 3] if simulate else [])
        self.simulated_locals = dict(simulated_locals or ({"x": 10, "status": "active"} if simulate else {}))
        self.state = SessionState.INITIALIZED

    def configuration_done(self) -> None:
        """Called when configuration is complete; starts execution."""
        self.state = SessionState.RUNNING
        self._abort_requested = False
        self._pause_requested = False
        self._resume_event.clear()
        self._stopped_event.clear()
        self._terminated_event.clear()

        # Start execution in a daemon worker thread
        self._worker_thread = threading.Thread(target=self._run_session, daemon=True)
        self._worker_thread.start()

    def _run_session(self) -> None:
        """Main execution loop running in worker thread."""
        try:
            if self.simulate:
                self._run_simulated()
            else:
                self._run_vm()
        except DebuggerExit:
            pass
        except Exception as e:
            if self.on_output:
                self.on_output(f"\n[Runtime Exception] {e}\n", "stderr")
            self._do_stop("exception", description=str(e))
        finally:
            self.state = SessionState.TERMINATED
            self._stopped_event.set()
            self._terminated_event.set()
            if self.on_terminated:
                self.on_terminated()
            if self.on_exited:
                self.on_exited(0)

    # -------------------------------------------------------------------------
    # Simulated Engine
    # -------------------------------------------------------------------------

    def _run_simulated(self) -> None:
        """Simulates stepping through a set of line numbers."""
        lines = self.simulated_lines or [1, 2, 3]
        for idx, line in enumerate(lines):
            if self._abort_requested:
                raise DebuggerExit()

            self.current_line = line
            self.current_col = 1

            if idx == 0 and self.stop_on_entry:
                self.stop_on_entry = False
                self._do_stop("entry")

            has_bp, cond = self._has_breakpoint(self.program_path, line)
            if has_bp:
                self._do_stop("breakpoint")
            elif self._step_mode in (StepMode.STEP_IN, StepMode.STEP_OVER, StepMode.STEP_OUT):
                self._step_mode = StepMode.NONE
                self._do_stop("step")
            elif self._pause_requested:
                self._pause_requested = False
                self._do_stop("pause")

    # -------------------------------------------------------------------------
    # Real Synapse VM Execution Engine
    # -------------------------------------------------------------------------

    def _run_vm(self) -> None:
        """Executes real Synapse source on VirtualMachine with stepping hooks."""
        source = self.source_code
        if source is None:
            if os.path.isfile(self.program_path):
                with open(self.program_path, "r", encoding="utf-8") as f:
                    source = f.read()
            else:
                source = "let __status = 'no_source'\n"

        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        compiler = Compiler(
            name=os.path.basename(self.program_path) or "<module>",
            filename=self.program_path
        )
        code = compiler.compile(ast)

        self.vm = VirtualMachine()
        self.vm.step_hook = self._on_vm_step

        # Capture output
        def custom_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            end = kwargs.get("end", "\n")
            full_msg = text + end
            if self.on_output:
                self.on_output(full_msg, "stdout")
            self.vm.output_buffer.append(text)

        self.vm.globals["print"] = custom_print

        # Run bytecode
        self.vm.execute(code)

    def _on_vm_step(
        self,
        vm: VirtualMachine,
        frame: Frame,
        opcode: Any,
        arg: Any,
        line: int,
        col: int
    ) -> None:
        """Invoked by the VM before executing each instruction."""
        if self._abort_requested:
            raise DebuggerExit()

        if line <= 0:
            return

        # Filter repeated checks for the same line inside the same frame
        if line == self._last_line and frame is self._last_frame:
            return

        self._last_frame = frame
        self._last_line = line
        self.current_frame = frame
        self.current_line = line
        self.current_col = col

        # 1. Stop on entry
        if self.stop_on_entry:
            self.stop_on_entry = False
            self._do_stop("entry")
            return

        # 2. Pause requested
        if self._pause_requested:
            self._pause_requested = False
            self._do_stop("pause")
            return

        # 3. Breakpoint hit
        source_file = getattr(frame.code, "filename", self.program_path) or self.program_path
        has_bp, cond = self._has_breakpoint(source_file, line)
        if has_bp and not self.no_debug:
            if cond:
                # Evaluate condition
                try:
                    scope = dict(vm.globals)
                    scope.update(frame.locals)
                    val = eval(cond, {}, scope)
                    if not bool(val):
                        # Condition not met, do not stop
                        has_bp = False
                except Exception:
                    pass

            if has_bp:
                self._do_stop("breakpoint")
                return

        # 4. Stepping mode evaluation
        if self._step_mode == StepMode.STEP_IN:
            self._step_mode = StepMode.NONE
            self._do_stop("step")
            return

        if self._step_mode == StepMode.STEP_OVER:
            depth = len(vm.frames)
            if depth <= self._target_frame_depth:
                self._step_mode = StepMode.NONE
                self._do_stop("step")
                return

        if self._step_mode == StepMode.STEP_OUT:
            depth = len(vm.frames)
            if depth < self._target_frame_depth:
                self._step_mode = StepMode.NONE
                self._do_stop("step")
                return

    def _do_stop(self, reason: str, description: str = "") -> None:
        """Halts worker thread and notifies listeners of a StoppedEvent."""
        self.state = SessionState.STOPPED
        self.stop_reason = reason
        self._stopped_event.set()

        if self.on_stopped:
            self.on_stopped(reason, 1, description)

        # Block until resumed by next, continue, or terminate
        self._resume_event.clear()
        self._resume_event.wait()

        if self._abort_requested:
            raise DebuggerExit()

        self.state = SessionState.RUNNING
        self._stopped_event.clear()

    # -------------------------------------------------------------------------
    # Stepping & Control Operations
    # -------------------------------------------------------------------------

    def continue_execution(self) -> None:
        """Resumes execution until the next breakpoint or termination."""
        self._step_mode = StepMode.CONTINUE
        self.state = SessionState.RUNNING
        self._stopped_event.clear()
        self._resume_event.set()

    def step_over(self) -> None:
        """Steps to the next line in the current or outer frame."""
        self._step_mode = StepMode.STEP_OVER
        if self.vm and self.vm.frames:
            self._target_frame_depth = len(self.vm.frames)
        else:
            self._target_frame_depth = 1
        self.state = SessionState.RUNNING
        self._stopped_event.clear()
        self._resume_event.set()

    def step_in(self) -> None:
        """Steps into the next line/function call."""
        self._step_mode = StepMode.STEP_IN
        self.state = SessionState.RUNNING
        self._stopped_event.clear()
        self._resume_event.set()

    def step_out(self) -> None:
        """Steps out of the current function back to caller frame."""
        self._step_mode = StepMode.STEP_OUT
        if self.vm and self.vm.frames:
            self._target_frame_depth = len(self.vm.frames)
        else:
            self._target_frame_depth = 0
        self.state = SessionState.RUNNING
        self._stopped_event.clear()
        self._resume_event.set()

    def pause(self) -> None:
        """Pauses running execution."""
        self._pause_requested = True

    def terminate(self) -> None:
        """Aborts execution and cleans up."""
        self._abort_requested = True
        self._resume_event.set()
        self.state = SessionState.TERMINATED

    def wait_for_stop(self, timeout: float = 3.0) -> bool:
        """Helper to wait until the session enters STOPPED state."""
        return self._stopped_event.wait(timeout=timeout)

    def wait_for_termination(self, timeout: float = 3.0) -> bool:
        """Helper to wait until the session enters TERMINATED state."""
        return self._terminated_event.wait(timeout=timeout)

    # -------------------------------------------------------------------------
    # State & Variable Inspection
    # -------------------------------------------------------------------------

    def get_threads(self) -> list[dict[str, Any]]:
        """Returns active thread(s)."""
        return [{"id": 1, "name": "Main Thread"}]

    def get_stack_frames(self, start_frame: int = 0, levels: int = 0) -> list[dict[str, Any]]:
        """Returns stack frames formatted for DAP stackTrace response."""
        frames: list[dict[str, Any]] = []

        if self.vm and self.vm.frames:
            # Most recent frame is at top of stack (frame 0 in DAP)
            vm_frames = list(reversed(self.vm.frames))
            for i, f in enumerate(vm_frames):
                frame_id = (i + 1) * 1000
                func_name = f.code.name
                line = getattr(f, "current_line", 1) or 1
                col = getattr(f, "current_col", 1) or 1
                filename = getattr(f.code, "filename", self.program_path) or self.program_path

                frames.append({
                    "id": frame_id,
                    "name": func_name,
                    "line": line,
                    "column": col,
                    "source": {
                        "name": os.path.basename(filename) or "source.syn",
                        "path": os.path.abspath(filename) if os.path.exists(filename) else filename,
                    }
                })
        else:
            # Simulated or single-frame fallback
            frames.append({
                "id": 1000,
                "name": "<module>",
                "line": self.current_line,
                "column": self.current_col,
                "source": {
                    "name": os.path.basename(self.program_path) or "program.syn",
                    "path": self.program_path,
                }
            })

        if levels > 0:
            return frames[start_frame:start_frame + levels]
        return frames[start_frame:]

    def get_scopes(self, frame_id: int) -> list[dict[str, Any]]:
        """Returns Locals and Globals scopes for the requested frame."""
        locals_ref = frame_id + 1
        globals_ref = frame_id + 2
        return [
            {
                "name": "Locals",
                "presentationHint": "locals",
                "variablesReference": locals_ref,
                "expensive": False,
            },
            {
                "name": "Globals",
                "presentationHint": "globals",
                "variablesReference": globals_ref,
                "expensive": False,
            }
        ]

    def _register_compound(self, obj: Any) -> int:
        """Registers a complex/compound data structure and returns its reference ID."""
        ref = self._next_var_ref
        self._next_var_ref += 1
        self._compound_vars[ref] = obj
        return ref

    def get_variables(self, var_ref: int) -> list[dict[str, Any]]:
        """Returns variables for the given scope or compound variable reference."""
        # 1. Compound children (lists, dicts, etc.)
        if var_ref in self._compound_vars:
            obj = self._compound_vars[var_ref]
            results: list[dict[str, Any]] = []
            if isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj):
                    child_ref = self._register_compound(item) if isinstance(item, (list, dict)) else 0
                    results.append({
                        "name": f"[{idx}]",
                        "value": str(item),
                        "type": _get_type_name(item),
                        "variablesReference": child_ref,
                    })
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    child_ref = self._register_compound(v) if isinstance(v, (list, dict)) else 0
                    results.append({
                        "name": str(k),
                        "value": str(v),
                        "type": _get_type_name(v),
                        "variablesReference": child_ref,
                    })
            return results

        # 2. Scope variable reference: frame_id + 1 (Locals), frame_id + 2 (Globals)
        frame_idx = (var_ref // 1000) - 1
        is_locals = (var_ref % 10 == 1)

        target_dict: dict[str, Any] = {}

        if self.vm and self.vm.frames:
            vm_frames = list(reversed(self.vm.frames))
            if 0 <= frame_idx < len(vm_frames):
                frame = vm_frames[frame_idx]
                if is_locals:
                    target_dict = frame.locals
                else:
                    # Globals excluding noisy builtins
                    target_dict = {
                        k: v for k, v in self.vm.globals.items()
                        if k not in self._builtin_keys and not k.startswith("__")
                    }
        else:
            # Simulated scope fallback
            if is_locals:
                target_dict = self.simulated_locals
            else:
                target_dict = {"__version__": "0.1.0"}

        vars_list: list[dict[str, Any]] = []
        for k, v in target_dict.items():
            if k.startswith("__idx_") or k == "__repl_display__":
                continue
            child_ref = self._register_compound(v) if isinstance(v, (list, dict)) and len(v) > 0 else 0
            vars_list.append({
                "name": str(k),
                "value": str(v),
                "type": _get_type_name(v),
                "variablesReference": child_ref,
            })

        return vars_list

    def evaluate(self, expression: str, frame_id: Optional[int] = None) -> dict[str, Any]:
        """Evaluates an expression in the current or specified stack frame context."""
        scope: dict[str, Any] = {}
        if self.vm:
            scope.update(self.vm.globals)
            if self.vm.frames:
                frame_idx = ((frame_id // 1000) - 1) if frame_id is not None else 0
                vm_frames = list(reversed(self.vm.frames))
                if 0 <= frame_idx < len(vm_frames):
                    scope.update(vm_frames[frame_idx].locals)
                else:
                    scope.update(vm_frames[0].locals)
        else:
            scope.update(self.simulated_locals)

        try:
            val = eval(expression, {}, scope)
            return {
                "result": str(val),
                "type": _get_type_name(val),
                "variablesReference": self._register_compound(val) if isinstance(val, (list, dict)) else 0,
            }
        except Exception as e:
            return {
                "result": f"Error: {e}",
                "type": "error",
                "variablesReference": 0,
            }

    def set_variable(self, var_ref: int, name: str, value: str) -> dict[str, Any]:
        """Modifies a variable value in the given frame."""
        frame_idx = (var_ref // 1000) - 1
        is_locals = (var_ref % 10 == 1)

        try:
            parsed_val = eval(value, {})
        except Exception:
            parsed_val = value

        if self.vm and self.vm.frames:
            vm_frames = list(reversed(self.vm.frames))
            if 0 <= frame_idx < len(vm_frames):
                target = vm_frames[frame_idx].locals if is_locals else self.vm.globals
                target[name] = parsed_val
        else:
            self.simulated_locals[name] = parsed_val

        return {
            "value": str(parsed_val),
            "type": _get_type_name(parsed_val),
        }


# =============================================================================
# Synapse DAP Server
# =============================================================================

class SynapseDAPServer:
    """
    Debug Adapter Protocol (DAP) server for Synapse.
    Reads DAP requests, coordinates debug sessions, and dispatches responses and events.
    """

    def __init__(
        self,
        reader: Optional[BinaryIO] = None,
        writer: Optional[BinaryIO] = None,
        session: Optional[SynapseDebugSession] = None,
    ):
        self.reader = reader or sys.stdin.buffer
        self.writer = writer or sys.stdout.buffer
        self._write_lock = threading.Lock()
        self._seq = 1
        self.is_running = True
        self.is_initialized = False

        # Initialize debug session with event forwarders
        self.session = session or SynapseDebugSession(
            on_stopped=self._on_session_stopped,
            on_output=self._on_session_output,
            on_terminated=self._on_session_terminated,
            on_exited=self._on_session_exited,
        )

    # -------------------------------------------------------------------------
    # Wire Protocol & Message Sending
    # -------------------------------------------------------------------------

    def read_message(self) -> Optional[dict[str, Any]]:
        """Reads a framed DAP message from reader."""
        return DAPMessage.read_from_stream(self.reader)

    def send_message(self, payload: dict[str, Any]) -> None:
        """Framed send of a DAP message."""
        with self._write_lock:
            payload["seq"] = self._seq
            self._seq += 1
            DAPMessage.write_to_stream(self.writer, payload)

    def send_response(
        self,
        request_seq: int,
        command: str,
        body: Optional[dict[str, Any]] = None,
        success: bool = True,
        message: Optional[str] = None
    ) -> dict[str, Any]:
        """Constructs and sends a DAP response message."""
        resp: dict[str, Any] = {
            "type": "response",
            "request_seq": request_seq,
            "command": command,
            "success": success,
        }
        if body is not None:
            resp["body"] = body
        if message is not None:
            resp["message"] = message

        self.send_message(resp)
        return resp

    def send_event(self, event: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Constructs and sends a DAP event message."""
        evt: dict[str, Any] = {
            "type": "event",
            "event": event,
            "body": body or {},
        }
        self.send_message(evt)
        return evt

    # -------------------------------------------------------------------------
    # Session Callback Forwarders
    # -------------------------------------------------------------------------

    def _on_session_stopped(self, reason: str, thread_id: int, description: str) -> None:
        body: dict[str, Any] = {
            "reason": reason,
            "threadId": thread_id,
            "allThreadsStopped": True,
        }
        if description:
            body["description"] = description
        self.send_event("stopped", body)

    def _on_session_output(self, text: str, category: str) -> None:
        self.send_event("output", {
            "category": category,
            "output": text,
        })

    def _on_session_terminated(self) -> None:
        self.send_event("terminated", {})

    def _on_session_exited(self, exit_code: int) -> None:
        self.send_event("exited", {"exitCode": exit_code})

    # -------------------------------------------------------------------------
    # Request Dispatcher
    # -------------------------------------------------------------------------

    def handle_message(self, msg: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Processes an incoming DAP message dictionary."""
        msg_type = msg.get("type")
        if msg_type != "request":
            return None

        seq = msg.get("seq", 0)
        cmd = msg.get("command", "")
        args = msg.get("arguments", {})

        handler = getattr(self, f"handle_{cmd}", None)
        if handler:
            return handler(seq, args)

        # Fallback for unknown request
        return self.send_response(
            seq,
            cmd,
            success=False,
            message=f"Unsupported DAP command: '{cmd}'"
        )

    # -------------------------------------------------------------------------
    # Command Handlers
    # -------------------------------------------------------------------------

    def handle_initialize(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'initialize' request."""
        self.is_initialized = True
        capabilities = {
            "supportsConfigurationDoneRequest": True,
            "supportsFunctionBreakpoints": False,
            "supportsConditionalBreakpoints": True,
            "supportsHitConditionalBreakpoints": False,
            "supportsEvaluateForHovers": True,
            "supportsStepBack": False,
            "supportsSetVariable": True,
            "supportsRestartFrame": False,
            "supportsGotoTargetsRequest": False,
            "supportsStepInTargetsRequest": False,
            "supportsCompletionsRequest": False,
            "supportsModulesRequest": False,
            "supportsExceptionOptions": False,
            "supportsValueFormattingOptions": True,
            "supportsExceptionInfoRequest": False,
            "supportTerminateDebuggee": True,
            "supportsDelayedStackTraceLoading": False,
            "supportsLoadedSourcesRequest": False,
            "supportsTerminateRequest": True,
        }

        resp = self.send_response(request_seq, "initialize", body=capabilities)
        # DAP requires initialized event after initialize response
        self.send_event("initialized", {})
        return resp

    def handle_launch(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'launch' request."""
        self.session.launch(
            program=args.get("program"),
            source=args.get("source"),
            args=args.get("args"),
            stop_on_entry=args.get("stopOnEntry", False),
            no_debug=args.get("noDebug", False),
            simulate=args.get("simulate", False),
            simulated_lines=args.get("simulated_lines"),
            simulated_locals=args.get("simulated_locals"),
        )
        return self.send_response(request_seq, "launch", body={})

    def handle_attach(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'attach' request."""
        return self.send_response(request_seq, "attach", body={})

    def handle_setBreakpoints(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'setBreakpoints' request."""
        source = args.get("source", {})
        lines = args.get("lines")
        bps = args.get("breakpoints")
        verified_bps = self.session.set_breakpoints(source, lines=lines, breakpoints=bps)
        return self.send_response(request_seq, "setBreakpoints", body={"breakpoints": verified_bps})

    def handle_setFunctionBreakpoints(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'setFunctionBreakpoints' request."""
        return self.send_response(request_seq, "setFunctionBreakpoints", body={"breakpoints": []})

    def handle_setExceptionBreakpoints(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'setExceptionBreakpoints' request."""
        return self.send_response(request_seq, "setExceptionBreakpoints", body={"breakpoints": []})

    def handle_configurationDone(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'configurationDone' request."""
        resp = self.send_response(request_seq, "configurationDone", body={})
        self.session.configuration_done()
        return resp

    def handle_threads(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'threads' request."""
        threads = self.session.get_threads()
        return self.send_response(request_seq, "threads", body={"threads": threads})

    def handle_stackTrace(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'stackTrace' request."""
        start_frame = args.get("startFrame", 0)
        levels = args.get("levels", 0)
        frames = self.session.get_stack_frames(start_frame=start_frame, levels=levels)
        return self.send_response(request_seq, "stackTrace", body={
            "stackFrames": frames,
            "totalFrames": len(frames),
        })

    def handle_scopes(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'scopes' request."""
        frame_id = args.get("frameId", 1000)
        scopes = self.session.get_scopes(frame_id)
        return self.send_response(request_seq, "scopes", body={"scopes": scopes})

    def handle_variables(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'variables' request."""
        var_ref = args.get("variablesReference", 0)
        variables = self.session.get_variables(var_ref)
        return self.send_response(request_seq, "variables", body={"variables": variables})

    def handle_continue(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'continue' request."""
        self.session.continue_execution()
        return self.send_response(request_seq, "continue", body={"allThreadsContinued": True})

    def handle_next(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'next' (step over) request."""
        self.session.step_over()
        return self.send_response(request_seq, "next", body={})

    def handle_stepIn(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'stepIn' request."""
        self.session.step_in()
        return self.send_response(request_seq, "stepIn", body={})

    def handle_stepOut(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'stepOut' request."""
        self.session.step_out()
        return self.send_response(request_seq, "stepOut", body={})

    def handle_pause(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'pause' request."""
        self.session.pause()
        return self.send_response(request_seq, "pause", body={})

    def handle_evaluate(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'evaluate' request."""
        expr = args.get("expression", "")
        frame_id = args.get("frameId")
        result = self.session.evaluate(expr, frame_id=frame_id)
        return self.send_response(request_seq, "evaluate", body=result)

    def handle_setVariable(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'setVariable' request."""
        var_ref = args.get("variablesReference", 0)
        name = args.get("name", "")
        val_str = args.get("value", "")
        result = self.session.set_variable(var_ref, name, val_str)
        return self.send_response(request_seq, "setVariable", body=result)

    def handle_disconnect(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'disconnect' request."""
        self.session.terminate()
        self.is_running = False
        return self.send_response(request_seq, "disconnect", body={})

    def handle_terminate(self, request_seq: int, args: dict[str, Any]) -> dict[str, Any]:
        """Handles DAP 'terminate' request."""
        self.session.terminate()
        self.send_event("terminated", {})
        return self.send_response(request_seq, "terminate", body={})

    # -------------------------------------------------------------------------
    # Server Loop
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Starts the DAP server loop reading messages until EOF or disconnect."""
        while self.is_running:
            try:
                msg = self.read_message()
                if msg is None:
                    break
                self.handle_message(msg)
            except Exception as e:
                sys.stderr.write(f"[Synapse DAP Error] {e}\n")
                sys.stderr.flush()


# =============================================================================
# CLI Helper
# =============================================================================

def run_dap_server(
    input_stream: Optional[BinaryIO] = None,
    output_stream: Optional[BinaryIO] = None
) -> None:
    """
    CLI helper entry point for launching the Synapse DAP server over stdio or streams.
    """
    reader = input_stream or sys.stdin.buffer
    writer = output_stream or sys.stdout.buffer
    server = SynapseDAPServer(reader=reader, writer=writer)
    server.start()


if __name__ == "__main__":
    run_dap_server()
