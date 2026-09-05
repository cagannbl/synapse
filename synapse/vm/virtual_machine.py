import importlib
import json
import math
import os
import time
from typing import Any, Callable, Optional
from synapse.vm.opcodes import Opcode
from synapse.vm.compiler import CodeObject
from synapse.core.tensor import Tensor, tensor, zeros, ones, randn, QuantizedTensor, quantize, dequantize
from synapse.core.autograd import grad
from synapse.core.generators import TokenStream, SynapseGenerator, StreamPipeline
from synapse.orm import Database, Field, Model, QuerySet
from synapse.nn.safetensors import safe_open, load_file as load_safetensors, save_file as save_safetensors
from synapse.core.arrow_ipc import read_arrow_ipc, write_arrow_ipc
import synapse.nn as nn
import synapse.optim as optim
from synapse.core.device import Device
from synapse.core.cuda_backend import is_cuda_available, device_count
from synapse.nn.dataloader import DataLoader
from synapse.optim.lr_scheduler import StepLR, CosineAnnealingLR
from synapse.core.memory import memory, VectorMemory
from synapse.ai.agent_swarm import swarm, debate, DistributedNode, SwarmMesh
from synapse.core.web import serve, SynapseResponse, SynapseRequest, SynapseSSEResponse, SynapseWebSocket, SynapseWebSocketRoute
from synapse.core.task_pool import spawn, channel, parallel_map, wait_all, Channel, TaskFuture
from synapse.core.dataframe import dataframe, DataFrame, col, read_csv, read_parquet
from synapse.stdlib import std


# Opcode.STORE_ATTR fallback if not yet defined in opcodes.py
if not hasattr(Opcode, "STORE_ATTR"):
    try:
        setattr(Opcode, "STORE_ATTR", "STORE_ATTR")
    except Exception:
        pass


class VMRuntimeError(Exception):
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        if line > 0:
            if column > 0:
                full_msg = f"[Line {line}:{column}] {message}"
            else:
                full_msg = f"[Line {line}] {message}"
        else:
            full_msg = message
        super().__init__(full_msg)


# =============================================================================
# Built-in Standard Library Helpers (Zero-Dependency)
# =============================================================================

read_file = std.fs.read_file
write_file = std.fs.write_file
append_file = std.fs.append_file
file_exists = std.fs.file_exists
sleep = std.time.sleep
http_get = std.http.get
http_post = std.http.post
regex_match = std.regex.match
regex_search = std.regex.search
regex_replace = std.regex.replace


def json_parse(text: str) -> Any:
    """JSON metnini Python/Synapse veri yapısına dönüştürür."""
    return json.loads(text)


def json_stringify(obj: Any, indent: int = 2) -> str:
    """Python/Synapse nesnesini JSON string formatına dönüştürür."""
    return json.dumps(obj, indent=indent, ensure_ascii=False)



def env(key: str, default: str = "") -> str:
    """Ortam değişkeni değerini döner."""
    return os.environ.get(key, default)


def _syn_assert(condition: Any, message: str = "Assertion failed"):
    if not condition:
        raise AssertionError(str(message))


def _syn_assert_eq(actual: Any, expected: Any, message: Optional[str] = None):
    if actual != expected:
        raise AssertionError(message or f"Expected {expected!r}, got {actual!r}")


def _syn_assert_ne(actual: Any, expected: Any, message: Optional[str] = None):
    if actual == expected:
        raise AssertionError(message or f"Expected values to differ, but both are {actual!r}")


def _syn_assert_true(val: Any, message: str = "Expected true"):
    if not bool(val):
        raise AssertionError(str(message))


def _syn_assert_false(val: Any, message: str = "Expected false"):
    if bool(val):
        raise AssertionError(str(message))


def _syn_fail(message: str = "Test failed"):
    raise AssertionError(str(message))



class SynapseFunction:
    def __init__(self, code: CodeObject, env: dict[str, Any]):
        self.code = code
        self.env = env
        self.name = code.name

    def __call__(self, vm: "VirtualMachine", *args, **kwargs) -> Any:
        # Yeni bir frame aç ve argümanları locals'a ata
        frame_locals = dict(self.env)
        for i, param_name in enumerate(self.code.params):
            if i < len(args):
                frame_locals[param_name] = args[i]
            elif param_name in kwargs:
                frame_locals[param_name] = kwargs[param_name]

        return vm.run_code(self.code, frame_locals)


from synapse.ai.prompt_engine import PromptEngine
from synapse.ai.agent_runtime import AgentRuntime


class PromptObject:
    def __init__(self, name: str, params: list[str], fields: dict[str, Any], vm: "VirtualMachine"):
        self.name = name
        self.params = params
        self.fields = fields
        self.vm = vm
        self.engine = PromptEngine()

    def __call__(self, *args, **kwargs) -> Any:
        param_dict = {}
        for i, p in enumerate(self.params):
            if i < len(args):
                param_dict[p] = args[i]
            elif p in kwargs:
                param_dict[p] = kwargs[p]

        sys_prompt = self._eval_field("system", param_dict) or "You are an AI assistant."
        user_prompt = self._eval_field("user", param_dict) or ""
        temperature = self._eval_field("temperature", param_dict) or 0.7

        return self.engine.execute(
            system_template=str(sys_prompt),
            user_template=str(user_prompt),
            params=param_dict,
            temperature=float(temperature)
        )

    def _eval_field(self, field_name: str, param_dict: dict[str, Any]) -> Any:
        if field_name not in self.fields:
            return None
        expr = self.fields[field_name]
        if hasattr(expr, "value"):
            val = expr.value
            if isinstance(val, str):
                for k, v in param_dict.items():
                    val = val.replace(f"{{{k}}}", str(v))
                return val
            return val
        elif hasattr(expr, "name") and expr.name in param_dict:
            return param_dict[expr.name]
        return str(expr)


class Frame:
    def __init__(self, code: CodeObject, locals_dict: dict[str, Any]):
        self.code = code
        self.pc = 0
        self.locals = locals_dict
        self.stack: list[Any] = []
        self.current_line: int = 1
        self.current_col: int = 1


class VirtualMachine:
    def __init__(self):
        self.globals: dict[str, Any] = {}
        self._setup_builtins()
        self.frames: list[Frame] = []
        self.output_buffer: list[str] = []
        self.step_hook: Optional[Callable[["VirtualMachine", Frame, Opcode, Any, int, int], None]] = None

    def _setup_builtins(self):
        self.globals = {
            "tensor": tensor,
            "zeros": zeros,
            "ones": ones,
            "randn": randn,
            "grad": grad,
            "print": print,
            "len": len,
            "range": range,
            "sum": sum,
            "abs": abs,
            "min": min,
            "max": max,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "relu": lambda t: t.relu() if hasattr(t, "relu") else max(0, t),
            "sigmoid": lambda t: t.sigmoid() if hasattr(t, "sigmoid") else (1 / (1 + 2.71828 ** (-t))),
            "tanh": lambda t: t.tanh() if hasattr(t, "tanh") else math.tanh(t),
            "gelu": lambda t: t.gelu() if hasattr(t, "gelu") else (0.5 * t * (1.0 + math.tanh(0.7978845608 * (t + 0.044715 * t ** 3)))),
            "softmax": lambda t, axis=-1: t.softmax(axis=axis) if hasattr(t, "softmax") else t,
            # Phase 1: No-GIL Multi-Threading & Work-Stealing Task Scheduler
            "spawn": lambda fn, *args, **kwargs: (
                spawn(lambda *a, **kw: fn(self._clone_for_worker(), *a, **kw), *args, **kwargs)
                if isinstance(fn, SynapseFunction) else spawn(fn, *args, **kwargs)
            ),
            "channel": channel,
            "parallel_map": lambda fn, items, max_workers=None: (
                parallel_map(lambda item: fn(self._clone_for_worker(), item), items, max_workers=max_workers)
                if isinstance(fn, SynapseFunction) else parallel_map(fn, items, max_workers=max_workers)
            ),
            "wait_all": wait_all,
            "Channel": Channel,
            "TaskFuture": TaskFuture,
            # AI Neural Network Library (nn & optim)
            "nn": nn,
            "optim": optim,
            "Linear": nn.Linear,
            "Sequential": nn.Sequential,
            "ReLU": nn.ReLU,
            "Sigmoid": nn.Sigmoid,
            "MSELoss": nn.MSELoss,
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            # Hardware & Device Management
            "Device": Device,
            "cuda_is_available": is_cuda_available,
            "is_cuda_available": is_cuda_available,
            "device_count": device_count,
            # DataLoader & Learning Rate Schedulers
            "DataLoader": DataLoader,
            "StepLR": StepLR,
            "CosineAnnealingLR": CosineAnnealingLR,
            # AI Semantic Memory & Swarm Orchestration
            "memory": memory,
            "VectorMemory": VectorMemory,
            "swarm": swarm,
            "debate": debate,
            "DistributedNode": DistributedNode,
            "SwarmMesh": SwarmMesh,
            # Built-in Standard Library (std)
            "std": std,
            # Convenient global shortcuts
            "read_file": std.fs.read_file,
            "write_file": std.fs.write_file,
            "append_file": std.fs.append_file,
            "file_exists": std.fs.file_exists,
            "now": std.time.now,
            "sleep": std.time.sleep,
            "sha256": std.crypto.sha256,
            "clamp": std.math.clamp,
            "lerp": std.math.lerp,
            "http_get": std.http.get,
            "http_post": std.http.post,
            "regex_match": std.regex.match,
            "regex_search": std.regex.search,
            "regex_replace": std.regex.replace,
            "json_parse": json_parse,
            "json_stringify": json_stringify,
            "env": env,
            # Standard Library Extended Shortcuts
            "connect_db": std.db.connect,
            "gzip_compress": std.compress.gzip_compress,
            "gzip_decompress": std.compress.gzip_decompress,
            "b64_encode": std.compress.b64_encode,
            "b64_decode": std.compress.b64_decode,
            "path_join": std.path.join,
            "path_exists": std.path.exists,
            "path_dirname": std.path.dirname,
            "path_basename": std.path.basename,
            "try_catch": lambda try_fn, catch_fn=None, default=None: (
                self._safe_try_catch(try_fn, catch_fn, default)
            ),
            # Native Web Server & HTTP Engine
            "serve": lambda port=8080, routes=None, host="127.0.0.1", static_dir="public", blocking=True, ws_routes=None: serve(
                port=port, routes=routes, ws_routes=ws_routes, host=host, static_dir=static_dir, blocking=blocking, vm_ref=self
            ),
            "response": SynapseResponse,
            "SynapseResponse": SynapseResponse,
            "SynapseRequest": SynapseRequest,
            "SynapseSSEResponse": SynapseSSEResponse,
            "sse_response": SynapseSSEResponse,
            "SynapseWebSocket": SynapseWebSocket,
            "SynapseWebSocketRoute": SynapseWebSocketRoute,
            # Phase 2: Data Science & Arrow-style Columnar DataFrame
            "dataframe": dataframe,
            "DataFrame": DataFrame,
            "col": col,
            "read_csv": read_csv,
            "read_parquet": read_parquet,
            "read_arrow_ipc": read_arrow_ipc,
            "write_arrow_ipc": write_arrow_ipc,
            # Quantized Tensors & Edge AI
            "QuantizedTensor": QuantizedTensor,
            "quantize": quantize,
            "dequantize": dequantize,
            # Enterprise ORM & Database
            "Database": Database,
            "Field": Field,
            "Model": Model,
            "QuerySet": QuerySet,
            # TokenStream & Lazy Streaming
            "TokenStream": TokenStream,
            "SynapseGenerator": SynapseGenerator,
            "StreamPipeline": StreamPipeline,
            # Hugging Face SafeTensors
            "safe_open": safe_open,
            "load_safetensors": load_safetensors,
            "save_safetensors": save_safetensors,
            # Native Assertions & Test Primitives
            "assert": _syn_assert,
            "assert_eq": _syn_assert_eq,
            "assert_ne": _syn_assert_ne,
            "assert_true": _syn_assert_true,
            "assert_false": _syn_assert_false,
            "fail": _syn_fail,
            # Interactive REPL printer
            "__repl_display__": lambda val: print(val) if val is not None else None,
        }

    def _safe_try_catch(self, try_fn: Any, catch_fn: Any = None, default: Any = None) -> Any:
        """Safely executes try_fn with fallback catch_fn or default value."""
        try:
            if callable(try_fn):
                if hasattr(try_fn, "code") and hasattr(try_fn, "env"):
                    return try_fn(self)
                return try_fn()
            return try_fn
        except Exception as e:
            if callable(catch_fn):
                if hasattr(catch_fn, "code") and hasattr(catch_fn, "env"):
                    return catch_fn(self, str(e))
                return catch_fn(str(e))
            return default if default is not None else str(e)

    def set_custom_print(self):
        """Testler ve REPL için stdout yerine buffer'a yazan print."""
        def custom_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            self.output_buffer.append(msg)
            print(*args, **kwargs)
        self.globals["print"] = custom_print

    def _clone_for_worker(self) -> "VirtualMachine":
        """Asenkron iş parçacıkları için izole frame stack ve ortak globals ile VM üretir."""
        worker = VirtualMachine()
        worker.globals = dict(self.globals)
        worker.step_hook = getattr(self, "step_hook", None)
        return worker

    def execute(self, code: CodeObject) -> Any:
        return self.run_code(code, dict(self.globals))

    def run_code(self, code: CodeObject, initial_locals: dict[str, Any]) -> Any:
        frame = Frame(code, initial_locals)
        self.frames.append(frame)
        current_line = 0
        current_col = 0

        try:
            while frame.pc < len(code.instructions):
                opcode, arg = code.instructions[frame.pc]
                frame.pc += 1

                lines = getattr(code, "instruction_lines", None)
                if lines is not None:
                    try:
                        current_line = code.instruction_lines[frame.pc - 1]
                        frame.current_line = current_line
                    except (IndexError, KeyError, TypeError):
                        pass

                cols = getattr(code, "instruction_columns", None)
                if cols is not None:
                    try:
                        current_col = code.instruction_columns[frame.pc - 1]
                        frame.current_col = current_col
                    except (IndexError, KeyError, TypeError):
                        pass

                if getattr(self, "step_hook", None) is not None:
                    self.step_hook(self, frame, opcode, arg, current_line, current_col)

                # ----------------------------------------------------
                # Constants & Names
                # ----------------------------------------------------
                if opcode == Opcode.LOAD_CONST:
                    val = code.constants[arg]
                    frame.stack.append(val)

                elif opcode == Opcode.LOAD_NAME:
                    name = code.names[arg]
                    if name in frame.locals:
                        frame.stack.append(frame.locals[name])
                    elif name in self.globals:
                        frame.stack.append(self.globals[name])
                    else:
                        raise VMRuntimeError(f"Name '{name}' is not defined", line=current_line, column=current_col)

                elif opcode == Opcode.STORE_NAME:
                    name = code.names[arg]
                    val = frame.stack.pop()
                    frame.locals[name] = val
                    # Eğer en üst frame ise global de güncellensin
                    if len(self.frames) == 1:
                        self.globals[name] = val

                elif opcode == Opcode.POP_TOP:
                    if frame.stack:
                        frame.stack.pop()

                # ----------------------------------------------------
                # Arithmetic & Tensor Ops
                # ----------------------------------------------------
                elif opcode == Opcode.BINARY_ADD:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    frame.stack.append(a + b)

                elif opcode == Opcode.BINARY_SUB:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    frame.stack.append(a - b)

                elif opcode == Opcode.BINARY_MUL:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    frame.stack.append(a * b)

                elif opcode == Opcode.BINARY_DIV:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    frame.stack.append(a / b)

                elif opcode == Opcode.BINARY_FLOOR_DIV:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    frame.stack.append(a // b)

                elif opcode == Opcode.BINARY_MOD:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    frame.stack.append(a % b)

                elif opcode == Opcode.BINARY_POW:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    frame.stack.append(a ** b)

                elif opcode == Opcode.BINARY_MATMUL:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    frame.stack.append(a @ b)

                elif opcode == Opcode.UNARY_NEGATIVE:
                    a = frame.stack.pop()
                    frame.stack.append(-a)

                elif opcode == Opcode.UNARY_NOT:
                    a = frame.stack.pop()
                    frame.stack.append(not a)

                # ----------------------------------------------------
                # Comparisons
                # ----------------------------------------------------
                elif opcode == Opcode.COMPARE_OP:
                    b = frame.stack.pop()
                    a = frame.stack.pop()
                    if arg == "==":
                        frame.stack.append(a == b)
                    elif arg == "!=":
                        frame.stack.append(a != b)
                    elif arg == "<":
                        frame.stack.append(a < b)
                    elif arg == "<=":
                        frame.stack.append(a <= b)
                    elif arg == ">":
                        frame.stack.append(a > b)
                    elif arg == ">=":
                        frame.stack.append(a >= b)
                    elif arg == "and":
                        frame.stack.append(a and b)
                    elif arg == "or":
                        frame.stack.append(a or b)
                    else:
                        raise VMRuntimeError(f"Unknown comparison operator: {arg}", line=current_line, column=current_col)

                # ----------------------------------------------------
                # Jumps & Control Flow
                # ----------------------------------------------------
                elif opcode == Opcode.JUMP:
                    frame.pc = arg

                elif opcode == Opcode.JUMP_IF_FALSE:
                    val = frame.stack.pop()
                    if not val:
                        frame.pc = arg

                elif opcode == Opcode.JUMP_IF_TRUE:
                    val = frame.stack.pop()
                    if val:
                        frame.pc = arg

                # ----------------------------------------------------
                # Functions & Calls
                # ----------------------------------------------------
                elif opcode == Opcode.MAKE_FUNCTION:
                    fn_code = frame.stack.pop()
                    syn_fn = SynapseFunction(fn_code, dict(frame.locals))
                    frame.stack.append(syn_fn)

                elif opcode == Opcode.CALL_FUNCTION:
                    has_kwargs = (arg >= 1000)
                    arg_count = (arg - 1000) if has_kwargs else arg

                    kwargs = frame.stack.pop() if has_kwargs else {}
                    args = []
                    for _ in range(arg_count):
                        args.append(frame.stack.pop())
                    args.reverse()

                    callee = frame.stack.pop()

                    if isinstance(callee, SynapseFunction):
                        result = callee(self, *args, **kwargs)
                    elif isinstance(callee, PromptObject):
                        result = callee(*args, **kwargs)
                    elif callable(callee):
                        result = callee(*args, **kwargs)
                    else:
                        raise VMRuntimeError(f"'{callee}' is not callable", line=current_line, column=current_col)

                    frame.stack.append(result)

                elif opcode == Opcode.RETURN_VALUE:
                    val = frame.stack.pop() if frame.stack else None
                    self.frames.pop()
                    return val

                # ----------------------------------------------------
                # Data Structures & Member Access
                # ----------------------------------------------------
                elif opcode == Opcode.BUILD_LIST:
                    elements = []
                    for _ in range(arg):
                        elements.append(frame.stack.pop())
                    elements.reverse()
                    frame.stack.append(elements)

                elif opcode == Opcode.BUILD_DICT:
                    d = {}
                    for _ in range(arg):
                        v = frame.stack.pop()
                        k = frame.stack.pop()
                        d[k] = v
                    frame.stack.append(d)

                elif opcode == Opcode.BINARY_SUBSCR:
                    sub = frame.stack.pop()
                    obj = frame.stack.pop()
                    frame.stack.append(obj[sub])

                elif opcode == Opcode.STORE_SUBSCR:
                    val = frame.stack.pop()
                    sub = frame.stack.pop()
                    obj = frame.stack.pop()
                    obj[sub] = val

                elif opcode == Opcode.LOAD_ATTR:
                    attr_name = code.names[arg]
                    obj = frame.stack.pop()
                    if hasattr(obj, attr_name):
                        frame.stack.append(getattr(obj, attr_name))
                    elif isinstance(obj, dict) and attr_name in obj:
                        frame.stack.append(obj[attr_name])
                    else:
                        raise VMRuntimeError(f"'{type(obj)}' object has no attribute '{attr_name}'", line=current_line, column=current_col)

                elif opcode == Opcode.STORE_ATTR:
                    attr_name = code.names[arg]
                    val = frame.stack.pop()
                    obj = frame.stack.pop()
                    if isinstance(obj, dict):
                        obj[attr_name] = val
                    else:
                        setattr(obj, attr_name, val)

                # ----------------------------------------------------
                # AI-Native & Python Interop
                # ----------------------------------------------------
                elif opcode == Opcode.BUILD_TENSOR:
                    kwargs = frame.stack.pop()
                    data = frame.stack.pop()
                    req_grad = kwargs.get("requires_grad", False)
                    t = tensor(data, requires_grad=req_grad)
                    frame.stack.append(t)

                elif opcode == Opcode.DEFINE_PROMPT:
                    name, params, fields = code.constants[arg]
                    prompt_obj = PromptObject(name, params, fields, self)
                    frame.stack.append(prompt_obj)

                elif opcode == Opcode.DEFINE_AGENT:
                    name, fields = code.constants[arg]
                    model_val = fields.get("model")
                    model_str = model_val.value if hasattr(model_val, "value") else "default"
                    inst_val = fields.get("instructions")
                    inst_str = inst_val.value if hasattr(inst_val, "value") else ""

                    resolved_tools = []
                    tools_val = fields.get("tools")
                    if tools_val is not None:
                        tool_items = []
                        if hasattr(tools_val, "elements"):
                            tool_items = tools_val.elements
                        elif isinstance(tools_val, list):
                            tool_items = tools_val
                        else:
                            tool_items = [tools_val]

                        for item in tool_items:
                            tool_name = getattr(item, "name", getattr(item, "value", str(item)))
                            if tool_name in frame.locals:
                                resolved_tools.append(frame.locals[tool_name])
                            elif tool_name in self.globals:
                                resolved_tools.append(self.globals[tool_name])

                    agent_obj = AgentRuntime(
                        name=name,
                        model=model_str,
                        tools=resolved_tools,
                        instructions=inst_str,
                        vm=self,
                    )
                    frame.stack.append(agent_obj)

                elif opcode == Opcode.IMPORT_PYTHON:
                    mod_str, alias, is_python = code.constants[arg]
                    try:
                        from synapse.interop.eco_bridge import TransparentPyResolver
                        imported_module = TransparentPyResolver.resolve_import(mod_str)
                        var_name = alias or (mod_str.split(".")[-1] if not is_python else mod_str)
                        frame.locals[var_name] = imported_module
                        self.globals[var_name] = imported_module
                        if "." in mod_str and not alias:
                            leaf_name = mod_str.split(".")[-1]
                            frame.locals[leaf_name] = imported_module
                            self.globals[leaf_name] = imported_module
                        if is_python:
                            root_pkg = mod_str.split(".")[0]
                            for ns_key in ("py", "python"):
                                py_ns = self.globals.get(ns_key)
                                if py_ns is None or not hasattr(py_ns, "__dict__"):
                                    class _PyNamespace:
                                        pass
                                    py_ns = _PyNamespace()
                                    self.globals[ns_key] = py_ns
                                    frame.locals[ns_key] = py_ns
                                setattr(py_ns, root_pkg, imported_module)
                    except Exception as e:
                        raise VMRuntimeError(f"Failed to import Python module '{mod_str}': {e}", line=current_line, column=current_col) from e

                else:
                    raise VMRuntimeError(f"Unhandled opcode: {opcode}", line=current_line, column=current_col)

        except VMRuntimeError as e:
            if e.line == 0 and current_line:
                raise VMRuntimeError(e.message, line=current_line, column=e.column or current_col) from e
            raise
        except Exception as e:
            raise VMRuntimeError(str(e), line=current_line, column=current_col) from e
        finally:
            if self.frames and self.frames[-1] is frame:
                self.frames.pop()

        return None
