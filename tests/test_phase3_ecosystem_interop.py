import json
import os
import tempfile
import numpy as np
import pytest
from synapse.core.tensor import Tensor, tensor
from synapse.core.dataframe import DataFrame, dataframe
from synapse.interop.python_bridge import (
    PythonBridge, to_synapse, from_synapse, to_synapse_tensor, to_numpy, SynapseInteropError
)
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.codegen.pyext_emitter import PyExtEmitter
from synapse.codegen.native_compiler import NativeCompiler
from synapse.pkg.manager import PackageManager
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine


def run_synapse(code: str) -> VirtualMachine:
    tokens = Lexer(code).tokenize()
    ast = Parser(tokens).parse()
    compiled = Compiler().compile(ast)
    vm = VirtualMachine()
    vm.execute(compiled)
    return vm


# =========================================================================
# 1. Python FFI Bi-Directional Conversion & Callbacks
# =========================================================================
def test_interop_tensor_conversion():
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    syn_t = to_synapse(np_arr)
    assert isinstance(syn_t, Tensor)
    assert np.allclose(syn_t.data, np_arr)

    back_np = from_synapse(syn_t)
    assert isinstance(back_np, np.ndarray)
    assert np.allclose(back_np, np_arr)


def test_interop_dataframe_conversion():
    df = DataFrame({"col_a": [1, 2, 3], "col_b": [10.0, 20.0, 30.0]})
    py_dict = from_synapse(df)
    assert isinstance(py_dict, dict)
    assert py_dict["col_a"] == [1, 2, 3]


def test_interop_callback_wrapping():
    # Pass a Synapse callback into a Python standard function
    def python_higher_order_fn(func, values):
        return [func(v) for v in values]

    # Callback receiving Synapse Tensor and returning modified
    def synapse_callback(x):
        return x * 10

    wrapped_cb = from_synapse(synapse_callback)
    results = python_higher_order_fn(wrapped_cb, [1, 2, 3])
    assert results == [10, 20, 30]


def test_interop_submodule_and_attribute_access():
    path_bridge = PythonBridge.load("os.path")
    joined = path_bridge.join("home", "user", "project")
    assert "home" in joined and "project" in joined

    math_bridge = PythonBridge.load("math")
    assert math_bridge.sin(0.0) == 0.0
    assert abs(math_bridge.pi - 3.14159) < 1e-4


def test_interop_error_propagation():
    math_bridge = PythonBridge.load("math")
    with pytest.raises(SynapseInteropError) as exc_info:
        # math.sqrt of negative number raises ValueError in Python
        math_bridge.sqrt(-1.0)
    assert "math.sqrt" in str(exc_info.value)


# =========================================================================
# 2. Python C-Extension Code Generation (PyExtEmitter)
# =========================================================================
def test_pyext_emitter_code_generation():
    source = """
fn add_doubles(a: float, b: float) -> float:
    return a + b

fn multiply_ints(x: int, y: int) -> int:
    return x * y

fn is_positive(val: float) -> bool:
    return val > 0.0
"""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    emitter = PyExtEmitter(module_name="fast_synapse_math")
    c_code = emitter.emit(ast)

    # Verify standard CPython C-Extension elements
    assert "#define PY_SSIZE_T_CLEAN" in c_code
    assert "#include <Python.h>" in c_code
    assert 'PyInit_fast_synapse_math' in c_code
    assert 'static PyMethodDef fast_synapse_math_methods[]' in c_code
    assert 'static struct PyModuleDef fast_synapse_math_module' in c_code

    # Verify argument parsing and wrappers
    assert 'PyArg_ParseTuple(args, "dd"' in c_code
    assert 'PyArg_ParseTuple(args, "ii"' in c_code
    assert 'PyFloat_FromDouble' in c_code
    assert 'PyLong_FromLong' in c_code
    assert 'PyBool_FromLong' in c_code


def test_native_compiler_pyext_transpile():
    nc = NativeCompiler()
    source = """
fn scale(val: float, factor: float) -> float:
    return val * factor
"""
    c_code = nc.transpile_pyext(source, module_name="scaler_module")
    assert "PyInit_scaler_module" in c_code
    assert "scaler_module_methods" in c_code
    assert "PyArg_ParseTuple" in c_code


# =========================================================================
# 3. Deterministic Package Lockfile (synapse.lock)
# =========================================================================
def test_package_manager_generate_lockfile():
    with tempfile.TemporaryDirectory() as tmp_dir:
        pm = PackageManager()
        pm.init_project(tmp_dir, name="test_locked_app")

        # Generate lockfile
        lock_path = pm.generate_lockfile(tmp_dir)
        assert os.path.isfile(lock_path)
        assert os.path.basename(lock_path) == "synapse.lock"

        with open(lock_path, "r", encoding="utf-8") as f:
            lock_data = json.load(f)

        assert lock_data["version"] == 1
        assert "generated_by" in lock_data
        assert "packages" in lock_data


# =========================================================================
# 4. Script-Level Python Interop in Synapse VM
# =========================================================================
def test_synapse_script_interop():
    code = """
import py.math as m
import py.json as j

let angle = 0.0
let sin_val = m.sin(angle)
let serialized = j.dumps({"status": "synapse_interop_green"})
"""
    vm = run_synapse(code)
    assert vm.globals["sin_val"] == 0.0
    assert 'status' in vm.globals["serialized"]
    assert 'synapse_interop_green' in vm.globals["serialized"]
