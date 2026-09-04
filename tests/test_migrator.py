"""
Tests for PythonToSynapseMigrator (Faz 4: Python-to-Synapse Akıllı Kod Dönüştürücü).

Covers:
- Function signature conversions (def -> fn, types, defaults).
- Variable assignment conversions (let, const).
- Tensor creation and operations (np.array, torch.tensor, zeros, ones, matmul).
- Control flow structures (for, while, if/elif/else, return, break, continue).
- Complex expressions and binary/unary ops.
- Stripping redundant framework imports and mapping standard libraries.
- File I/O migration (migrate_file).
- Full end-to-end AI training pipeline migration.
- Syntax verification using Synapse's Lexer and Parser.
"""

import os
import tempfile
import pytest

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.parser.ast_nodes import (
    Program,
    FunctionDef,
    VarDeclStmt,
    AssignStmt,
    IfStmt,
    WhileStmt,
    ForStmt,
    TensorLiteralExpr,
    CallExpr,
)
from synapse.tools.migrator import PythonToSynapseMigrator, migrate_code, migrate_file


def parse_synapse(source: str) -> Program:
    """Helper to verify and parse converted Synapse code without syntax errors."""
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


# =============================================================================
# 1. Function Signatures & Declarations
# =============================================================================

def test_function_signature_simple():
    py_code = """
def multiply(x, y):
    return x * y
"""
    syn_code = migrate_code(py_code)
    assert "fn multiply(x, y):" in syn_code
    assert "return x * y" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 1
    fn = ast_prog.statements[0]
    assert isinstance(fn, FunctionDef)
    assert fn.name == "multiply"
    assert len(fn.params) == 2


def test_function_signature_typed_and_defaults():
    py_code = """
def train_model(lr: float = 0.001, epochs: int = 100) -> float:
    return lr * epochs
"""
    syn_code = migrate_code(py_code)
    assert "fn train_model(lr: float = 0.001, epochs: int = 100) -> float:" in syn_code
    assert "return lr * epochs" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 1
    fn = ast_prog.statements[0]
    assert isinstance(fn, FunctionDef)
    assert fn.name == "train_model"
    assert len(fn.params) == 2
    assert fn.params[0].name == "lr"
    assert fn.params[1].name == "epochs"


def test_function_with_tensor_annotations():
    py_code = """
def predict(batch: np.ndarray, weight: torch.Tensor) -> torch.Tensor:
    return batch @ weight
"""
    syn_code = migrate_code(py_code)
    assert "fn predict(batch: Tensor, weight: Tensor) -> Tensor:" in syn_code
    assert "return batch @ weight" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 1
    fn = ast_prog.statements[0]
    assert isinstance(fn, FunctionDef)
    assert fn.name == "predict"


# =============================================================================
# 2. Variable Assignments & Constants
# =============================================================================

def test_variable_assignment_untyped():
    py_code = """
x = 42
name = 'Synapse'
rate = 0.05
"""
    syn_code = migrate_code(py_code)
    assert "let x = 42" in syn_code
    assert "let name = 'Synapse'" in syn_code
    assert "let rate = 0.05" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 3
    for s in ast_prog.statements:
        assert isinstance(s, VarDeclStmt)
        assert s.is_const is False


def test_variable_assignment_typed():
    py_code = """
x: int = 10
rate: float = 0.001
flag: bool = True
"""
    syn_code = migrate_code(py_code)
    assert "let x: int = 10" in syn_code
    assert "let rate: float = 0.001" in syn_code
    assert "let flag: bool = true" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 3
    for s in ast_prog.statements:
        assert isinstance(s, VarDeclStmt)
        assert s.is_const is False


def test_uppercase_constants():
    py_code = """
MAX_BUFFER_SIZE = 1024
PI: float = 3.14159
LEARNING_RATE = 0.01
"""
    syn_code = migrate_code(py_code)
    assert "const MAX_BUFFER_SIZE = 1024" in syn_code
    assert "const PI: float = 3.14159" in syn_code
    assert "const LEARNING_RATE = 0.01" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 3
    for s in ast_prog.statements:
        assert isinstance(s, VarDeclStmt)
        assert s.is_const is True


# =============================================================================
# 3. Tensor Operations (NumPy & PyTorch)
# =============================================================================

def test_numpy_array_conversion():
    py_code = """
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = numpy.array([1, 2, 3])
c = np.asarray([4.0, 5.0])
"""
    syn_code = migrate_code(py_code)
    assert "import numpy" not in syn_code
    assert "let a = tensor([[1.0, 2.0], [3.0, 4.0]])" in syn_code
    assert "let b = tensor([1, 2, 3])" in syn_code
    assert "let c = tensor([4.0, 5.0])" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 3
    for s in ast_prog.statements:
        assert isinstance(s, VarDeclStmt)
        assert isinstance(s.value, TensorLiteralExpr)


def test_torch_tensor_conversion():
    py_code = """
import torch
w = torch.tensor([0.5, 1.5], requires_grad=True)
b = torch.Tensor([2.0])
z = torch.zeros([3, 3])
o = torch.ones([2, 4])
"""
    syn_code = migrate_code(py_code)
    assert "import torch" not in syn_code
    assert "let w = tensor([0.5, 1.5], requires_grad=true)" in syn_code
    assert "let b = tensor([2.0])" in syn_code
    assert "let z = zeros([3, 3])" in syn_code
    assert "let o = ones([2, 4])" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 4
    assert isinstance(ast_prog.statements[0].value, TensorLiteralExpr)


def test_matrix_multiplication_conversion():
    py_code = """
import numpy as np
import torch
c = np.dot(a, b)
d = torch.matmul(x, w)
e = x @ w
"""
    syn_code = migrate_code(py_code)
    assert "let c = (a @ b)" in syn_code
    assert "let d = (x @ w)" in syn_code
    assert "let e = x @ w" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 3


# =============================================================================
# 4. Import Cleanup and Interop Mapping
# =============================================================================

def test_strip_unnecessary_imports_and_map_stdlib():
    py_code = """
import numpy as np
import torch
from typing import List, Dict, Optional
import math
import time as tm
"""
    syn_code = migrate_code(py_code)
    assert "numpy" not in syn_code
    assert "torch" not in syn_code
    assert "typing" not in syn_code
    assert "import py.math as math" in syn_code
    assert "import py.time as tm" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 2


# =============================================================================
# 5. Control Flow (Loops, If/Elif/Else, Break/Continue)
# =============================================================================

def test_for_loop_conversion():
    py_code = """
total = 0
for i in range(10):
    total += i
"""
    syn_code = migrate_code(py_code)
    assert "let total = 0" in syn_code
    assert "for i in range(10):" in syn_code
    assert "total += i" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 2
    assert isinstance(ast_prog.statements[1], ForStmt)


def test_while_loop_and_break_continue():
    py_code = """
count = 10
while count > 0:
    if count == 5:
        break
    elif count == 8:
        count -= 1
        continue
    count -= 1
"""
    syn_code = migrate_code(py_code)
    assert "let count = 10" in syn_code
    assert "while count > 0:" in syn_code
    assert "if count == 5:" in syn_code
    assert "break" in syn_code
    assert "elif count == 8:" in syn_code
    assert "continue" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 2
    assert isinstance(ast_prog.statements[1], WhileStmt)


def test_if_elif_else_conversion():
    py_code = """
score = 85
if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:
    grade = 'C'
else:
    grade = 'F'
"""
    syn_code = migrate_code(py_code)
    assert "if score >= 90:" in syn_code
    assert "elif score >= 80:" in syn_code
    assert "elif score >= 70:" in syn_code
    assert "else:" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 2
    assert isinstance(ast_prog.statements[1], IfStmt)
    assert len(ast_prog.statements[1].elif_branches) == 2
    assert ast_prog.statements[1].else_branch is not None


# =============================================================================
# 6. Expressions, Binary Ops, and Precedence
# =============================================================================

def test_complex_expressions_and_binary_ops():
    py_code = """
a = 5
b = 10
c = 2
d = 4
e = 1
result = -(a + b) * c / (d - e) ** 2
flag = not (a > 0 and b < 20 or c == 0)
"""
    syn_code = migrate_code(py_code)
    assert "-(a + b) * c / (d - e) ** 2" in syn_code
    assert "not ((a > 0 and b < 20) or c == 0)" in syn_code or "not (a > 0 and b < 20 or c == 0)" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 7


def test_augmented_assignment_in_place():
    py_code = """
x = 10
x += 5
x *= 2
x -= 1
x /= 3
"""
    syn_code = migrate_code(py_code)
    assert "let x = 10" in syn_code
    assert "x += 5" in syn_code
    assert "x *= 2" in syn_code
    assert "x -= 1" in syn_code
    assert "x /= 3" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 5
    assert isinstance(ast_prog.statements[0], VarDeclStmt)
    for s in ast_prog.statements[1:]:
        assert isinstance(s, AssignStmt)


# =============================================================================
# 7. Neural Network & PyTorch Functional Mapping
# =============================================================================

def test_nn_layers_and_activations():
    py_code = """
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

fc = nn.Linear(64, 10)
relu_layer = nn.ReLU()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(lr=0.001)
out = F.relu(x)
"""
    syn_code = migrate_code(py_code)
    assert "let fc = Linear(64, 10)" in syn_code
    assert "let relu_layer = ReLU()" in syn_code
    assert "let loss_fn = MSELoss()" in syn_code
    assert "let optimizer = Adam(lr=0.001)" in syn_code
    assert "let out = relu(x)" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 5


# =============================================================================
# 8. File I/O Migration
# =============================================================================

def test_migrate_file():
    py_content = """import numpy as np

def add_bias(t: np.ndarray, b: float = 1.0) -> np.ndarray:
    return t + b

MAX_STEPS = 50
w = np.array([0.1, 0.2])
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = os.path.join(tmp_dir, "model.py")
        out_path = os.path.join(tmp_dir, "model.syn")

        with open(in_path, "w", encoding="utf-8") as f:
            f.write(py_content)

        migrated_str = migrate_file(in_path, out_path)

        assert os.path.isfile(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            saved_str = f.read()

        assert migrated_str == saved_str
        assert "fn add_bias(t: Tensor, b: float = 1.0) -> Tensor:" in saved_str
        assert "const MAX_STEPS = 50" in saved_str
        assert "let w = tensor([0.1, 0.2])" in saved_str

        ast_prog = parse_synapse(saved_str)
        assert len(ast_prog.statements) == 3


# =============================================================================
# 9. End-to-End AI Model Training Loop Migration
# =============================================================================

def test_end_to_end_ai_training_script_migration():
    py_script = """import numpy as np
import torch

def compute_loss(w, x, y):
    y_pred = x @ w
    diff = y_pred - y
    return (diff * diff).sum()

LEARNING_RATE = 0.05
w = torch.tensor([[0.5], [1.5]], requires_grad=True)
x = np.array([[1.0, 2.0], [3.0, 4.0]])
y = np.array([[3.5], [9.5]])

loss = compute_loss(w, x, y)
loss.backward()
print("Gradient computed:", w.grad)
"""
    syn_script = migrate_code(py_script)

    assert "import numpy" not in syn_script
    assert "import torch" not in syn_script
    assert "fn compute_loss(w, x, y):" in syn_script
    assert "const LEARNING_RATE = 0.05" in syn_script
    assert "let w = tensor([[0.5], [1.5]], requires_grad=true)" in syn_script
    assert "let x = tensor([[1.0, 2.0], [3.0, 4.0]])" in syn_script
    assert "let y = tensor([[3.5], [9.5]])" in syn_script
    assert "loss.backward()" in syn_script

    # Validate complete syntax correctness with Lexer and Parser
    ast_prog = parse_synapse(syn_script)
    assert isinstance(ast_prog, Program)
    assert len(ast_prog.statements) == 8


# =============================================================================
# 10. Dotted Imports & Specific Symbol Import Mapping
# =============================================================================

def test_dotted_and_symbol_imports():
    py_code = """
import os.path
import os.path as osp
from os.path import exists, isfile
from . import local_module
"""
    syn_code = migrate_code(py_code)
    assert "import py.os.path as path" in syn_code
    assert "import py.os.path as osp" in syn_code
    assert "import py.os.path.exists as exists" in syn_code
    assert "import py.os.path.isfile as isfile" in syn_code
    assert "import local_module" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 5


# =============================================================================
# 11. For Loop Tuple Unpacking
# =============================================================================

def test_for_loop_tuple_unpacking():
    py_code = """
pairs = [[1, 2], [3, 4]]
total = 0
for x, y in pairs:
    total += x * y
"""
    syn_code = migrate_code(py_code)
    assert "for _item" in syn_code
    assert "let x = _item" in syn_code
    assert "let y = _item" in syn_code
    assert "total += x * y" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 3


# =============================================================================
# 12. Membership Testing ('in' and 'not in' desugaring to 'contains')
# =============================================================================

def test_membership_in_and_not_in():
    py_code = """
def check_item(collection, target):
    if target in collection:
        return True
    elif target not in collection:
        return False
    return None
"""
    syn_code = migrate_code(py_code)
    assert "if contains(collection, target):" in syn_code
    assert "elif not contains(collection, target):" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 1


# =============================================================================
# 13. Chained Comparisons Desugaring
# =============================================================================

def test_chained_comparisons_desugaring():
    py_code = """
x = 5
if 0 < x <= 10:
    res = 1
"""
    syn_code = migrate_code(py_code)
    assert "(0 < x) and (x <= 10)" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 2


# =============================================================================
# 14. List & Dict Comprehension Desugaring
# =============================================================================

def test_list_and_dict_comprehension_desugaring():
    py_code = """
items = [1, 2, 3, 4]
squares = [x * 2 for x in items]
evens = [x for x in items if x % 2 == 0]
mapping = {k: v for k, v in items}
"""
    syn_code = migrate_code(py_code)
    assert "let squares = []" in syn_code
    assert "squares.append(x * 2)" in syn_code
    assert "let evens = []" in syn_code
    assert "evens.append(x)" in syn_code
    assert "let mapping = {}" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert isinstance(ast_prog, Program)


# =============================================================================
# 15. Ternary IfExp Desugaring
# =============================================================================

def test_ternary_ifexp_desugaring():
    py_code = """
cond = True
val = 42 if cond else 0

def get_sign(x: float) -> float:
    return 1.0 if x > 0.0 else -1.0
"""
    syn_code = migrate_code(py_code)
    assert "let val = none" in syn_code
    assert "if cond:" in syn_code
    assert "val = 42" in syn_code
    assert "val = 0" in syn_code
    assert "if x > 0.0:" in syn_code
    assert "return 1.0" in syn_code
    assert "return -1.0" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 4


# =============================================================================
# 16. Lambda Desugaring
# =============================================================================

def test_lambda_desugaring():
    py_code = """
multiply = lambda a, b: a * b
add_one = lambda x: x + 1
"""
    syn_code = migrate_code(py_code)
    assert "fn multiply(a, b):" in syn_code
    assert "return a * b" in syn_code
    assert "fn add_one(x):" in syn_code
    assert "return x + 1" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 2


# =============================================================================
# 17. Context Manager & Try Preservation
# =============================================================================

def test_context_manager_and_try_preservation():
    py_code = """
with torch.no_grad():
    y = x @ w
    loss = y.sum()

try:
    res = 1 / 0
except Exception:
    pass
"""
    syn_code = migrate_code(py_code)
    assert "# with" in syn_code
    assert "let y = x @ w" in syn_code
    assert "let loss = y.sum()" in syn_code
    assert "# try:" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) >= 3


# =============================================================================
# 18. Class Definition & Methods Conversion
# =============================================================================

def test_class_definition_methods_conversion():
    py_code = """
class NeuralLinear:
    def __init__(self, in_features: int, out_features: int):
        self.w = zeros([in_features, out_features])
        self.b = zeros([out_features])

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b
"""
    syn_code = migrate_code(py_code)
    assert "# class NeuralLinear" in syn_code
    assert "fn NeuralLinear_init(self, in_features: int, out_features: int):" in syn_code
    assert "self.w = zeros([in_features, out_features])" in syn_code
    assert "fn NeuralLinear_forward(self, x: Tensor) -> Tensor:" in syn_code
    assert "return x @ self.w + self.b" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 2


# =============================================================================
# 19. Method-based Matrix Multiplication & Tensor Properties
# =============================================================================

def test_method_matmul_and_properties():
    py_code = """
c = a.dot(b)
d = x.matmul(w)
e = x.mm(w)
bs = x.size(0)
dims = x.dim()
xt = x.t()
avail = torch.cuda.is_available()
dev = torch.device('cuda')
"""
    syn_code = migrate_code(py_code)
    assert "let c = (a @ b)" in syn_code
    assert "let d = (x @ w)" in syn_code
    assert "let e = (x @ w)" in syn_code
    assert "let bs = x.shape[0]" in syn_code
    assert "let dims = x.ndim" in syn_code
    assert "let xt = x.T" in syn_code
    assert "let avail = is_cuda_available()" in syn_code
    assert "let dev = 'cuda'" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 8


# =============================================================================
# 20. NumPy/PyTorch Math and Helper Factory Extensions
# =============================================================================

def test_math_and_helper_factories():
    py_code = """
import numpy as np
import torch

z_like = np.zeros_like(x)
o_like = torch.ones_like(x)
arr = np.linspace(0.0, 1.0, 100)
clamped = torch.clamp(x, min=0.0, max=1.0)
y = np.exp(x) + np.log(x) + np.sqrt(x) + np.sin(x)
"""
    syn_code = migrate_code(py_code)
    assert "let z_like = zeros_like(x)" in syn_code
    assert "let o_like = ones_like(x)" in syn_code
    assert "let arr = linspace(0.0, 1.0, 100)" in syn_code
    assert "let clamped = clamp(x, min=0.0, max=1.0)" in syn_code
    assert "exp(x) + log(x) + sqrt(x) + sin(x)" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 5


# =============================================================================
# 21. Multiple Function Return Unpacking Without Side Effects
# =============================================================================

def test_tuple_unpacking_without_side_effects():
    py_code = """
a, b = fetch_coordinates()
"""
    syn_code = migrate_code(py_code)
    assert "let _unpack" in syn_code
    assert "let a = _unpack" in syn_code
    assert "let b = _unpack" in syn_code

    ast_prog = parse_synapse(syn_code)
    assert len(ast_prog.statements) == 3

