"""
Synapse Dual-Engine Differential Conformance Test Suite.

Validates that Synapse VirtualMachine (bytecode interpreter) and C Native Compiler
(CEmitter + NativeCompiler AOT) achieve 100% semantic equivalence and numerical conformance.

Covers:
1. Arithmetic operators & precedence
2. Variable assignments & mutations
3. if/elif/else conditions
4. while loop
5. for loop
6. Function calls & recursion
7. Tensor creation & indexing
8. Tensor addition & multiplication
9. Scalar matrix operations
10. Complex mathematical formulas
11. break & continue control flow
12. Multi-function call chain
"""

import pytest
from synapse.testing.conformance import EngineConformanceHarness, ConformanceReport


@pytest.fixture
def harness() -> EngineConformanceHarness:
    """Provides a fresh conformance testing harness instance."""
    return EngineConformanceHarness()


def test_arithmetic_operators_and_precedence(harness: EngineConformanceHarness):
    """1. Test arithmetic operators, nested parentheses, and operator precedence."""
    source = """
let a = 10 + 5 * 2
let b = (10 + 5) * 2
let c = 100 - 20 * 3 + 10
let d = (50 - 10) / 2
let e = 17 % 5
print(a)
print(b)
print(c)
print(d)
print(e)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert report.vm_res["variables"]["a"] == 20
    assert report.vm_res["variables"]["b"] == 30
    assert report.vm_res["variables"]["c"] == 50
    assert report.vm_res["variables"]["e"] == 2


def test_variable_assignments_and_mutations(harness: EngineConformanceHarness):
    """2. Test variable declarations, reassignments, and compound mutations (+=, -=, *=)."""
    source = """
let count = 10
count += 15
count *= 2
count -= 10
count = count + 5
print(count)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert report.vm_res["variables"]["count"] == 45


def test_if_elif_else_control_flow(harness: EngineConformanceHarness):
    """3. Test multiple branches of if / elif / else conditionals."""
    source = """
fn evaluate_grade(score: int) -> int:
    if score >= 90:
        return 1
    elif score >= 75:
        return 2
    elif score >= 60:
        return 3
    else:
        return 4

let g1 = evaluate_grade(95)
let g2 = evaluate_grade(80)
let g3 = evaluate_grade(65)
let g4 = evaluate_grade(40)

print(g1)
print(g2)
print(g3)
print(g4)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert report.vm_res["variables"]["g1"] == 1
    assert report.vm_res["variables"]["g2"] == 2
    assert report.vm_res["variables"]["g3"] == 3
    assert report.vm_res["variables"]["g4"] == 4


def test_while_loop(harness: EngineConformanceHarness):
    """4. Test while loop execution, accumulators, and termination condition."""
    source = """
let i = 1
let total = 0
let product = 1
while i <= 5:
    total += i
    product *= i
    i += 1

print(total)
print(product)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert report.vm_res["variables"]["total"] == 15
    assert report.vm_res["variables"]["product"] == 120


def test_for_loop(harness: EngineConformanceHarness):
    """5. Test for loop with range(n) and range(start, stop, step)."""
    source = """
let s1 = 0
for i in range(6):
    s1 += i

let s2 = 0
for j in range(2, 10, 2):
    s2 += j

print(s1)
print(s2)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert report.vm_res["variables"]["s1"] == 15
    assert report.vm_res["variables"]["s2"] == 20


def test_function_calls_and_recursion(harness: EngineConformanceHarness):
    """6. Test function definitions, argument passing, and recursive calls."""
    source = """
fn factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

fn fib(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)

let fact6 = factorial(6)
let fib7 = fib(7)

print(fact6)
print(fib7)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert report.vm_res["variables"]["fact6"] == 720
    assert report.vm_res["variables"]["fib7"] == 13


def test_tensor_creation_and_indexing(harness: EngineConformanceHarness):
    """7. Test tensor creation from array literals and array indexing."""
    source = """
let v = tensor([10.0, 25.0, 50.0, 75.0])
let v0 = v[0]
let v2 = v[2]
print(v0)
print(v2)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert abs(report.vm_res["variables"]["v0"] - 10.0) < 1e-5
    assert abs(report.vm_res["variables"]["v2"] - 50.0) < 1e-5


def test_tensor_addition_and_multiplication(harness: EngineConformanceHarness):
    """8. Test element-wise tensor addition, multiplication, and sum reductions."""
    source = """
let t1 = tensor([1.0, 2.0, 3.0])
let t2 = tensor([4.0, 5.0, 6.0])
let t_add = t1 + t2
let t_mul = t1 * t2
let s_add = t_add.sum().item()
let s_mul = t_mul.sum().item()
print(s_add)
print(s_mul)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert abs(report.vm_res["variables"]["s_add"] - 21.0) < 1e-5
    assert abs(report.vm_res["variables"]["s_mul"] - 32.0) < 1e-5


def test_scalar_matrix_operations(harness: EngineConformanceHarness):
    """9. Test 2D matrix scaling, scalar addition, and reduction."""
    source = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let scaled = A * 2.5
let shifted = scaled + 1.0
let sum_scaled = scaled.sum().item()
let sum_shifted = shifted.sum().item()
print(sum_scaled)
print(sum_shifted)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert abs(report.vm_res["variables"]["sum_scaled"] - 25.0) < 1e-5
    assert abs(report.vm_res["variables"]["sum_shifted"] - 29.0) < 1e-5


def test_complex_math_formulas(harness: EngineConformanceHarness):
    """10. Test nested polynomial and algebraic equations."""
    source = """
let x = 3.0
let y = 4.0
let poly = (x * x + y * y) / 5.0
let formula = ((x + y) * (y - x) + 15.0) / 2.0
print(poly)
print(formula)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert abs(report.vm_res["variables"]["poly"] - 5.0) < 1e-5
    assert abs(report.vm_res["variables"]["formula"] - 11.0) < 1e-5


def test_break_and_continue(harness: EngineConformanceHarness):
    """11. Test loop flow control via break and continue statements."""
    source = """
let sum = 0
for i in range(10):
    if i == 3:
        continue
    if i == 7:
        break
    sum += i

let w_sum = 0
let k = 0
while k < 10:
    k += 1
    if k == 2:
        continue
    if k == 6:
        break
    w_sum += k

print(sum)
print(w_sum)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert report.vm_res["variables"]["sum"] == 18
    assert report.vm_res["variables"]["w_sum"] == 13


def test_multi_function_call_chain(harness: EngineConformanceHarness):
    """12. Test function pipelines and sequential invocation chains."""
    source = """
fn add_ten(x: int) -> int:
    return x + 10

fn double_it(x: int) -> int:
    return x * 2

fn square_it(x: int) -> int:
    return x * x

fn pipeline_call(v: int) -> int:
    let a = add_ten(v)
    let b = double_it(a)
    let c = square_it(b)
    return c

let res = pipeline_call(2)
print(res)
"""
    report = harness.assert_conformance(source)
    assert report.is_match
    assert report.vm_res["variables"]["res"] == 576


def test_harness_diff_generation_on_mismatch(harness: EngineConformanceHarness):
    """Validates that compare() correctly produces unified diff on output discrepancies."""
    res1 = {"engine": "vm", "success": True, "stdout": "Output A\n42\n", "tensors": {}}
    res2 = {"engine": "c_native", "success": True, "stdout": "Output B\n42\n", "tensors": {}}

    report = harness.compare(res1, res2)
    assert not report.is_match
    assert "Output A" in report.diff
    assert "Output B" in report.diff


def test_harness_fuzzy_numeric_tolerance(harness: EngineConformanceHarness):
    """Validates that numerical tolerance allows small formatting variations."""
    res1 = {"engine": "vm", "success": True, "stdout": "Loss: 0.1234567\n", "tensors": {}}
    res2 = {"engine": "c_native", "success": True, "stdout": "Loss: 0.1234570\n", "tensors": {}}

    report = harness.compare(res1, res2, tolerance=1e-4)
    assert report.is_match


def test_harness_skip_mode_handling():
    """Validates that skip mode marks reports as skipped when compiler is absent."""
    h = EngineConformanceHarness(fallback_mode="skip")
    res_vm = h.run_on_vm("let a = 1\nprint(a)")
    res_c = h.run_on_c_native("let a = 1\nprint(a)")

    if not h.has_c_compiler():
        assert res_c["skipped"] is True
        report = h.compare(res_vm, res_c)
        assert report.skipped is True
