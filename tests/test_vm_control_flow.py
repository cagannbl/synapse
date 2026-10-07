"""
Bytecode VM regression tests for control flow that used to be silently dropped
by the compiler: match/case, the `?` operator, and iterator-based for loops.
"""

import pytest

from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine, VMRuntimeError


def run(source: str) -> VirtualMachine:
    ast = Parser(Lexer(source).tokenize()).parse()
    vm = VirtualMachine()
    vm.execute(Compiler().compile(ast))
    return vm


# ---------------------------------------------------------------------------
# match / case
# ---------------------------------------------------------------------------

def test_match_literal_guard_capture_and_wildcard():
    vm = run(
        """
fn classify(x):
    match x:
        case 1:
            return "one"
        case "hi":
            return "greeting"
        case n if n > 10:
            return "big"
        case _:
            return "other"

let a = classify(1)
let b = classify(42)
let c = classify("hi")
let d = classify(5)
"""
    )
    assert [vm.globals[k] for k in "abcd"] == ["one", "big", "greeting", "other"]


def test_match_first_matching_case_wins_and_no_match_is_noop():
    vm = run(
        """
let hits = []
match 2:
    case 2:
        hits.append("first")
    case 2:
        hits.append("second")
match 99:
    case 1:
        hits.append("never")
"""
    )
    assert vm.globals["hits"] == ["first"]


def test_match_result_and_option_constructors():
    vm = run(
        """
fn describe(r):
    match r:
        case Ok(v):
            return v * 2
        case Err(e):
            return "err:" + e
        case Some(0):
            return "zero"
        case Some(v):
            return v
        case None:
            return "none"

let a = describe(Result.Ok(21))
let b = describe(Err("bad"))
let c = describe(Some(0))
let d = describe(Some(7))
let e = describe(None)
"""
    )
    assert [vm.globals[k] for k in "abcde"] == [42, "err:bad", "zero", 7, "none"]


def test_match_on_undefined_name_raises():
    with pytest.raises(VMRuntimeError, match="Name 'missing' is not defined"):
        run(
            """
match missing:
    case _:
        print("unreachable")
"""
        )


# ---------------------------------------------------------------------------
# ? operator
# ---------------------------------------------------------------------------

def test_try_operator_unwraps_ok_and_propagates_err():
    vm = run(
        """
fn half(n):
    if n % 2 == 0:
        return Ok(n / 2)
    return Err("odd")

fn quarter(n):
    let h = half(n)?
    let q = half(h)?
    return Ok(q)

let good = quarter(8)
let bad = quarter(6)
"""
    )
    assert vm.globals["good"].value == 2
    assert vm.globals["bad"].error == "odd"


def test_try_operator_propagates_none_option():
    vm = run(
        """
fn first_positive(xs):
    for x in xs:
        if x > 0:
            return Some(x)
    return None

fn doubled(xs):
    let v = first_positive(xs)?
    return Some(v * 2)

let a = doubled([-1, 3])
let b = doubled([-1, -2])
"""
    )
    assert vm.globals["a"].value == 6
    assert vm.globals["b"].is_none()


# ---------------------------------------------------------------------------
# for loops & dict literals
# ---------------------------------------------------------------------------

def test_for_iterates_ranges_dicts_strings_and_tensors():
    vm = run(
        """
let out = []
for i in range(3):
    out.append(i)
for k in {"a": 1, "b": 2}:
    out.append(k)
for ch in "xy":
    out.append(ch)
let rows = 0
for row in tensor([[1.0, 2.0], [3.0, 4.0]]):
    rows += 1
"""
    )
    assert vm.globals["out"] == [0, 1, 2, "a", "b", "x", "y"]
    assert vm.globals["rows"] == 2


def test_for_break_and_continue():
    vm = run(
        """
let out = []
for x in [1, 2, 3, 4, 5]:
    if x == 2:
        continue
    if x == 4:
        break
    out.append(x)
"""
    )
    assert vm.globals["out"] == [1, 3]


def test_for_drains_channel_until_closed():
    vm = run(
        """
let ch = channel(capacity=8)
fn producer():
    for i in range(5):
        ch.send(i)
    ch.close()
    return "done"

let fut = spawn(producer)
let received = []
for item in ch:
    received.append(item)
let status = fut.result()
"""
    )
    assert vm.globals["received"] == [0, 1, 2, 3, 4]
    assert vm.globals["status"] == "done"


def test_dict_literal_preserves_insertion_order():
    vm = run('let d = {"z": 1, "a": 2, "m": 3}\nlet keys = list(d.keys())')
    assert vm.globals["keys"] == ["z", "a", "m"]


def test_nested_loops_reusing_target_name_and_recursive_match():
    vm = run(
        """
let out = []
for i in [1, 2]:
    for i in [10, 20]:
        out.append(i)
    out.append(i)

fn depth(r):
    match r:
        case Some(v):
            return depth(v) + 1
        case _:
            return 0

let d = depth(Some(Some(Some(None))))
"""
    )
    assert vm.globals["out"] == [10, 20, 20, 10, 20, 20]
    assert vm.globals["d"] == 3


def test_prompt_fields_are_evaluated_as_expressions(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    vm = run(
        """
let suffix = "!"
prompt Greet(name: str):
    system: "Be brief."
    user: "Say hi to " + name + suffix

let reply = Greet("Ada")
"""
    )
    assert "Say hi to Ada!" in vm.globals["reply"]
    assert "BinaryExpr" not in vm.globals["reply"]
