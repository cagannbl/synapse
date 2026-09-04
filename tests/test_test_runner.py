import os
import pytest
from synapse.testing.runner import TestResult, TestRunner, run_tests


def test_test_result_properties():
    res_pass = TestResult(file_path="test.syn", test_name="test_1", passed=True, duration_ms=1.5)
    assert res_pass.status == "PASSED"

    res_fail = TestResult(
        file_path="test.syn",
        test_name="test_2",
        passed=False,
        duration_ms=2.3,
        error_message="expected 1, got 2",
        error_line=10,
        error_column=5
    )
    assert res_fail.status == "FAILED"
    assert res_fail.error_line == 10
    assert res_fail.error_column == 5


def test_runner_discovery(tmp_path):
    # test_*.syn
    (tmp_path / "test_alpha.syn").write_text("fn test_a():\n    assert(true)\n", encoding="utf-8")
    # *_test.syn
    (tmp_path / "beta_test.syn").write_text("fn test_b():\n    assert(true)\n", encoding="utf-8")
    # Subdirectory
    sub = tmp_path / "subfolder"
    sub.mkdir()
    (sub / "test_gamma.syn").write_text("fn test_c():\n    assert(true)\n", encoding="utf-8")
    # Non-test .syn file
    (tmp_path / "helper.syn").write_text("let x = 10\n", encoding="utf-8")

    runner = TestRunner()
    found = runner.discover_tests(str(tmp_path))

    basenames = [os.path.basename(f) for f in found]
    assert "test_alpha.syn" in basenames
    assert "beta_test.syn" in basenames
    assert "test_gamma.syn" in basenames
    assert "helper.syn" not in basenames


def test_runner_isolated_execution(tmp_path):
    # Verify tests run in isolated VMs: mutations in test_one do not affect test_two
    syn_file = str(tmp_path / "test_state.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("""
let counter = 0

fn test_step_one():
    counter += 10
    assert_eq(counter, 10)

fn test_step_two():
    counter += 5
    assert_eq(counter, 5)
""")

    runner = TestRunner()
    results = runner.run_file(syn_file)
    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[0].duration_ms >= 0


def test_runner_rich_assertions(tmp_path):
    syn_file = str(tmp_path / "test_asserts.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("""
fn test_all_assertions():
    assert(1 == 1, "assert failed")
    assert_eq(42, 42)
    assert_ne(1, 2)
    assert_true(true)
    assert_false(false)
    assert_gt(10, 5)
    assert_ge(10, 10)
    assert_lt(5, 10)
    assert_le(5, 5)
    assert_in("b", ["a", "b", "c"])
    assert_not_in("z", ["a", "b", "c"])
    assert_almost_eq(3.14159, 3.14158, delta=0.001)
""")

    runner = TestRunner()
    results = runner.run_file(syn_file)
    assert len(results) == 1
    assert results[0].passed is True


def test_runner_failure_details(tmp_path):
    syn_file = str(tmp_path / "test_failing.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("""
fn test_ok():
    assert(true)

fn test_boom():
    let expected = 100
    let got = 50
    assert_eq(got, expected)
""")

    runner = TestRunner()
    results = runner.run_file(syn_file)
    assert len(results) == 2
    assert results[0].passed is True
    assert results[1].passed is False
    assert "Expected 100, got 50" in results[1].error_message
    assert results[1].error_line == 8
    assert results[1].error_pointer is not None
    assert "^" in results[1].error_pointer


def test_runner_script_mode(tmp_path):
    syn_file = str(tmp_path / "test_script.syn")
    with open(syn_file, "w", encoding="utf-8") as f:
        f.write("""
let a = 5
let b = 10
assert_eq(a + b, 15)
""")

    runner = TestRunner()
    results = runner.run_file(syn_file)
    assert len(results) == 1
    assert results[0].test_name == "(file)"
    assert results[0].passed is True


def test_runner_run_tests_exit_code(tmp_path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    (good_dir / "test_good.syn").write_text("fn test_pass():\n    assert(true)\n", encoding="utf-8")

    assert run_tests(str(good_dir)) == 0

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "test_bad.syn").write_text("fn test_fail():\n    fail(\"forced fail\")\n", encoding="utf-8")

    assert run_tests(str(bad_dir)) == 1

