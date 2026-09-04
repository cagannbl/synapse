import os
import pytest
from synapse.lexer.lexer import Lexer, TokenType
from synapse.parser.parser import Parser
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine, VMRuntimeError
from synapse.core.tensor import tensor, Tensor, TensorShapeMismatchError
from synapse.core.diagnostics import diagnose_code


def run_source(source: str, vm: VirtualMachine = None) -> VirtualMachine:
    """Helper to tokenize, parse, compile, and execute Synapse source code."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    if vm is None:
        vm = VirtualMachine()
    vm.execute(code)
    return vm


def test_multiline_triple_quoted_strings():
    """Verify that both \"\"\" and ''' multiline strings preserve newlines and content."""
    # 1. Lexer level verification for double quotes
    src_double = 'let s = """Line 1\nLine 2\nLine 3"""'
    tokens_double = Lexer(src_double).tokenize()
    str_token_double = next(t for t in tokens_double if t.type == TokenType.STRING)
    assert str_token_double.value == "Line 1\nLine 2\nLine 3"
    assert "\n" in str_token_double.value

    # 2. Lexer level verification for single quotes
    src_single = "let s = '''Alpha\nBeta\nGamma'''"
    tokens_single = Lexer(src_single).tokenize()
    str_token_single = next(t for t in tokens_single if t.type == TokenType.STRING)
    assert str_token_single.value == "Alpha\nBeta\nGamma"
    assert "\n" in str_token_single.value

    # 3. VM execution verification with embedded formatting and multiline blocks
    source_vm = (
        'let prompt_template = """You are a software architect.\n'
        'Instructions:\n'
        '1. Ensure type safety\n'
        '2. Optimize memory layouts"""\n'
        '\n'
        "let query_payload = '''SELECT id, name, model\n"
        'FROM neural_models\n'
        'WHERE status = "active"\'\'\'\n'
    )
    vm = run_source(source_vm)
    assert vm.globals["prompt_template"] == (
        "You are a software architect.\n"
        "Instructions:\n"
        "1. Ensure type safety\n"
        "2. Optimize memory layouts"
    )
    assert vm.globals["query_payload"] == (
        "SELECT id, name, model\n"
        "FROM neural_models\n"
        'WHERE status = "active"'
    )


def test_store_attr_object_and_dict():
    """Verify obj.x = 10 and data.key = 'value' assignments work via STORE_ATTR."""
    # 1. Store attribute on custom Python object
    class ModelArtifact:
        def __init__(self):
            self.x = 0
            self.title = "untitled"

    obj = ModelArtifact()
    vm = VirtualMachine()
    vm.globals["obj"] = obj

    source_obj = (
        "obj.x = 10\n"
        'obj.title = "ResNet-50"\n'
    )
    tokens = Lexer(source_obj).tokenize()
    ast = Parser(tokens).parse()
    code = Compiler().compile(ast)
    vm.execute(code)

    assert obj.x == 10
    assert obj.title == "ResNet-50"

    # 2. Store attribute on dictionaries (STORE_ATTR dictionary support)
    source_dict = (
        'let data = {"status": "pending"}\n'
        'data.key = "value"\n'
        'data.status = "completed"\n'
        "data.epochs = 25\n"
        "data.learning_rate = 0.001\n"
    )
    vm2 = run_source(source_dict)
    d = vm2.globals["data"]
    assert isinstance(d, dict)
    assert d["key"] == "value"
    assert d["status"] == "completed"
    assert d["epochs"] == 25
    assert d["learning_rate"] == 0.001


def test_dict_dot_notation():
    """Verify reading dictionary keys using data.key dot notation (LOAD_ATTR)."""
    source = (
        'let data = {\n'
        '    "model": "gpt-4o",\n'
        '    "temperature": 0.7,\n'
        '    "active": true,\n'
        '    "params": {"lr": 0.0005, "batch_size": 32}\n'
        '}\n'
        'let selected_model = data.model\n'
        'let current_temp = data.temperature\n'
        'let is_active = data.active\n'
        'let inner_lr = data.params.lr\n'
        'let inner_batch = data.params.batch_size\n'
    )
    vm = run_source(source)
    assert vm.globals["selected_model"] == "gpt-4o"
    assert vm.globals["current_temp"] == 0.7
    assert vm.globals["is_active"] is True
    assert vm.globals["inner_lr"] == 0.0005
    assert vm.globals["inner_batch"] == 32

    # Verify error on non-existent dict attribute
    source_err = (
        'let d = {"name": "test"}\n'
        'let val = d.non_existent_key\n'
    )
    with pytest.raises(VMRuntimeError) as exc_info:
        run_source(source_err)
    assert "non_existent_key" in str(exc_info.value)


def test_runtime_error_line_tracking():
    """Verify that when runtime errors occur, VMRuntimeError.line points to the exact line number."""
    # 1. Undefined variable at line 4
    source_undef = (
        "let a = 100\n"                    # Line 1
        "let b = 200\n"                    # Line 2
        "let c = a + b\n"                  # Line 3
        "let d = unknown_symbol_ref * 2\n" # Line 4
        "let e = 500\n"                    # Line 5
    )
    with pytest.raises(VMRuntimeError) as exc_undef:
        run_source(source_undef)
    assert exc_undef.value.line == 4
    assert "[Line 4]" in str(exc_undef.value)
    assert "unknown_symbol_ref" in str(exc_undef.value)

    # 2. Missing attribute access at line 3
    source_attr = (
        'let info = {"version": 1}\n' # Line 1
        "let multiplier = 5\n"         # Line 2
        "let missing = info.secret\n"  # Line 3
    )
    with pytest.raises(VMRuntimeError) as exc_attr:
        run_source(source_attr)
    assert exc_attr.value.line == 3
    assert "[Line 3]" in str(exc_attr.value)
    assert "secret" in str(exc_attr.value)


def test_tensor_shape_mismatch_error():
    """Verify TensorShapeMismatchError on invalid matrix multiplication and diagnostics output."""
    # 1. Direct Python level tensor mismatch
    t1 = tensor([[1, 2]])  # (1, 2)
    t2 = tensor([[1, 2]])  # (1, 2)
    with pytest.raises(TensorShapeMismatchError) as exc_info:
        _ = t1 @ t2
    err_str = str(exc_info.value)
    assert "Cannot multiply tensor of shape (1, 2) with tensor of shape (1, 2)" in err_str
    assert "Inner dimensions must match: 2 != 1" in err_str

    # 2. Diagnostic engine static verification
    source_code = (
        "let a = tensor([[1, 2]])\n"
        "let b = tensor([[1, 2]])\n"
        "let c = a @ b\n"
    )
    report = diagnose_code(source_code)
    assert report.status == "error"
    assert report.error_type == "TensorShapeMismatchError"
    assert report.source_line == "let c = a @ b"
    assert "^" in report.pointer
    assert report.ai_prompt_hint is not None
    assert "matrix multiplication (@)" in report.ai_prompt_hint
    assert ".T" in report.ai_prompt_hint

    # 3. Diagnostic engine runtime verification
    try:
        _ = t1 @ t2
    except TensorShapeMismatchError as err:
        runtime_rep = diagnose_code(source_code, error=err)
        assert runtime_rep.status == "error"
        assert runtime_rep.error_type == "TensorShapeMismatchError"
        assert runtime_rep.source_line == "let c = a @ b"
        assert "^" in runtime_rep.pointer
        assert runtime_rep.ai_prompt_hint is not None
        assert "matrix multiplication" in runtime_rep.ai_prompt_hint


def test_stdlib_file_io_and_json(tmp_path):
    """Verify write_file, file_exists, read_file, json_stringify, and json_parse stdlib functions."""
    target_file = str((tmp_path / "synapse_stage2_io.json").as_posix())

    source = f'''
let file_path = "{target_file}"
let before_exists = file_exists(file_path)

let original_data = {{
    "language": "Synapse",
    "stage": 2,
    "version": "2.0.0",
    "features": ["multiline_strings", "dot_notation", "diagnostics", "stdlib"],
    "metrics": {{"accuracy": 0.985, "loss": 0.015}}
}}

let serialized_str = json_stringify(original_data)
let bytes_written = write_file(file_path, serialized_str)
let after_write_exists = file_exists(file_path)

let read_content = read_file(file_path)
let restored_data = json_parse(read_content)

let lang_name = restored_data.language
let lang_stage = restored_data.stage
let loss_metric = restored_data.metrics.loss
'''
    vm = run_source(source)

    assert vm.globals["before_exists"] is False
    assert vm.globals["after_write_exists"] is True
    assert vm.globals["bytes_written"] > 0
    assert vm.globals["read_content"] == vm.globals["serialized_str"]

    restored = vm.globals["restored_data"]
    assert isinstance(restored, dict)
    assert restored["language"] == "Synapse"
    assert restored["stage"] == 2
    assert restored["metrics"]["loss"] == 0.015

    assert vm.globals["lang_name"] == "Synapse"
    assert vm.globals["lang_stage"] == 2
    assert vm.globals["loss_metric"] == 0.015

    # Clean up and verify file_exists returns False
    if os.path.exists(target_file):
        os.remove(target_file)

    cleanup_check_source = f'let clean_exists = file_exists("{target_file}")\n'
    vm_clean = run_source(cleanup_check_source)
    assert vm_clean.globals["clean_exists"] is False
