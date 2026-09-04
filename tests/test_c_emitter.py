import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.codegen.c_emitter import CEmitter


def transpile(source: str) -> str:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return CEmitter().emit(ast)


def test_emit_tensor_math_and_matmul():
    source = """
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[5.0, 6.0], [7.0, 8.0]])
let C = A @ B
let At = A.T
"""
    c_code = transpile(source)

    assert "syn_tensor_create" in c_code
    assert "syn_matmul(A, B)" in c_code
    assert "syn_transpose(A)" in c_code
    assert "int main(" in c_code


def test_emit_autograd_and_training_loop():
    source = """
let w = tensor(0.5, requires_grad=true)
let x = tensor(2.0)
let y_true = tensor(6.0)

let step = 1
while step <= 10:
    let y_pred = w * x
    let diff = y_pred - y_true
    let loss = diff * diff
    loss.backward()
    let grad_val = w.grad.item()
    step += 1
"""
    c_code = transpile(source)

    assert "syn_tensor_scalar(0.5" in c_code
    assert "while ((step <= 10))" in c_code
    assert "syn_backward(loss)" in c_code
    assert "syn_tensor_item((w->grad))" in c_code


def test_emit_pipeline_operator():
    source = """
let raw = tensor([1.0, -2.0, 3.0])
let out = raw |> relu
"""
    c_code = transpile(source)

    assert "syn_relu(raw)" in c_code


def test_emit_function_def_with_types():
    source = """
fn add(a: float, b: float) -> float:
    return a + b

fn scale(t: Tensor, factor: float) -> Tensor:
    return t * factor

fn do_nothing():
    pass
"""
    c_code = transpile(source)

    assert "double add(double a, double b);" in c_code
    assert "syn_tensor_t* scale(syn_tensor_t* t, double factor);" in c_code
    assert "void do_nothing(void);" in c_code
    assert "double add(double a, double b) {" in c_code
    assert "return (a + b);" in c_code
    assert "syn_tensor_t* scale(syn_tensor_t* t, double factor) {" in c_code
    assert "syn_mul_scalar(t, factor)" in c_code
    assert "void do_nothing(void) {" in c_code


def test_emit_if_elif_else_control_flow():
    source = """
let x = 10
if x > 15:
    print("Greater than 15")
elif x > 5:
    print("Greater than 5")
else:
    print("Small")
"""
    c_code = transpile(source)

    assert "if ((x > 15)) {" in c_code
    assert "} else if ((x > 5)) {" in c_code
    assert "} else {" in c_code
    assert 'printf("%s\\n", "Greater than 15");' in c_code


def test_emit_for_loops_range():
    source = """
for i in range(10):
    print(i)

for j in range(2, 8):
    print(j)

for k in range(10, 0, -1):
    print(k)

for n in 5:
    pass
"""
    c_code = transpile(source)

    assert "for (int i = 0; i < 10; i++) {" in c_code
    assert "for (int j = 2; j < 8; j++) {" in c_code
    assert "for (int k = 10; k > 0; k += -1) {" in c_code
    assert "for (int n = 0; n < 5; n++) {" in c_code


def test_emit_var_decl_types_and_assignments():
    source = """
let counter: int = 0
let lr: float = 0.001
const max_iter: int = 100

counter += 1
lr *= 0.5

let W = tensor([1.0, 2.0])
let dW = tensor([0.1, 0.2])
W += dW
W -= dW
"""
    c_code = transpile(source)

    assert "int counter = 0;" in c_code
    assert "double lr = 0.001;" in c_code
    assert "const int max_iter = 100;" in c_code
    assert "counter += 1;" in c_code
    assert "lr *= 0.5;" in c_code
    assert "W = syn_add(W, dW);" in c_code
    assert "W = syn_sub(W, dW);" in c_code


def test_emit_break_continue():
    source = """
let i = 0
while i < 10:
    if i == 5:
        break
    else:
        continue
    i += 1
"""
    c_code = transpile(source)

    assert "break;" in c_code
    assert "continue;" in c_code


def test_emit_print_formatting():
    source = """
let step = 1
let loss = 0.045
print("Step:", step, "Loss:", loss)

let T = tensor([1.0, 2.0])
print("Tensor T:", T)
"""
    c_code = transpile(source)

    assert 'printf("%s %d %s %.6f\\n", "Step:", step, "Loss:", (double)(loss));' in c_code
    assert 'syn_tensor_print("", T)' in c_code


def test_emit_line_directives_enabled():
    source = """
let a = 1
let b = 2
fn foo(x: int) -> int:
    return x * 2

if a > 0:
    print(b)
"""
    ast = Parser(Lexer(source).tokenize()).parse()
    emitter = CEmitter(include_line_directives=True)
    c_code = emitter.emit(ast, filename=r"synapse\test\module.syn")

    # Windows path '\\' must be converted to '/'
    assert '#line 2 "synapse/test/module.syn"' in c_code
    assert '#line 3 "synapse/test/module.syn"' in c_code
    assert '#line 4 "synapse/test/module.syn"' in c_code
    assert '#line 5 "synapse/test/module.syn"' in c_code
    assert '#line 7 "synapse/test/module.syn"' in c_code
    assert '#line 8 "synapse/test/module.syn"' in c_code


def test_emit_line_directives_disabled():
    source = """
let a = 1
let b = 2
fn foo(x: int) -> int:
    return x * 2
"""
    ast = Parser(Lexer(source).tokenize()).parse()
    emitter = CEmitter(include_line_directives=False)
    c_code = emitter.emit(ast, filename="test.syn")

    assert "#line" not in c_code


def test_emit_line_directives_duplicate_line_suppressed():
    from synapse.parser.ast_nodes import Program, VarDeclStmt, LiteralExpr
    # Create two statements with the same line number 42
    s1 = VarDeclStmt(name="x", value=LiteralExpr(value=10, line=42), line=42)
    s2 = VarDeclStmt(name="y", value=LiteralExpr(value=20, line=42), line=42)
    prog = Program(statements=[s1, s2])

    emitter = CEmitter(include_line_directives=True)
    c_code = emitter.emit(prog, filename="same_line.syn")

    # Only one #line 42 directive should be emitted
    assert c_code.count('#line 42 "same_line.syn"') == 1


def test_emit_loop_arena_and_scope_management():
    source = """
let n = 5
for i in range(n):
    let local_t = tensor([1.0, 2.0, 3.0])
    let acc = 0.0

let step = 0
while step < 10:
    let step_t = tensor([4.0, 5.0])
    step += 1
"""
    c_code = transpile(source)

    # Loop arena oluşturma ve free kontrolleri
    assert "syn_arena_t* _loop_arena = syn_arena_create(1024 * 1024);" in c_code
    assert "syn_arena_scope_t _loop_scope = syn_arena_scope_enter(_loop_arena);" in c_code
    assert "syn_arena_scope_leave(_loop_scope);" in c_code
    assert "if (_loop_arena) syn_arena_free(_loop_arena);" in c_code

    # Döngü içindeki yerel tensörlerin arena üzerinde tahsis edildiği
    assert "syn_tensor_create_arena(_loop_arena" in c_code

    # Birden fazla döngüde arena çakışmaması (benzersiz isimler)
    assert "_loop_arena_1" in c_code
    assert "_loop_scope_1" in c_code


def test_emit_monomorphic_tagged_unions_option_and_result():
    source = """
fn get_num(flag: int) -> Option[int]:
    if flag > 0:
        return Option.Some(42)
    return Option.None

fn process_tensor(flag: int) -> Option[Tensor]:
    if flag > 0:
        let t = tensor([1.0, 2.0])
        return Option.Some(t)
    return Option.None

fn compute_result(x: int) -> Result[int, str]:
    if x >= 0:
        return Result.Ok(x * 2)
    return Result.Err("Negative value")
"""
    c_code = transpile(source)

    # Tag enum tanımları
    assert "typedef enum { SYN_OPT_NONE = 0, SYN_OPT_SOME = 1 } SynOptTag;" in c_code
    assert "typedef enum { SYN_RES_ERR = 0, SYN_RES_OK = 1 } SynResTag;" in c_code

    # Monomorfik struct tanımları
    assert "typedef struct { SynOptTag tag; union { int val; } as; } SynOption_int;" in c_code
    assert "typedef struct { SynOptTag tag; union { syn_tensor_t* val; } as; } SynOption_tensor;" in c_code
    assert "typedef struct { SynResTag tag; union { int ok; const char* err; } as; } SynResult_int_str;" in c_code

    # Inline yapıcı fonksiyonlar
    assert "static inline SynOption_int syn_option_int_some(int v)" in c_code
    assert "static inline SynOption_int syn_option_int_none(void)" in c_code
    assert "static inline SynResult_int_str syn_result_int_str_ok(int v)" in c_code
    assert "static inline SynResult_int_str syn_result_int_str_err(const char* e)" in c_code

    # Yapıcı çağrı eşlemeleri
    assert "syn_option_int_some(42)" in c_code
    assert "syn_option_int_none()" in c_code
    assert "syn_result_int_str_ok((x * 2))" in c_code
    assert 'syn_result_int_str_err("Negative value")' in c_code

    # Kör syn_tensor_t* type-punning'i kaldırıldı
    assert "syn_tensor_t* get_num" not in c_code


def test_emit_pattern_matching_match_stmt():
    source = """
fn handle_opt(opt: Option[int]) -> int:
    match opt:
        case Option.Some(x):
            return x + 10
        case Option.None:
            return 0

fn handle_res(res: Result[int, str]) -> int:
    match res:
        case Result.Ok(val):
            return val * 2
        case Result.Err(err):
            return -1
"""
    c_code = transpile(source)

    # Subject geçici değişken ataması
    assert "SynOption_int _match_subj_1 = opt;" in c_code
    assert "SynResult_int_str _match_subj_2 = res;" in c_code

    # Switch ifadeleri
    assert "switch (_match_subj_1.tag) {" in c_code
    assert "case SYN_OPT_SOME: {" in c_code
    assert "int x = _match_subj_1.as.val;" in c_code
    assert "case SYN_OPT_NONE: {" in c_code

    assert "switch (_match_subj_2.tag) {" in c_code
    assert "case SYN_RES_OK: {" in c_code
    assert "int val = _match_subj_2.as.ok;" in c_code
    assert "case SYN_RES_ERR: {" in c_code
    assert "const char* err = _match_subj_2.as.err;" in c_code


def test_emit_try_expr_lowering():
    source = """
fn step1(x: int) -> Result[int, str]:
    if x > 0:
        return Result.Ok(x + 1)
    return Result.Err("invalid")

fn step2(x: int) -> Result[int, str]:
    let val = step1(x)?
    return Result.Ok(val * 10)
"""
    c_code = transpile(source)

    # Pure ISO C99 statement unnesting (MSVC compatible, no GNU ({ ... }) extensions)
    assert "({\n" not in c_code and "({ " not in c_code
    assert "_try_res_1" in c_code
    assert "if (_try_res_1.tag == SYN_RES_ERR)" in c_code
    assert "return _try_res_1;" in c_code
    assert "_try_res_1.as.ok" in c_code


def test_c99_full_program_compilation():
    import os
    import subprocess
    from synapse.codegen.native_compiler import NativeCompiler

    compiler_info = NativeCompiler().find_c_compiler()
    if not compiler_info:
        pytest.skip("No C compiler found on system")

    source = """
fn calc(val: int) -> Result[int, str]:
    if val > 0:
        return Result.Ok(val * 2)
    return Result.Err("Must be positive")

fn process(val: int) -> Result[int, str]:
    let num = calc(val)?
    match calc(num):
        case Result.Ok(v):
            return Result.Ok(v + 1)
        case Result.Err(e):
            return Result.Err(e)

let res = process(5)
let out = 0
match res:
    case Result.Ok(v):
        out = v
    case Result.Err(e):
        out = -1

for i in range(3):
    let tmp = tensor([1.0, 2.0])
    let x = i
"""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    c_code = CEmitter().emit(ast)

    # Runtime header dosyasının bulunduğu dizin
    runtime_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "synapse", "runtime")
    )
    runtime_c = os.path.join(runtime_dir, "synapse_runtime.c")

    test_c_file = os.path.abspath("test_generated_c99.c")
    test_exe_file = os.path.abspath("test_generated_c99.exe")

    with open(test_c_file, "w", encoding="utf-8") as f:
        f.write(c_code)

    compiler_path, c_type = compiler_info
    if c_type == "zig":
        cmd = [compiler_path, "cc", test_c_file, runtime_c, f"-I{runtime_dir}", "-lm", "-o", test_exe_file]
    elif c_type in ("gcc", "clang"):
        cmd = [compiler_path, "-O2", test_c_file, runtime_c, f"-I{runtime_dir}", "-lm", "-o", test_exe_file]
    elif c_type == "cl":
        cmd = [compiler_path, "/O2", test_c_file, runtime_c, f"/I{runtime_dir}", f"/Fe:{test_exe_file}"]
    else:
        cmd = [compiler_path, test_c_file, runtime_c, f"-I{runtime_dir}", "-o", test_exe_file]

    try:
        compile_res = subprocess.run(cmd, capture_output=True, text=True)
        assert compile_res.returncode == 0, f"Compilation failed: {compile_res.stderr}"

        # Programı çalıştır ve başarılı çıktığını doğrula
        run_res = subprocess.run([test_exe_file], capture_output=True, text=True)
        assert run_res.returncode == 0
    finally:
        if os.path.exists(test_c_file):
            os.remove(test_c_file)
        if os.path.exists(test_exe_file):
            os.remove(test_exe_file)


