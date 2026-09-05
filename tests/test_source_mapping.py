import os
import re
import tempfile
import pytest
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.codegen.c_emitter import CEmitter
from synapse.codegen.native_compiler import NativeCompiler


def transpile_to_c(source: str, filename: str = "<source>", include_line_directives: bool = True) -> str:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    emitter = CEmitter(include_line_directives=include_line_directives, current_filename=filename)
    return emitter.emit(ast, filename=filename)


def test_line_directives_generated_for_statements():
    """Temel statement'lar (değişken bildirimi, atama, fonksiyon çağrısı) için #line direktiflerinin üretildiğini doğrular."""
    source = """let a = 10
let b = 20
a += b
print(a)
"""
    c_code = transpile_to_c(source, filename="main.syn")

    assert '#line 1 "main.syn"' in c_code
    assert '#line 2 "main.syn"' in c_code
    assert '#line 3 "main.syn"' in c_code
    assert '#line 4 "main.syn"' in c_code


def test_windows_path_normalization():
    """Windows ters eğik çizgi dosya yollarının POSIX standardına ('/') normalize edildiğini ve GCC/Clang \\U kaçış hatasını önlediğini doğrular."""
    win_path = r"C:\Users\cagan\Desktop\SynapseApp\src\neural_net.syn"
    source = "let weight = 0.5\nprint(weight)\n"

    c_code = transpile_to_c(source, filename=win_path)

    expected_normalized = "C:/Users/cagan/Desktop/SynapseApp/src/neural_net.syn"
    assert f'#line 1 "{expected_normalized}"' in c_code
    assert f'#line 2 "{expected_normalized}"' in c_code

    # Asla derleyiciyi kıracak ters eğik çizgi (\\Users, \\s vb.) içermemeli
    assert r"\Users" not in c_code
    assert "\\\\" not in c_code or "\\n" in c_code  # Sadece C string kaçışları kalabilir


def test_windows_path_with_spaces_and_special_chars():
    """Boşluk ve özel karakterler içeren Windows yollarının güvenli normalize edildiğini doğrular."""
    win_path = r"C:\Program Files (x86)\My Synapse Project\module_1.syn"
    source = "let lr = 0.001\n"

    c_code = transpile_to_c(source, filename=win_path)

    expected_path = "C:/Program Files (x86)/My Synapse Project/module_1.syn"
    assert f'#line 1 "{expected_path}"' in c_code


def test_include_line_directives_disabled():
    """include_line_directives=False olduğunda hiçbir #line direktifinin basılmadığını doğrular."""
    source = """let a = 1
let b = 2
if a < b:
    print("less")
"""
    # 1. CEmitter doğrudan
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    emitter = CEmitter(include_line_directives=False, current_filename="test.syn")
    c_code = emitter.emit(ast)

    assert "#line" not in c_code

    # 2. NativeCompiler üzerinden
    nc = NativeCompiler()
    c_code_nc = nc.transpile(source, filename="test.syn", include_line_directives=False)
    assert "#line" not in c_code_nc


def test_no_redundant_line_directives_on_same_line():
    """Aynı satırdaki ardışık ifadelerde tekrar #line üretilmediğini ve dosya boyutunun gereksiz şişmediğini doğrular."""
    # Tek satırlık fonksiyon ve tek satırlık blok
    source = "fn identity(x: int) -> int: return x\n"
    c_code = transpile_to_c(source, filename="same_line.syn")

    # Satır 1 için yalnızca 1 adet #line 1 üretilmeli
    line_1_matches = re.findall(r'#line 1 "same_line\.syn"', c_code)
    assert len(line_1_matches) == 1


def test_function_definition_and_return_mapping():
    """Fonksiyon tanımları ve gövdelerindeki return/var satır eşlemelerinin doğru yapıldığını doğrular."""
    source = """fn compute(a: int, b: int) -> int:
    let temp = a * 2
    return temp + b

let res = compute(3, 4)
"""
    c_code = transpile_to_c(source, filename="func_test.syn")

    assert '#line 1 "func_test.syn"' in c_code
    assert '#line 2 "func_test.syn"' in c_code
    assert '#line 3 "func_test.syn"' in c_code
    assert '#line 5 "func_test.syn"' in c_code

    # Fonksiyon imzasından hemen önce yer aldığını doğrula
    lines = c_code.splitlines()
    fn_idx = next(i for i, l in enumerate(lines) if "int compute(int a, int b) {" in l)
    assert lines[fn_idx - 1].strip() == '#line 1 "func_test.syn"'


def test_loops_while_and_for_line_mapping():
    """While ve for döngülerinde satır direktiflerinin döngü başlangıcına yerleştiğini doğrular."""
    source = """let counter = 0
while counter < 5:
    counter += 1

for i in 5:
    print(i)
"""
    c_code = transpile_to_c(source, filename="loops.syn")

    assert '#line 1 "loops.syn"' in c_code
    assert '#line 2 "loops.syn"' in c_code
    assert '#line 3 "loops.syn"' in c_code
    assert '#line 5 "loops.syn"' in c_code
    assert '#line 6 "loops.syn"' in c_code


def test_if_elif_else_branch_line_mapping():
    """If, elif ve else dallarının doğru satır direktifleriyle etiketlendiğini doğrular."""
    source = """let score = 85
if score >= 90:
    print("A")
elif score >= 80:
    print("B")
else:
    print("C")
"""
    c_code = transpile_to_c(source, filename="branches.syn")

    assert '#line 1 "branches.syn"' in c_code
    assert '#line 2 "branches.syn"' in c_code
    assert '#line 3 "branches.syn"' in c_code
    assert '#line 4 "branches.syn"' in c_code
    assert '#line 5 "branches.syn"' in c_code
    assert '#line 7 "branches.syn"' in c_code


def test_custom_filename_in_constructor_and_emit_override():
    """current_filename'in hem yapıcıda hem de emit çağrısında esnekçe belirtilebildiğini doğrular."""
    source = "let x = 42\n"
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()

    # 1. Yapıcıda belirtildiğinde
    emitter1 = CEmitter(current_filename=r"C:\init\file.syn")
    c1 = emitter1.emit(ast)
    assert '#line 1 "C:/init/file.syn"' in c1

    # 2. Emit çağrısında ezildiğinde
    emitter2 = CEmitter(current_filename=r"C:\init\file.syn")
    c2 = emitter2.emit(ast, filename=r"D:\override\new_file.syn")
    assert '#line 1 "D:/override/new_file.syn"' in c2
    assert "init" not in c2


def test_line_directive_c_preprocessor_syntax_integrity():
    """Üretilen C kodundaki tüm #line direktiflerinin geçerli C99 preprocessor formatında olduğunu doğrular."""
    source = """fn square(x: int) -> int:
    return x * x

let total = 0
for i in 10:
    total += square(i)

if total > 100:
    print("Large")
"""
    c_code = transpile_to_c(source, filename="integrity.syn")

    # Tüm #line satırlarını yakala
    line_directives = [l.strip() for l in c_code.splitlines() if l.strip().startswith("#line")]
    assert len(line_directives) > 0

    pattern = re.compile(r'^#line \d+ "[^"\n]+"$')
    for d in line_directives:
        assert pattern.match(d), f"Geçersiz #line direktifi sentaksı: {d}"


def test_end_to_end_native_c_compiler_build():
    """Üretilen #line direktifli C kodunun gerçek bir C derleyicisi (Clang/GCC/Zig/MSVC) ile hatasız derlendiğini doğrular."""
    nc = NativeCompiler()
    compiler_info = nc.find_c_compiler()
    if not compiler_info:
        pytest.skip("Sistemde kurulu C derleyicisi bulunamadı, derleme adımı atlanıyor.")

    source = """fn multiply(a: int, b: int) -> int:
    return a * b

let x = 7
let y = 6
let res = multiply(x, y)
print(res)
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        syn_path = os.path.join(tmp_dir, "calc_app.syn")
        c_path = os.path.join(tmp_dir, "calc_app.c")
        exe_path = os.path.join(tmp_dir, "calc_app.exe")

        with open(syn_path, "w", encoding="utf-8") as f:
            f.write(source)

        # Transpile et (Windows dosya yolu ile)
        c_code = nc.transpile_file(syn_path, c_path, include_line_directives=True)

        # Dosya içinde normalize edilmiş yol ile #line direktiflerinin varlığını doğrula
        normalized_syn_path = syn_path.replace("\\", "/")
        assert f'#line 1 "{normalized_syn_path}"' in c_code

        # Gerçek C derleyicisi ile derle
        ok, res_msg = nc.build_executable(c_path, exe_path)
        assert ok is True, f"C derlemesi başarısız oldu:\n{res_msg}"
        assert os.path.isfile(exe_path), f"Çıktı binary dosyası bulunamadı: {exe_path}"
