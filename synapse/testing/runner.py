import os
import sys
import time
import re
import io
import contextlib
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple

from synapse.lexer.lexer import Lexer, LexerError
from synapse.parser.parser import Parser, ParseError
from synapse.parser.ast_nodes import FunctionDef
from synapse.vm.compiler import Compiler, CodeObject
from synapse.vm.virtual_machine import VirtualMachine, VMRuntimeError, SynapseFunction


# =============================================================================
# ANSI Colors & Sleek Minimalist Terminal Utilities
# =============================================================================

def _enable_ansi():
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    if sys.platform == "win32":
        try:
            os.system("")
        except Exception:
            pass

_enable_ansi()

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[90m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
WHITE = "\033[97m"
BOLD_GREEN = "\033[1;92m"
BOLD_RED = "\033[1;91m"
BOLD_WHITE = "\033[1;97m"

ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    return ANSI_REGEX.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", "") or "utf-8"
    try:
        "┌─┬─┐═✓✗●".encode(enc)
        return True
    except Exception:
        return False


def get_box_chars():
    if _supports_unicode():
        return {
            "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
            "h": "─", "v": "│",
            "tj": "┬", "bj": "┴", "lj": "├", "rj": "┤", "mj": "┼",
            "d_h": "═", "check": "✓", "cross": "✗", "bullet": "●"
        }
    else:
        return {
            "tl": "+", "tr": "+", "bl": "+", "br": "+",
            "h": "-", "v": "|",
            "tj": "+", "bj": "+", "lj": "+", "rj": "+", "mj": "+",
            "d_h": "=", "check": "[PASS]", "cross": "[FAIL]", "bullet": "*"
        }


def pad_cell(text: str, width: int, align: str = "left") -> str:
    vis = visible_len(text)
    pad = max(0, width - vis)
    if align == "right":
        return " " * pad + text
    return text + " " * pad


# =============================================================================
# Test Result & Data Structures
# =============================================================================

@dataclass
class TestResult:
    __test__ = False
    file_path: str
    test_name: str
    passed: bool
    duration_ms: float
    error_message: Optional[str] = None
    error_line: Optional[int] = None
    error_column: Optional[int] = None
    error_pointer: Optional[str] = None
    stdout_output: Optional[str] = None

    @property
    def status(self) -> str:
        return "PASSED" if self.passed else "FAILED"


# =============================================================================
# Test Runner Implementation
# =============================================================================

class TestRunner:
    __test__ = False

    def __init__(self, verbose: bool = False, color: bool = True):
        self.verbose = verbose
        self.color = color

    def discover_tests(self, target_path: str) -> List[str]:
        """Verilen dizin veya dosyada test_*.syn ve *_test.syn dosyalarını özyinelemeli olarak bulur."""
        target = os.path.abspath(target_path)
        if os.path.isfile(target):
            if target.endswith(".syn"):
                return [target]
            return []

        if not os.path.exists(target):
            return []

        discovered: List[str] = []
        for root, dirs, files in os.walk(target):
            # Gizli dizinleri ve önbellek klasörlerini atla
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                if f.endswith(".syn"):
                    name = f[:-4]
                    if f.startswith("test_") or name.endswith("_test"):
                        discovered.append(os.path.join(root, f))

        discovered.sort()
        return discovered

    def _create_isolated_vm(self) -> VirtualMachine:
        """Her test için izole, bağımsız bir VirtualMachine üretir ve assertion araçlarını ekler."""
        vm = VirtualMachine()

        def _assert(cond: Any, msg: str = "Assertion failed"):
            if not cond:
                raise AssertionError(str(msg))

        def _assert_eq(actual: Any, expected: Any, msg: Optional[str] = None):
            if actual != expected:
                raise AssertionError(msg or f"Expected {expected!r}, got {actual!r}")

        def _assert_ne(actual: Any, expected: Any, msg: Optional[str] = None):
            if actual == expected:
                raise AssertionError(msg or f"Expected values to differ, both are {actual!r}")

        def _assert_true(val: Any, msg: str = "Expected true"):
            if not bool(val):
                raise AssertionError(str(msg))

        def _assert_false(val: Any, msg: str = "Expected false"):
            if bool(val):
                raise AssertionError(str(msg))

        def _assert_gt(a: Any, b: Any, msg: Optional[str] = None):
            if not (a > b):
                raise AssertionError(msg or f"Expected {a!r} > {b!r}")

        def _assert_ge(a: Any, b: Any, msg: Optional[str] = None):
            if not (a >= b):
                raise AssertionError(msg or f"Expected {a!r} >= {b!r}")

        def _assert_lt(a: Any, b: Any, msg: Optional[str] = None):
            if not (a < b):
                raise AssertionError(msg or f"Expected {a!r} < {b!r}")

        def _assert_le(a: Any, b: Any, msg: Optional[str] = None):
            if not (a <= b):
                raise AssertionError(msg or f"Expected {a!r} <= {b!r}")

        def _assert_in(item: Any, container: Any, msg: Optional[str] = None):
            if item not in container:
                raise AssertionError(msg or f"Expected {item!r} in {container!r}")

        def _assert_not_in(item: Any, container: Any, msg: Optional[str] = None):
            if item in container:
                raise AssertionError(msg or f"Expected {item!r} not in {container!r}")

        def _assert_almost_eq(a: Any, b: Any, delta: float = 1e-5, msg: Optional[str] = None):
            diff = abs(a - b)
            if diff > delta:
                raise AssertionError(msg or f"Expected {a!r} ≈ {b!r} (diff {diff} > delta {delta})")

        def _fail(msg: str = "Test failed"):
            raise AssertionError(str(msg))

        vm.globals["assert"] = _assert
        vm.globals["assert_eq"] = _assert_eq
        vm.globals["assert_ne"] = _assert_ne
        vm.globals["assert_true"] = _assert_true
        vm.globals["assert_false"] = _assert_false
        vm.globals["assert_gt"] = _assert_gt
        vm.globals["assert_ge"] = _assert_ge
        vm.globals["assert_lt"] = _assert_lt
        vm.globals["assert_le"] = _assert_le
        vm.globals["assert_in"] = _assert_in
        vm.globals["assert_not_in"] = _assert_not_in
        vm.globals["assert_almost_eq"] = _assert_almost_eq
        vm.globals["fail"] = _fail

        return vm

    def _format_error(
        self, exc: Exception, file_path: str, source_lines: List[str]
    ) -> Tuple[str, Optional[int], Optional[int], Optional[str]]:
        err_line = None
        err_col = None

        if isinstance(exc, VMRuntimeError):
            err_line = exc.line if exc.line > 0 else None
            err_col = exc.column if exc.column > 0 else None
            if exc.__cause__:
                if isinstance(exc.__cause__, AssertionError):
                    cause_str = str(exc.__cause__)
                    err_msg = f"AssertionError: {cause_str}" if cause_str else "AssertionError"
                else:
                    err_msg = f"{type(exc.__cause__).__name__}: {exc.__cause__}"
            else:
                err_msg = exc.message
        elif isinstance(exc, (LexerError, ParseError)):
            err_line = getattr(exc, "line", None)
            err_col = getattr(exc, "column", None)
            err_msg = str(exc)
        elif isinstance(exc, AssertionError):
            cause_str = str(exc)
            err_msg = f"AssertionError: {cause_str}" if cause_str else "AssertionError"
        else:
            err_msg = f"{type(exc).__name__}: {exc}"

        # Görsel hata işaretçisi (Pointer Snippet)
        pointer = None
        if err_line and 1 <= err_line <= len(source_lines):
            line_content = source_lines[err_line - 1]
            gutter = f"    {err_line:4d} | "
            caret_line = " " * len(gutter)
            if err_col and err_col > 0:
                caret_line += " " * max(0, err_col - 1)
            else:
                indent = len(line_content) - len(line_content.lstrip())
                caret_line += " " * indent
            caret_line += f"^ {err_msg}"
            pointer = f"{gutter}{line_content}\n{caret_line}"

        return err_msg, err_line, err_col, pointer

    def run_file(self, file_path: str) -> List[TestResult]:
        """Tek bir .syn dosyasını çalıştırır."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            return [TestResult(
                file_path=file_path,
                test_name="(file)",
                passed=False,
                duration_ms=0.0,
                error_message=f"Could not read file: {e}"
            )]

        source_lines = source.splitlines()

        # Kaynağı tokenleştir, parse et ve derle
        compile_start = time.perf_counter()
        try:
            tokens = Lexer(source).tokenize()
            ast = Parser(tokens).parse()
            code = Compiler(name=os.path.basename(file_path)).compile(ast)
        except Exception as e:
            duration_ms = (time.perf_counter() - compile_start) * 1000.0
            err_msg, err_line, err_col, pointer = self._format_error(e, file_path, source_lines)
            return [TestResult(
                file_path=file_path,
                test_name="(compile)",
                passed=False,
                duration_ms=duration_ms,
                error_message=err_msg,
                error_line=err_line,
                error_column=err_col,
                error_pointer=pointer
            )]

        # Dosyadaki test fonksiyonlarını keşfet: fn test_* veya fn *_test
        test_fn_names = [
            stmt.name for stmt in ast.statements
            if isinstance(stmt, FunctionDef) and (stmt.name.startswith("test_") or stmt.name.endswith("_test"))
        ]

        # Senaryo 1: fn test_* fonksiyonları var -> Her birini izole VM'de çalıştır
        if test_fn_names:
            # Önce modül düzeyindeki tanımları doğrulamak için pre-check yap
            init_vm = self._create_isolated_vm()
            try:
                init_vm.execute(code)
            except Exception as e:
                err_msg, err_line, err_col, pointer = self._format_error(e, file_path, source_lines)
                return [TestResult(
                    file_path=file_path,
                    test_name="(module_setup)",
                    passed=False,
                    duration_ms=0.0,
                    error_message=err_msg,
                    error_line=err_line,
                    error_column=err_col,
                    error_pointer=pointer
                )]

            results: List[TestResult] = []
            for fn_name in test_fn_names:
                vm = self._create_isolated_vm()
                captured_stdout = io.StringIO()
                start_t = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(captured_stdout):
                        vm.execute(code)
                        if fn_name not in vm.globals:
                            raise VMRuntimeError(f"Test function '{fn_name}' not found in globals")
                        test_fn = vm.globals[fn_name]
                        if isinstance(test_fn, SynapseFunction):
                            params = getattr(test_fn.code, "params", [])
                            if params:
                                raise VMRuntimeError(f"Test function '{fn_name}' cannot require arguments, got: {params}")
                            test_fn(vm)
                        elif callable(test_fn):
                            test_fn()
                        else:
                            raise VMRuntimeError(f"'{fn_name}' is not callable")

                    duration_ms = (time.perf_counter() - start_t) * 1000.0
                    results.append(TestResult(
                        file_path=file_path,
                        test_name=fn_name,
                        passed=True,
                        duration_ms=duration_ms,
                        stdout_output=captured_stdout.getvalue()
                    ))
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_t) * 1000.0
                    err_msg, err_line, err_col, pointer = self._format_error(e, file_path, source_lines)
                    results.append(TestResult(
                        file_path=file_path,
                        test_name=fn_name,
                        passed=False,
                        duration_ms=duration_ms,
                        error_message=err_msg,
                        error_line=err_line,
                        error_column=err_col,
                        error_pointer=pointer,
                        stdout_output=captured_stdout.getvalue()
                    ))
            return results

        # Senaryo 2: Test fonksiyonu yok -> Dosya düzeyindeki assertion ve kodları çalıştır
        else:
            vm = self._create_isolated_vm()
            captured_stdout = io.StringIO()
            start_t = time.perf_counter()
            try:
                with contextlib.redirect_stdout(captured_stdout):
                    vm.execute(code)
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                return [TestResult(
                    file_path=file_path,
                    test_name="(file)",
                    passed=True,
                    duration_ms=duration_ms,
                    stdout_output=captured_stdout.getvalue()
                )]
            except Exception as e:
                duration_ms = (time.perf_counter() - start_t) * 1000.0
                err_msg, err_line, err_col, pointer = self._format_error(e, file_path, source_lines)
                return [TestResult(
                    file_path=file_path,
                    test_name="(file)",
                    passed=False,
                    duration_ms=duration_ms,
                    error_message=err_msg,
                    error_line=err_line,
                    error_column=err_col,
                    error_pointer=pointer,
                    stdout_output=captured_stdout.getvalue()
                )]

    def run_all(self, target_path: str) -> List[TestResult]:
        """Hedef dizindeki tüm testleri keşfeder ve sırayla yürütür."""
        test_files = self.discover_tests(target_path)
        all_results: List[TestResult] = []
        for f in test_files:
            all_results.extend(self.run_file(f))
        return all_results

    def print_report(self, results: List[TestResult], total_time_ms: float, target_path: str):
        """Şık, modern terminal tablosu ve özet raporunu yazdırır."""
        box = get_box_chars()

        if not results:
            print()
            print(f"{DIM}" + box["d_h"] * 78 + f"{RESET}")
            print(f"  {BOLD_WHITE}Synapse Test Runner{RESET}")
            print(f"{DIM}" + box["h"] * 78 + f"{RESET}")
            print(f"  {YELLOW}No Synapse test files found matching 'test_*.syn' or '*_test.syn' in: {target_path}{RESET}")
            print(f"{DIM}" + box["d_h"] * 78 + f"{RESET}\n")
            return

        # Sütun genişliklerini hesapla
        cwd = os.getcwd()
        display_files = []
        for r in results:
            try:
                rel = os.path.relpath(r.file_path, cwd).replace("\\", "/")
            except Exception:
                rel = os.path.basename(r.file_path)
            display_files.append(rel)

        max_file_len = max(len(f) for f in display_files) if display_files else 20
        max_test_len = max(len(r.test_name) for r in results) if results else 20

        col_file = max(20, min(42, max(len("File"), max_file_len)))
        col_test = max(20, min(38, max(len("Test"), max_test_len)))
        col_status = 8
        col_time = 12

        # Tablo başlığı
        top_border = f"{DIM}{box['tl']}{box['h'] * (col_file + 2)}{box['tj']}{box['h'] * (col_test + 2)}{box['tj']}{box['h'] * (col_status + 2)}{box['tj']}{box['h'] * (col_time + 2)}{box['tr']}{RESET}"
        header_row = f"{DIM}{box['v']}{RESET} {BOLD_WHITE}{pad_cell('File', col_file)}{RESET} {DIM}{box['v']}{RESET} {BOLD_WHITE}{pad_cell('Test', col_test)}{RESET} {DIM}{box['v']}{RESET} {BOLD_WHITE}{pad_cell('Status', col_status)}{RESET} {DIM}{box['v']}{RESET} {BOLD_WHITE}{pad_cell('Time (ms)', col_time, align='right')}{RESET} {DIM}{box['v']}{RESET}"
        mid_border = f"{DIM}{box['lj']}{box['h'] * (col_file + 2)}{box['mj']}{box['h'] * (col_test + 2)}{box['mj']}{box['h'] * (col_status + 2)}{box['mj']}{box['h'] * (col_time + 2)}{box['rj']}{RESET}"
        bot_border = f"{DIM}{box['bl']}{box['h'] * (col_file + 2)}{box['bj']}{box['h'] * (col_test + 2)}{box['bj']}{box['h'] * (col_status + 2)}{box['bj']}{box['h'] * (col_time + 2)}{box['br']}{RESET}"

        print()
        print(top_border)
        print(header_row)
        print(mid_border)

        for rel_file, r in zip(display_files, results):
            f_text = rel_file if len(rel_file) <= col_file else "..." + rel_file[-(col_file - 3):]
            t_text = r.test_name if len(r.test_name) <= col_test else r.test_name[:col_test - 3] + "..."

            if r.passed:
                status_cell = f"{BOLD_GREEN}PASSED{RESET}"
            else:
                status_cell = f"{BOLD_RED}FAILED{RESET}"

            time_cell = f"{DIM}{r.duration_ms:8.2f} ms{RESET}"

            line = (
                f"{DIM}{box['v']}{RESET} "
                f"{pad_cell(f_text, col_file)} {DIM}{box['v']}{RESET} "
                f"{pad_cell(WHITE + t_text + RESET, col_test)} {DIM}{box['v']}{RESET} "
                f"{pad_cell(status_cell, col_status)} {DIM}{box['v']}{RESET} "
                f"{pad_cell(time_cell, col_time, align='right')} "
                f"{DIM}{box['v']}{RESET}"
            )
            print(line)

        print(bot_border)

        # Hatalar bölümü
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            print(f"\n{BOLD_RED}{box['h'] * 2} Failures ({len(failed_results)}) {box['h'] * 60}{RESET}")
            for idx, f_res in enumerate(failed_results, 1):
                try:
                    rel_f = os.path.relpath(f_res.file_path, cwd).replace("\\", "/")
                except Exception:
                    rel_f = os.path.basename(f_res.file_path)
                print(f"\n{BOLD_RED}[{idx}] {rel_f} > {f_res.test_name}{RESET}")
                if f_res.error_message:
                    print(f"    {BOLD}Error:{RESET} {f_res.error_message}")
                if f_res.error_line:
                    loc = f"{rel_f}:{f_res.error_line}"
                    if f_res.error_column:
                        loc += f":{f_res.error_column}"
                    print(f"    {DIM}Location:{RESET} {loc}")
                if f_res.error_pointer:
                    print(f"\n{f_res.error_pointer}")
                if f_res.stdout_output and f_res.stdout_output.strip():
                    print(f"    {DIM}Captured stdout:{RESET}")
                    for out_line in f_res.stdout_output.strip().splitlines():
                        print(f"      {out_line}")

        # Verbose modunda geçen testlerin de çıktılarını göster
        if self.verbose:
            passed_with_stdout = [r for r in results if r.passed and r.stdout_output and r.stdout_output.strip()]
            if passed_with_stdout:
                print(f"\n{DIM}{box['h'] * 2} Verbose Captured Output {box['h'] * 50}{RESET}")
                for p_res in passed_with_stdout:
                    print(f"{CYAN}{box['bullet']} {p_res.test_name} ({p_res.file_path}):{RESET}")
                    for out_line in p_res.stdout_output.strip().splitlines():
                        print(f"  {out_line}")

        # Özet tablosu
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        pass_rate = (passed_count / len(results)) * 100.0 if results else 0.0

        print()
        print(f"{DIM}" + box["d_h"] * 78 + f"{RESET}")
        print(f"  {BOLD_WHITE}Synapse Test Suite Summary{RESET}")
        print(f"{DIM}" + box["h"] * 78 + f"{RESET}")
        print(f"  Total Tests (Toplam Test):       {BOLD_WHITE}{len(results)}{RESET}")
        print(f"  Passed (Gecen):                  {BOLD_GREEN}{passed_count}{RESET}")
        print(f"  Failed (Kalan / Hatali):         {BOLD_RED if failed_count > 0 else DIM}{failed_count}{RESET}")
        rate_color = BOLD_GREEN if pass_rate == 100.0 else (BOLD_YELLOW if pass_rate >= 80.0 else BOLD_RED)
        print(f"  Success Rate (Basari Orani):     {rate_color}{pass_rate:.1f}%{RESET}")
        print(f"  Total Duration (Toplam Sure):    {CYAN}{total_time_ms:.2f} ms{RESET}")
        print(f"{DIM}" + box["d_h"] * 78 + f"{RESET}")

        if failed_count == 0:
            print(f"  {BOLD_GREEN}{box['check']} ALL {len(results)} TESTS PASSED (100%){RESET}\n")
        else:
            print(f"  {BOLD_RED}{box['cross']} {failed_count} OF {len(results)} TESTS FAILED ({pass_rate:.1f}% pass rate){RESET}\n")

    def run(self, target_path: str = "tests") -> int:
        """Tüm testleri çalıştırır, terminal raporunu döker ve 0 veya 1 exit code döner."""
        start_time = time.perf_counter()
        results = self.run_all(target_path)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.print_report(results, total_time_ms, target_path)

        failed_count = sum(1 for r in results if not r.passed)
        return 0 if failed_count == 0 else 1


def run_tests(path: str = "tests", verbose: bool = False) -> int:
    """CLI ve harici çağrılar için ana test koşucu giriş noktası."""
    runner = TestRunner(verbose=verbose)
    return runner.run(path)
