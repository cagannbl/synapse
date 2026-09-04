import os
import sys
import shutil
import subprocess
from typing import Optional, Tuple
from synapse.lexer.lexer import Lexer
from synapse.parser.parser import Parser
from synapse.codegen.c_emitter import CEmitter


class NativeCompiler:
    def __init__(self):
        self.runtime_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "runtime")
        )
        self.runtime_c = os.path.join(self.runtime_dir, "synapse_runtime.c")
        self.runtime_h = os.path.join(self.runtime_dir, "synapse_runtime.h")

        # Standalone binary veya harici dizin modunda gömülü çalışma zamanından aç
        if not (os.path.isfile(self.runtime_c) and os.path.isfile(self.runtime_h)):
            try:
                from synapse.codegen.embedded_runtime import ensure_runtime_extracted
                self.runtime_dir, self.runtime_c, self.runtime_h = ensure_runtime_extracted()
            except Exception:
                pass

    def find_c_compiler(self) -> Optional[Tuple[str, str]]:
        """
        Sistemdeki C derleyicisini tespit eder.
        Dönüş: (compiler_executable, compiler_type: 'gcc' | 'clang' | 'cl' | 'tcc' | 'zig')
        """
        # 0. CC çevre değişkeni
        env_cc = os.environ.get("CC")
        if env_cc:
            resolved = shutil.which(env_cc) or (env_cc if os.path.isfile(env_cc) else None)
            if resolved:
                lower = resolved.lower()
                if "clang" in lower:
                    return resolved, "clang"
                elif "gcc" in lower:
                    return resolved, "gcc"
                elif "cl" in lower:
                    return resolved, "cl"
                elif "tcc" in lower:
                    return resolved, "tcc"
                elif "zig" in lower:
                    return resolved, "zig"
                return resolved, "gcc"

        # 1. PATH üzerindeki derleyiciler
        compilers = [
            ("clang", "clang"),
            ("gcc", "gcc"),
            ("zig", "zig"),
            ("tcc", "tcc"),
            ("cl", "cl"),
        ]

        def _is_functional(p: str, t: str) -> bool:
            try:
                import subprocess
                if t == "zig":
                    res = subprocess.run([p, "cc", "--version"], capture_output=True, timeout=2)
                    return res.returncode == 0
                elif t == "cl":
                    subprocess.run([p], capture_output=True, timeout=2)
                    return True
                else:
                    res = subprocess.run([p, "--version"], capture_output=True, timeout=2)
                    return res.returncode == 0
            except Exception:
                return False

        for cmd, c_type in compilers:
            path = shutil.which(cmd)
            if path and _is_functional(path, c_type):
                return path, c_type

        # 2. Yaygın MinGW, LLVM veya Windows dizinleri
        common_paths = [
            (r"C:\Program Files\LLVM\bin\clang.exe", "clang"),
            (os.path.expandvars(r"%LOCALAPPDATA%\Programs\LLVM\bin\clang.exe"), "clang"),
            (r"C:\msys64\ucrt64\bin\gcc.exe", "gcc"),
            (r"C:\msys64\clang64\bin\clang.exe", "clang"),
            (r"C:\msys64\mingw64\bin\gcc.exe", "gcc"),
            (r"C:\MinGW\bin\gcc.exe", "gcc"),
            (r"C:\w64devkit\bin\gcc.exe", "gcc"),
            (r"C:\tcc\tcc.exe", "tcc"),
        ]

        import glob
        winget_zig = glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\zig.zig_*\*\zig.exe"))
        for wz in winget_zig:
            common_paths.append((wz, "zig"))
        common_paths.append((os.path.expandvars(r"%USERPROFILE%\scoop\apps\zig\current\zig.exe"), "zig"))

        for path, c_type in common_paths:
            if os.path.isfile(path) and _is_functional(path, c_type):
                return path, c_type

        return None

    def transpile(self, source_code: str, filename: str = "<source>", include_line_directives: bool = True) -> str:
        """Synapse kaynak kodunu saf C koduna transpile eder."""
        tokens = Lexer(source_code).tokenize()
        ast = Parser(tokens).parse()
        emitter = CEmitter(include_line_directives=include_line_directives)
        return emitter.emit(ast, filename=filename)

    def transpile_file(self, syn_path: str, c_output_path: str, include_line_directives: bool = True) -> str:
        """Bir .syn dosyasını .c dosyasına transpile eder."""
        with open(syn_path, "r", encoding="utf-8") as f:
            source = f.read()

        c_code = self.transpile(source, filename=syn_path, include_line_directives=include_line_directives)

        with open(c_output_path, "w", encoding="utf-8") as f:
            f.write(c_code)

        return c_code

    def transpile_pyext(self, source_code: str, module_name: str = "synapse_ext") -> str:
        """Synapse kaynak kodunu CPython C-Uzantısı (.c) koduna dönüştürür."""
        from synapse.codegen.pyext_emitter import PyExtEmitter
        tokens = Lexer(source_code).tokenize()
        ast = Parser(tokens).parse()
        emitter = PyExtEmitter(module_name=module_name)
        return emitter.emit(ast)

    def transpile_pyext_file(self, syn_path: str, c_output_path: str, module_name: Optional[str] = None) -> str:
        """Synapse dosyasını CPython C-Uzantısı (.c) dosyasına transpile eder."""
        with open(syn_path, "r", encoding="utf-8") as f:
            source = f.read()

        mod_name = module_name or os.path.splitext(os.path.basename(syn_path))[0]
        c_code = self.transpile_pyext(source, module_name=mod_name)

        with open(c_output_path, "w", encoding="utf-8") as f:
            f.write(c_code)

        return c_code

    def build_executable(self, c_source_path: str, output_exe_path: str) -> Tuple[bool, str]:
        """Üretilen C dosyasını C derleyicisi ile doğrudan bağımsız .exe'ye derler."""
        if not os.path.isfile(c_source_path):
            return False, f"C source file does not exist: {c_source_path}"

        if not os.path.isfile(self.runtime_c):
            return False, f"Synapse C runtime file does not exist: {self.runtime_c}"

        # Windows platformunda .exe uzantısını garantiye al
        if sys.platform == "win32" and not output_exe_path.lower().endswith(".exe"):
            output_exe_path += ".exe"

        # Çıktı dizinini oluştur
        out_dir = os.path.dirname(os.path.abspath(output_exe_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        compiler_info = self.find_c_compiler()
        if not compiler_info:
            return False, (
                "No native C compiler found (clang, gcc, cl, tcc, zig).\n"
                "Please install Clang or GCC (e.g. via 'winget install LLVM.LLVM' or 'winget install BrechtSanders.WinLibs.POSIX.UCRT.LLVM').\n"
                "The C source code has been emitted successfully and is ready to compile."
            )

        compiler_path, c_type = compiler_info

        # Performans optimizasyon bayrakları (-O3 -mavx2 veya MSVC /O2 /arch:AVX2)
        cmd_variants: list[list[str]] = []

        if c_type in ("gcc", "clang"):
            cmd_variants = [
                # 1. Öncelik: OpenMP HPC + Maksimum Optimizasyon + AVX2 + FMA
                [compiler_path, "-fopenmp", "-O3", "-mavx2", "-mfma", c_source_path, self.runtime_c, f"-I{self.runtime_dir}", "-lm", "-o", output_exe_path],
                # 2. Fallback: -O3 -mavx2
                [compiler_path, "-O3", "-mavx2", c_source_path, self.runtime_c, f"-I{self.runtime_dir}", "-lm", "-o", output_exe_path],
                # 3. Fallback: Standart -O3 optimizasyonu
                [compiler_path, "-O3", c_source_path, self.runtime_c, f"-I{self.runtime_dir}", "-lm", "-o", output_exe_path],
            ]
        elif c_type == "zig":
            cmd_variants = [
                [compiler_path, "cc", "-O3", "-mavx2", c_source_path, self.runtime_c, f"-I{self.runtime_dir}", "-lm", "-o", output_exe_path],
                [compiler_path, "cc", "-O3", c_source_path, self.runtime_c, f"-I{self.runtime_dir}", "-lm", "-o", output_exe_path],
            ]
        elif c_type == "cl":
            cmd_variants = [
                # 1. Öncelik: MSVC /openmp + /O2 + /arch:AVX2
                [compiler_path, "/openmp", "/O2", "/arch:AVX2", c_source_path, self.runtime_c, f"/I{self.runtime_dir}", f"/Fe:{output_exe_path}"],
                # 2. Fallback: MSVC /O2 + /arch:AVX2
                [compiler_path, "/O2", "/arch:AVX2", c_source_path, self.runtime_c, f"/I{self.runtime_dir}", f"/Fe:{output_exe_path}"],
                # 3. Fallback: MSVC /O2
                [compiler_path, "/O2", c_source_path, self.runtime_c, f"/I{self.runtime_dir}", f"/Fe:{output_exe_path}"],
            ]
        elif c_type == "tcc":
            cmd_variants = [
                [compiler_path, "-O2", c_source_path, self.runtime_c, f"-I{self.runtime_dir}", "-lm", "-o", output_exe_path],
            ]
        else:
            return False, f"Unsupported compiler type: {c_type}"

        last_error = ""
        for cmd in cmd_variants:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and os.path.isfile(output_exe_path):
                    return True, output_exe_path
                last_error = f"{res.stderr}\n{res.stdout}".strip()
            except Exception as e:
                last_error = str(e)

        return False, f"C Compilation failed with {c_type}:\n{last_error}"

    def compile_source_to_executable(self, source_code: str, output_exe_path: str, temp_c_path: Optional[str] = None) -> Tuple[bool, str]:
        """Synapse kaynak kodunu doğrudan derleyip çalıştırılabilir .exe haline getirir."""
        if temp_c_path is None:
            base, _ = os.path.splitext(output_exe_path)
            temp_c_path = f"{base}.c"

        c_code = self.transpile(source_code)
        with open(temp_c_path, "w", encoding="utf-8") as f:
            f.write(c_code)

        return self.build_executable(temp_c_path, output_exe_path)

    def compile_and_run(
        self,
        filepath: str,
        run_args: Optional[list] = None,
        no_cache: bool = False,
    ) -> Tuple[bool, int, str, str]:
        """
        Synapse dosyasını C99 AOT ile derleyip .syn_cache/bin/ altına koyar ve çalıştırır.
        Kaynak dosya değişmediyse ve binary güncelse derlemeyi atlar (0 ms cache hit).
        Dönüş: (success: bool, exit_code: int, stdout: str, stderr: str)
        """
        import hashlib

        if not os.path.isfile(filepath):
            return False, 1, "", f"File not found: {filepath}"

        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        file_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(filepath) or ".", ".syn_cache", "bin")
        os.makedirs(cache_dir, exist_ok=True)

        is_win = sys.platform.startswith("win")
        exe_name = f"{os.path.splitext(os.path.basename(filepath))[0]}_{file_hash}" + (".exe" if is_win else "")
        exe_path = os.path.join(cache_dir, exe_name)
        c_path = os.path.join(cache_dir, f"{exe_name}.c")

        if not no_cache and os.path.isfile(exe_path) and os.path.getmtime(exe_path) >= os.path.getmtime(filepath):
            pass
        else:
            success, err = self.compile_source_to_executable(source, exe_path, temp_c_path=c_path)
            if not success:
                return False, 1, "", err

        cmd = [os.path.abspath(exe_path)] + (run_args or [])
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            return True, res.returncode, res.stdout, res.stderr
        except Exception as e:
            return False, 1, "", str(e)


