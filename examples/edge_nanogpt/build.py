#!/usr/bin/env python3
"""
Synapse Edge NanoGPT Standalone C99 Binary Compiler
===================================================
Transpiles model.syn into pure ISO C99 and compiles an ultra-compact (<5 MB),
zero-dependency edge inference executable using NativeCompiler.
"""

import os
import sys
import argparse
from synapse.codegen.native_compiler import NativeCompiler


def build_edge_nanogpt(output_path: str = None) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_syn = os.path.join(current_dir, "model.syn")

    if output_path is None:
        is_win = sys.platform.startswith("win")
        output_path = os.path.join(current_dir, "nanogpt.exe" if is_win else "nanogpt")

    print(f"[*] Transpiling Edge NanoGPT (model.syn) to Pure C99...")
    compiler = NativeCompiler()
    c_info = compiler.find_c_compiler()
    if not c_info:
        print("[-] Error: No native C compiler found on system.")
        sys.exit(1)

    print(f"    Compiler detected: {c_info[0]} ({c_info[1]})")

    with open(model_syn, "r", encoding="utf-8") as f:
        source_code = f.read()

    temp_c = os.path.join(current_dir, "nanogpt_generated.c")
    success, res = compiler.compile_source_to_executable(
        source_code,
        output_path,
        temp_c_path=temp_c,
    )

    if not success:
        print(f"[-] Compilation error:\n{res}")
        sys.exit(1)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Edge NanoGPT C99 standalone binary built successfully!")
    print(f"    Binary: {output_path}")
    print(f"    Binary Size: {size_mb:.2f} MB (Target: < 5.0 MB)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Edge NanoGPT Standalone Binary")
    parser.add_argument("-o", "--output", help="Output binary path")
    args = parser.parse_args()

    build_edge_nanogpt(args.output)
