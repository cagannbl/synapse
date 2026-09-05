#!/usr/bin/env python3
"""
Synapse AI Standalone Binary Builder
====================================
Compiles the Synapse compiler and CLI into a zero-dependency, single-file executable
(synapse.exe on Windows, synapse on Linux/macOS) using Nuitka (primary) or PyInstaller (fallback).
"""

import os
import sys
import shutil
import argparse
import subprocess
from typing import Optional, Tuple


def check_builder_available(builder: str) -> bool:
    """Checks if the requested builder is installed and callable."""
    try:
        if builder == "nuitka":
            res = subprocess.run([sys.executable, "-m", "nuitka", "--version"], capture_output=True, text=True)
            return res.returncode == 0
        elif builder == "pyinstaller":
            res = subprocess.run(["pyinstaller", "--version"], capture_output=True, text=True)
            return res.returncode == 0
    except Exception:
        return False
    return False


def build_with_nuitka(entry_point: str, output_dir: str, output_name: str) -> Tuple[bool, str]:
    """Compiles single-file executable using Nuitka."""
    print(f"[*] Building with Nuitka (AOT C compilation)...")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile",
        "--assume-yes-for-downloads",
        "--include-package=synapse",
        f"--output-dir={output_dir}",
        f"--output-filename={output_name}",
        entry_point,
    ]

    print(f"    Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        out_path = os.path.join(output_dir, output_name)
        if os.path.isfile(out_path):
            return True, out_path
    return False, f"Nuitka build failed with exit code {res.returncode}"


def build_with_pyinstaller(entry_point: str, output_dir: str, output_name: str) -> Tuple[bool, str]:
    """Compiles single-file executable using PyInstaller."""
    print(f"[*] Building with PyInstaller...")
    os.makedirs(output_dir, exist_ok=True)

    # Runtime separator: ';' on Windows, ':' on POSIX
    sep = ";" if sys.platform.startswith("win") else ":"
    runtime_data = f"synapse/runtime/*{sep}synapse/runtime"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        os.path.splitext(output_name)[0],
        f"--distpath={output_dir}",
        f"--add-data={runtime_data}",
        "--hidden-import=synapse",
        "--hidden-import=synapse.codegen.embedded_runtime",
        entry_point,
    ]

    print(f"    Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        out_path = os.path.join(output_dir, output_name)
        if os.path.isfile(out_path):
            return True, out_path
    return False, f"PyInstaller build failed with exit code {res.returncode}"


def main():
    parser = argparse.ArgumentParser(description="Build Synapse standalone executable")
    parser.add_argument(
        "--builder",
        choices=["auto", "nuitka", "pyinstaller"],
        default="auto",
        help="Packaging engine to use (default: auto)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="dist",
        help="Directory to place output binary (default: dist)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify environment and command flags without executing compiler",
    )

    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    # Ensure embedded runtime is up to date before packaging
    from synapse.codegen.embedded_runtime import ensure_runtime_extracted
    ensure_runtime_extracted()

    entry_point = os.path.join("synapse", "cli.py")
    is_win = sys.platform.startswith("win")
    output_name = "synapse.exe" if is_win else "synapse"

    chosen_builder = args.builder
    if chosen_builder == "auto":
        if check_builder_available("nuitka"):
            chosen_builder = "nuitka"
        elif check_builder_available("pyinstaller"):
            chosen_builder = "pyinstaller"
        else:
            chosen_builder = "nuitka"  # default target

    print(f"==================================================")
    print(f"Synapse AI Standalone Binary Builder")
    print(f"  Target OS:     {sys.platform}")
    print(f"  Builder:       {chosen_builder}")
    print(f"  Entry Point:   {entry_point}")
    print(f"  Output Binary: {os.path.join(args.output_dir, output_name)}")
    print(f"==================================================")

    if args.dry_run:
        print("[*] Dry-run complete. Environment and embedded runtime validated.")
        sys.exit(0)

    if chosen_builder == "nuitka":
        if not check_builder_available("nuitka"):
            print("[!] Nuitka is not installed. Install with: pip install nuitka")
            sys.exit(1)
        success, res = build_with_nuitka(entry_point, args.output_dir, output_name)
    else:
        if not check_builder_available("pyinstaller"):
            print("[!] PyInstaller is not installed. Install with: pip install pyinstaller")
            sys.exit(1)
        success, res = build_with_pyinstaller(entry_point, args.output_dir, output_name)

    if success:
        print(f"[+] Build successful! Standalone binary: {res}")
        print(f"    Size: {os.path.getsize(res) / (1024 * 1024):.2f} MB")
    else:
        print(f"[-] Build error: {res}")
        sys.exit(1)


if __name__ == "__main__":
    main()
