"""
Tests for Synapse CLI Instant Showcase Demos (synapse demo)
===========================================================
Validates:
- synapse demo --preset=nanogpt (compiles and runs the NanoGPT example natively)
- synapse demo --preset=matmul (2x2 matrix multiplication, autograd graph, O(1) arena)
- synapse demo (default preset -> nanogpt)
- synapse demo --preset=tour (6-step interactive tour in non-interactive mode)
- synapse demo --preset=dataloader (high-throughput zero-starvation dataloader)
- invalid preset rejection
"""

import re
import subprocess
import sys
import pytest


def run_cli(*args):
    cmd = [sys.executable, "-m", "synapse.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def test_cli_demo_nanogpt():
    """Verify 'synapse demo --preset=nanogpt' runs the C99 NanoGPT forward pass cleanly."""
    res = run_cli("demo", "--preset=nanogpt")
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert "SYNAPSE DEMO: EDGE NANOGPT" in res.stdout
    assert re.search(r"Size:\s+\d+\.\d KB", res.stdout)
    # The program output shown is the binary's real stdout
    assert "[Stage 1]" in res.stdout
    assert "[Stage 4]" in res.stdout
    assert "Edge NanoGPT forward pass completed." in res.stdout


def test_cli_demo_matmul():
    """Verify 'synapse demo --preset=matmul' runs 2x2 matmul, autograd graph, and arena lifecycle."""
    res = run_cli("demo", "--preset=matmul")
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert "SYNAPSE DEMO: MATMUL, AUTOGRAD & O(1) ARENA" in res.stdout
    assert "Matrix Multiplication" in res.stdout
    assert "Autograd" in res.stdout
    assert "A.grad" in res.stdout
    assert "B.grad" in res.stdout
    assert "syn_arena_t Kapasitesi:" in res.stdout
    assert "0 bytes (Garbage Collector Yok)" in res.stdout
    assert "PASS" in res.stdout


def test_cli_demo_default():
    """Verify running 'synapse demo' without arguments defaults to nanogpt showcase."""
    res = run_cli("demo")
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert "SYNAPSE DEMO: EDGE NANOGPT" in res.stdout
    assert re.search(r"Size:\s+\d+\.\d KB", res.stdout)


def test_cli_demo_tour():
    """Verify 'synapse demo --preset=tour --non-interactive' displays all 6 tour steps."""
    res = run_cli("demo", "--preset=tour", "--non-interactive")
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert "TOUR OF SYNAPSE" in res.stdout
    assert "ADIM 1/6" in res.stdout
    assert "ADIM 6/6" in res.stdout
    assert "Tour of Synapse başarıyla tamamlandı!" in res.stdout


def test_cli_demo_dataloader():
    """Verify 'synapse demo --preset=dataloader' executes zero-starvation dataloader benchmark."""
    res = run_cli("demo", "--preset=dataloader")
    assert res.returncode == 0, f"Error: {res.stderr}"
    assert "SYNAPSE DEMO: ZERO-STARVATION DATALOADER" in res.stdout
    assert "Tüm örnekler alındı: 512/512" in res.stdout
    assert "samples/sec" in res.stdout


def test_cli_demo_invalid_preset():
    """Verify invalid presets are rejected by the CLI parser."""
    res = run_cli("demo", "--preset=nonexistent_preset")
    assert res.returncode != 0
