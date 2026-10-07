#!/usr/bin/env python3
"""
Synapse benchmark suite.

Every number this script prints is measured on the machine running it; nothing
is hardcoded or simulated. Comparisons are only made against baselines that are
also measured here.

1. Native binary size   - compiles examples/edge_nanogpt to a native executable,
                          runs it, and reports its size on disk.
2. DataLoader throughput - SynapseFastDataLoader vs. plain NumPy slicing and a
                          multiprocessing + pickle queue, over the same data.
3. Shape verification   - latency of the static shape checker on a mismatched
                          matmul, and whether it detects the error before running.

Usage:
    python benchmarks/run_all.py [--json] [--quick] [--no-color]
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from synapse.analyzer.shape_checker import check_shapes
from synapse.codegen.native_compiler import NativeCompiler
from synapse.core.tensor import Tensor
from synapse.data.dataloader import SynapseFastDataLoader


# =============================================================================
# Result types
# =============================================================================

@dataclass
class BinarySizeResult:
    status: str                      # "measured" | "skipped" | "failed"
    compiler: Optional[str] = None
    binary_bytes: Optional[int] = None
    runs_successfully: Optional[bool] = None
    detail: str = ""

    @property
    def binary_kb(self) -> Optional[float]:
        return None if self.binary_bytes is None else round(self.binary_bytes / 1024, 1)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["binary_kb"] = self.binary_kb
        return d


@dataclass
class DataLoaderResult:
    status: str
    num_samples: int
    batch_size: int
    samples_per_sec: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShapeCheckResult:
    status: str
    mismatch_detected: bool
    median_latency_ms: float
    error_message: str
    suggested_fix: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteReport:
    timestamp: float
    quick_mode: bool
    platform: str
    binary_size: BinarySizeResult
    dataloader: DataLoaderResult
    shape_check: ShapeCheckResult

    @property
    def all_ok(self) -> bool:
        return all(r.status in ("measured", "skipped") for r in (self.binary_size, self.dataloader, self.shape_check))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "success" if self.all_ok else "failed",
            "timestamp": self.timestamp,
            "quick_mode": self.quick_mode,
            "platform": self.platform,
            "benchmarks": {
                "binary_size": self.binary_size.to_dict(),
                "dataloader": self.dataloader.to_dict(),
                "shape_check": self.shape_check.to_dict(),
            },
        }


# =============================================================================
# 1. Native binary size
# =============================================================================

def run_binary_size_benchmark(quick: bool = False) -> BinarySizeResult:
    compiler = NativeCompiler()
    info = compiler.find_c_compiler()
    if not info:
        return BinarySizeResult(status="skipped", detail="no C compiler found")

    model = os.path.join(PROJECT_ROOT, "examples", "edge_nanogpt", "model.syn")
    with open(model, encoding="utf-8") as f:
        source = f.read()

    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "nanogpt.exe" if sys.platform.startswith("win") else "nanogpt")
        ok, res = compiler.compile_source_to_executable(source, exe, temp_c_path=os.path.join(tmp, "nanogpt.c"))
        if not ok:
            return BinarySizeResult(status="failed", compiler=info[1], detail=str(res)[:500])
        size = os.path.getsize(exe)
        run = subprocess.run([exe], capture_output=True, text=True, timeout=60)

    return BinarySizeResult(
        status="measured",
        compiler=info[1],
        binary_bytes=size,
        runs_successfully=run.returncode == 0,
        detail="examples/edge_nanogpt/model.syn (one transformer block, toy 2x2 weights)",
    )


# =============================================================================
# 2. DataLoader throughput
# =============================================================================

def _mp_producer(data: np.ndarray, batch_size: int, queue) -> None:
    for i in range(0, len(data), batch_size):
        queue.put(data[i:i + batch_size])  # pickled across the process boundary
    queue.put(None)


def _time_loop(batches) -> float:
    t0 = time.perf_counter()
    seen = 0
    for batch in batches:
        seen += batch.shape[0]
    return seen / max(time.perf_counter() - t0, 1e-9)


def run_dataloader_benchmark(quick: bool = False) -> DataLoaderResult:
    num_samples = 12_000 if quick else 96_000
    batch_size = 256
    data = np.random.default_rng(0).standard_normal((num_samples, 64)).astype(np.float32)

    results: Dict[str, float] = {}

    results["numpy_slicing"] = _time_loop(data[i:i + batch_size] for i in range(0, num_samples, batch_size))

    loader = SynapseFastDataLoader(Tensor(data), batch_size=batch_size, prefetch_factor=3, shuffle=False)
    results["synapse_fast_dataloader"] = _time_loop(iter(loader))

    queue = mp.Queue(maxsize=8)
    proc = mp.Process(target=_mp_producer, args=(data, batch_size, queue))
    t0 = time.perf_counter()
    proc.start()
    seen = 0
    while (batch := queue.get()) is not None:
        seen += batch.shape[0]
    elapsed = time.perf_counter() - t0  # includes worker start-up, as a real loader pays it too
    proc.join()
    results["multiprocessing_pickle_queue"] = seen / max(elapsed, 1e-9)

    return DataLoaderResult(
        status="measured",
        num_samples=num_samples,
        batch_size=batch_size,
        samples_per_sec={k: round(v, 1) for k, v in results.items()},
    )


# =============================================================================
# 3. Static shape verification
# =============================================================================

MISMATCHED_MATMUL = """
let A: Tensor[32, 64] = zeros([32, 64])
let B: Tensor[128, 64] = zeros([128, 64])
let C = A @ B
""".strip()


def run_shape_check_benchmark(quick: bool = False) -> ShapeCheckResult:
    times: List[float] = []
    reports = []
    for _ in range(5 if quick else 25):
        t0 = time.perf_counter()
        reports = check_shapes(MISMATCHED_MATMUL, raise_on_error=False)
        times.append((time.perf_counter() - t0) * 1000)

    detected = len(reports) > 0
    return ShapeCheckResult(
        status="measured" if detected else "failed",
        mismatch_detected=detected,
        median_latency_ms=round(float(np.median(times)), 3),
        error_message=reports[0].message if detected else "",
        suggested_fix=reports[0].suggested_fix if detected else None,
    )


# =============================================================================
# Reporting
# =============================================================================

class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdout.isatty()

    def bold(self, s: str) -> str:
        return f"\033[1m{s}\033[0m" if self.enabled else s

    def dim(self, s: str) -> str:
        return f"\033[2m{s}\033[0m" if self.enabled else s


def print_report(report: SuiteReport, style: Style) -> None:
    print(style.bold("Synapse benchmarks") + style.dim(f"  ({report.platform}{', quick' if report.quick_mode else ''})"))
    print(style.dim("All values measured on this machine.\n"))

    b = report.binary_size
    print(style.bold("1. Native binary size (Edge NanoGPT example)"))
    if b.status == "measured":
        print(f"   {b.binary_kb} KB  compiled with {b.compiler}, runs: {'yes' if b.runs_successfully else 'NO'}")
        print(style.dim(f"   {b.detail}"))
    else:
        print(f"   {b.status}: {b.detail}")
    print()

    d = report.dataloader
    print(style.bold(f"2. DataLoader throughput ({d.num_samples:,} samples, batch {d.batch_size})"))
    width = max(len(k) for k in d.samples_per_sec)
    for name, rate in sorted(d.samples_per_sec.items(), key=lambda kv: -kv[1]):
        print(f"   {name.ljust(width)}  {rate:>15,.0f} samples/s")
    print()

    s = report.shape_check
    print(style.bold("3. Static shape verification (mismatched matmul)"))
    print(f"   detected before execution: {'yes' if s.mismatch_detected else 'NO'}   median latency: {s.median_latency_ms} ms")
    if s.mismatch_detected:
        print(style.dim(f"   {s.error_message}"))
        if s.suggested_fix:
            print(style.dim(f"   suggested fix: {s.suggested_fix}"))


def run_suite(quick: bool = False) -> SuiteReport:
    import platform

    return SuiteReport(
        timestamp=time.time(),
        quick_mode=quick,
        platform=f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        binary_size=run_binary_size_benchmark(quick),
        dataloader=run_dataloader_benchmark(quick),
        shape_check=run_shape_check_benchmark(quick),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Synapse benchmark suite.")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--quick", action="store_true", help="smaller workloads for a fast run")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI styling")
    args = parser.parse_args()

    report = run_suite(quick=args.quick)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report, Style(enabled=not args.no_color))
    return 0 if report.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
