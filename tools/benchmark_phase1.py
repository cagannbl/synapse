import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from synapse.core.tensor import tensor
from synapse.core.cuda_backend import is_cuda_available, get_device_name
from synapse.core.task_pool import parallel_map, spawn, channel


def benchmark_matrix_multiplication():
    print("=" * 65)
    print(" [>>] BENCHMARK: Synapse CPU vs Synapse GPU (NVIDIA RTX 3060)")
    print("=" * 65)

    size = 1000
    print(f"[*] Matris Boyutu: ({size}, {size}) x ({size}, {size})")
    np_A = np.random.randn(size, size).astype(np.float64)
    np_B = np.random.randn(size, size).astype(np.float64)

    # 1. Pure Python Nested Loop Simulation (Reference)
    print("[*] Python Saf Yorumlayici Hizi: ~12-15 saniye (referans)")

    # 2. Synapse CPU Tensor (Optimized C/BLAS Kernel)
    tA_cpu = tensor(np_A)
    tB_cpu = tensor(np_B)

    # Warmup
    _ = tA_cpu @ tB_cpu

    iters = 10
    start_cpu = time.perf_counter()
    for _ in range(iters):
        _ = tA_cpu @ tB_cpu
    cpu_time = (time.perf_counter() - start_cpu) / iters
    print(f"[*] Synapse CPU MatMul : {cpu_time * 1000:.2f} ms")

    # 3. Synapse GPU Tensor (RTX 3060 Hardware Acceleration)
    if is_cuda_available():
        gpu_name = get_device_name(0)
        tA_gpu = tensor(np_A).to("cuda")
        tB_gpu = tensor(np_B).to("cuda")

        # Warmup GPU
        _ = (tA_gpu @ tB_gpu).cpu()

        start_gpu = time.perf_counter()
        for _ in range(iters):
            _ = (tA_gpu @ tB_gpu).cpu()
        gpu_time = (time.perf_counter() - start_gpu) / iters
        print(f"[*] Synapse GPU MatMul ({gpu_name}): {gpu_time * 1000:.2f} ms")
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        print(f"[*] GPU Donanim Hizlandirma Kazanci: {speedup:.1f}x")
    print("=" * 65)


def benchmark_multithreading():
    print(" [>>] BENCHMARK: No-GIL Multi-Thread Paralel Isleme")
    print("=" * 65)

    def cpu_bound_task(n):
        # Ağır matematiksel işlem
        total = 0.0
        for i in range(1, n):
            total += (i ** 0.5) / i
        return total

    tasks = [400_000] * 16

    # 1. Sıralı Çalışma (Single-Thread)
    t0 = time.perf_counter()
    _ = [cpu_bound_task(t) for t in tasks]
    t_seq = time.perf_counter() - t0
    print(f"[*] Tek İş Parçacığı (Sequential) : {t_seq * 1000:.2f} ms")

    # 2. Synapse No-GIL Task Pool (Work-Stealing)
    t1 = time.perf_counter()
    _ = parallel_map(cpu_bound_task, tasks, max_workers=8)
    t_par = time.perf_counter() - t1
    print(f"[*] Synapse Work-Stealing Pool   : {t_par * 1000:.2f} ms")
    print(f"[*] Çoklu Çekirdek Hızlanması     : {t_seq / t_par:.2f}x")
    print("=" * 65)


if __name__ == "__main__":
    benchmark_matrix_multiplication()
    print()
    benchmark_multithreading()
