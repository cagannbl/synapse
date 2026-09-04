# Contributing to Synapse AI

Welcome to the **Synapse AI** open-source contributor community! 🚀

Synapse is an AI-native systems programming language engineered to bridge the gap between Python's high-level elegance and C's zero-cost bare-metal performance. Whether you are fixing a compiler bug, accelerating tensor kernels with SIMD, adding standard library modules, or writing documentation, your contributions are warmly welcomed.

This guide provides an end-to-end walkthrough of our compiler pipeline, local development workflow, code standards, and curated entry-level tasks.

---

## 🏛️ 1. Architecture Overview

Synapse is designed with a strictly decoupled, modular compiler pipeline. Every source program (`.syn`) traverses five core stages before running as high-performance machine code:

```text
  ┌─────────────────┐
  │   Source Code   │  (.syn / .ai)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │      Lexer      │  synapse/lexer/
  │ (Indentation &) │  Tokenizes stream; emits INDENT, DEDENT, NEWLINE;
  │ (Context Tokens)│  handles contextual keywords (tensor, agent, prompt).
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │     Parser      │  synapse/parser/
  │  (AST Builder)  │  Recursive descent parser; builds typed AST nodes;
  │                 │  handles pipeline operators (|>), match, try/catch.
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │   Shape Guard   │  synapse/analyzer/
  │  (Static Solver)│  StaticShapeChecker & TypeChecker; validates dimensions
  │                 │  (e.g., [B, S, D] @ [B, D, H]) at compile time.
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │   C99 Emitter   │  synapse/codegen/
  │ (AOT Transpiler)│  Translates AST directly to clean, portable ISO C99;
  │                 │  zero third-party dependencies; supports WASM/PyExt.
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Arena Runtime  │  synapse/runtime/
  │ (Memory & Ops)  │  synapse_runtime.c / .h: Linear bump arena allocator
  │                 │  (syn_arena_t), OpenMP SIMD, reverse-mode autograd.
  └─────────────────┘
```

### Compiler Subsystems Deep Dive

| Component | Directory | Key Files | Responsibility |
| :--- | :--- | :--- | :--- |
| **Lexer** | `synapse/lexer/` | `lexer.py`, `token.py` | Indentation-sensitive tokenizer, handling f-strings, pipeline `\|>`, and contextual AI keywords. |
| **Parser** | `synapse/parser/` | `parser.py`, `ast.py` | Generates the Abstract Syntax Tree (AST), precedence parsing (`@` matmul, operators). |
| **Shape Guard** | `synapse/analyzer/` | `shape_guard.py`, `shape_checker.py`, `type_checker.py` | Compile-time symbolic shape inference engine. Prevents runtime matrix dimension mismatches. |
| **C99 Emitter** | `synapse/codegen/` | `c_emitter.py`, `native_compiler.py`, `wasm_compiler.py` | Transpiles AST to standalone ISO C99. Integrates with GCC, Clang, Zig, or MSVC. |
| **Arena Runtime** | `synapse/runtime/` | `synapse_runtime.c`, `synapse_runtime.h` | Zero-allocation linear arena memory model (`syn_arena_t`), reverse-mode autograd graph, and OpenMP vector kernels. |
| **Virtual Machine** | `synapse/vm/` | `virtual_machine.py`, `bytecode.py` | High-speed bytecode interpreter used for interactive REPL (`synapse repl`) and rapid iteration. |
| **Standard Library**| `synapse/stdlib/` | `math.py`, `crypto.py`, `fs.py`, `http.py`, `db.py` | Native modules for math, cryptography, Arrow IPC, database ORM, and HTTP/WebSocket servers. |

---

## 💻 2. Development Environment Setup

### Prerequisites
- **Python:** 3.10 or higher
- **C Compiler:** Any modern C99-compliant compiler:
  - **Zig** (strongly recommended for zero-friction cross-compilation: `zig cc`)
  - **GCC** (`gcc` 9.0+)
  - **Clang** (`clang` 11.0+)
  - **MSVC** (`cl.exe` Visual Studio 2022)
- **Git**

### Step-by-Step Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/synapse-lang/synapse.git
cd synapse

# 2. Create and activate an isolated virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install Synapse in editable development mode with test dependencies
pip install -e .
pip install pytest pytest-cov ruff black mypy

# 4. Verify compiler auto-detection
python -c "from synapse.codegen.native_compiler import NativeCompiler; print('Detected compiler:', NativeCompiler().find_c_compiler())"

# 5. Run the entire test suite (all 827+ tests must pass)
pytest
```

> [!TIP]
> **Using Zig as the Default C Compiler:** If you don't want to install huge build tools on Windows or Linux, install Zig (`winget install zig.zig` or `brew install zig`). Synapse's `NativeCompiler` automatically detects `zig cc` and uses it for blazing-fast standalone compilation!

---

## 🎯 3. Top 5 "Good First Issues" for New Contributors

Looking for an exciting place to start? We have curated five high-impact, beginner-friendly issues designed to get you familiar with different parts of the Synapse codebase:

### 🌟 Issue #1: Add New Neural Activation Functions (`Swish`, `Mish`, `Hardtanh`)
- **Domain:** Deep Learning Core & C99 Emitter
- **Relevant Files:**
  - `synapse/runtime/synapse_runtime.h` (declare `syn_swish`, `syn_mish`, `syn_hardtanh`)
  - `synapse/runtime/synapse_runtime.c` (implement forward and autograd backward math)
  - `synapse/codegen/c_emitter.py` (add expression emitting under `elif expr.right.name == "swish":`)
  - `synapse/core/tensor.py` (add Python autograd methods and backwards callbacks)
  - `synapse/vm/virtual_machine.py` (register REPL/VM dispatch)
- **Task Description:**
  1. Implement the mathematical functions:
     - **Swish (SiLU):** $f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$
     - **Mish:** $f(x) = x \cdot \tanh(\ln(1 + e^x))$
     - **Hardtanh:** $f(x) = \max(-1, \min(1, x))$
  2. Implement their backward derivatives for autograd in `synapse_runtime.c`.
  3. Add unit tests in `tests/test_nn.py` verifying numerical stability against float values.

---

### ⚡ Issue #2: SIMD & OpenMP Vectorization for ARM64 (NEON) and RISC-V (RVV)
- **Domain:** HPC & Systems Engineering
- **Relevant Files:**
  - `synapse/runtime/synapse_runtime.c`
  - `synapse/codegen/native_compiler.py`
  - `tests/test_quantization_simd.py`
- **Task Description:**
  - `synapse_runtime.c` currently utilizes `#pragma omp parallel for schedule(static)` for x86_64 AVX2/AVX-512 vectorization.
  - Add compile-time architecture guards (`#if defined(__ARM_NEON)` and `#if defined(__riscv_vector)`) to emit architecture-native SIMD intrinsics for elementwise tensor additions, ReLU, and GEMM inner loops.
  - Update `NativeCompiler` in `native_compiler.py` to forward `-march=armv8-a+simd` or RISC-V target flags when targeting ARM64/RISC-V.

---

### 📦 Issue #3: Standard Library Data Structures: `PriorityQueue` and `RingBuffer`
- **Domain:** Standard Library & Data Structures
- **Relevant Files:**
  - `synapse/stdlib/` (create `synapse/stdlib/collections.py`)
  - `synapse/stdlib/__init__.py`
  - `tests/test_stdlib_extended.py`
- **Task Description:**
  - Implement a binary-heap-backed **`PriorityQueue`** supporting `.push(item, priority)`, `.pop()`, and `.peek()`.
  - Implement a fixed-capacity, cache-friendly **`RingBuffer`** (Circular Buffer) optimized for streaming audio/token queues with `.push(val)`, `.pop()`, and `.is_full()`.
  - Provide both VM interoperability and C99 runtime bindings so standalone binaries can use high-speed circular buffers without dynamic reallocations.

---

### 🎨 Issue #4: VS Code TextMate Syntax Highlighting Enhancements
- **Domain:** Developer Experience (DevEx) & Tooling
- **Relevant Files:**
  - `vscode-synapse/syntaxes/synapse.tmLanguage.json`
  - `vscode-synapse/package.json`
  - `tests/test_vscode_syntax.py`
- **Task Description:**
  - Expand `synapse.tmLanguage.json` to highlight modern Synapse keywords introduced in v3:
    - Concurrency primitives: `spawn`, `channel`, `select`
    - AI swarm primitives: `agent`, `prompt`, `swarm`, `debate`, `memory`
    - Tensor type annotations: `Tensor[M, N]`, `f32`, `f64`, `i32`
    - Functional pipeline operator: `|>`
  - Ensure theme scopes match standard TextMate conventions (`keyword.control.ai.synapse`, `storage.type.tensor.synapse`).

---

### 🧠 Issue #5: Example Models: Edge MobileNet or ResNet-18 in Native Synapse
- **Domain:** Examples & Deep Learning Applications
- **Relevant Files:**
  - `examples/` (create `examples/models/resnet18.syn` or `mobilenet_edge.syn`)
  - `examples/README.md`
- **Task Description:**
  - Write a modular implementation of ResNet-18 or MobileNet-v1 using Synapse's `nn` module (`Conv2d`, `Linear`, `ReLU`, `Sequential`).
  - Demonstrate loading pre-trained weights from `.safetensors` via `safetensors.load()`.
  - Add a compilation script showing how to transpile the model to a standalone 0.21MB executable using `synapse emit-c --profile=standalone`.

---

## 📐 4. Code Standards & Review Policy

To preserve the zero-overhead, safety, and reliability guarantees of Synapse, all pull requests must satisfy these requirements:

### A. Strict ISO C99 Portability
- Any C code in `synapse/runtime/` or emitted by `synapse/codegen/` must adhere strictly to the **ISO C99** standard.
- **No compiler-specific extensions** (no GCC/Clang built-ins unless properly guarded with `#ifdef`).
- Code must compile cleanly with `-Wall -Wextra -pedantic` without any warnings under GCC, Clang, MSVC, and Zig.

### B. Memory Lifecycle & Arena Safety
- Synapse uses a scoped Arena Allocator (`syn_arena_t`) to prevent memory fragmentation and GC pauses in standalone mode.
- Avoid unchecked `malloc()` / `free()` inside tight loops. Use `syn_arena_alloc()` or tensor arena constructors (`syn_tensor_create_arena`).
- Check that arena scopes (`syn_arena_scope_enter` / `syn_arena_scope_leave`) are strictly balanced.

### C. Zero Regressions & 100% Test Pass Rate
- Synapse has **827+ automated tests**. Your PR **must not break any existing test**.
- Every bug fix must include a regression test reproducing the issue.
- Every new feature must include comprehensive unit and integration tests.

```bash
# Run the fast unit tests
pytest tests/test_lexer.py tests/test_parser.py tests/test_c_emitter.py

# Run memory and shape verification tests
pytest tests/test_c_arena_scope.py tests/test_static_shape_checker.py

# Run full test suite before committing
pytest
```

### D. Git Commit Message Conventions
We follow the [Conventional Commits](https://www.conventionalcommits.org/) standard:

- `feat(codegen): add support for swish and mish activation functions`
- `fix(runtime): resolve memory leak in arena scope unwind`
- `perf(simd): implement NEON vectorization for tensor addition`
- `docs(launch): update Show HN technical architecture notes`
- `test(shape): add static verification tests for 3D broadcasting`

---

## 🚀 5. Pull Request Submission Workflow

1. **Create a topic branch:**
   ```bash
   git checkout -b feat/add-swish-activation
   ```
2. **Make your changes and write tests.**
3. **Format your code:**
   ```bash
   black synapse/ tests/
   ruff check synapse/ tests/ --fix
   ```
4. **Run all tests:**
   ```bash
   pytest
   ```
5. **Push and open a Pull Request:**
   - Follow the provided [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
   - Link the relevant issue (e.g., `Closes #42`).
   - Include test execution logs and benchmark results if applicable.

---

## 💬 6. Community & Communication

- **GitHub Discussions:** Ask architectural questions, propose RFCs, or share projects built with Synapse.
- **Issue Tracker:** Report verified bugs or track roadmap milestones.
- **Code of Conduct:** We are committed to providing a friendly, safe, and welcoming environment for all contributors. Be respectful and collaborative.

Thank you for helping build the future of AI-native systems programming! 🌟
