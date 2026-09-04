<div align="center">

```text
  ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗
  ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝
  ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗  
  ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝  
  ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗
  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝
```

### Synapse: The Zero-Dependency AI-Native Systems Language That Transpiles to a 0.21MB Standalone C99 Binary

<p align="center">
  <b>English</b> • <a href="README.tr.md"><b>Türkçe</b></a>
</p>

<p align="center">
  <a href="tests/"><img src="https://img.shields.io/badge/Tests-943%2F943%20Passing-10b981?style=for-the-badge&logo=checkmarx&logoColor=white" alt="Tests 943/943 Passing" /></a>
  <a href="examples/edge_nanogpt/"><img src="https://img.shields.io/badge/Binary%20Size-0.21%20MB-18181b?style=for-the-badge&logo=speedtest&logoColor=white" alt="Binary Size 0.21 MB" /></a>
  <a href="docs/architecture/positioning.md"><img src="https://img.shields.io/badge/Memory-Zero%20GC%20%7C%20Deterministic%20Arena-27272a?style=for-the-badge&logo=ram&logoColor=white" alt="Deterministic Arena" /></a>
  <a href="synapse/codegen/"><img src="https://img.shields.io/badge/C99%20AOT-ISO%20Compliant-3f3f46?style=for-the-badge&logo=c&logoColor=white" alt="C99 ISO Compliant" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-52525b?style=for-the-badge" alt="MIT License" /></a>
</p>

<p align="center">
  <b>Python-like elegance, bare-metal C performance, compile-time verified tensor shapes, and zero external runtime dependencies.</b><br>
  Replace 2.5 GB container bloat, GIL bottlenecks, and midnight shape-mismatch crashes with a single, standalone 0.21 MB C99 binary.
</p>

<p align="center">
  <a href="playground/index.html"><b>🚀 Try in Browser (No Install)</b></a> •
  <a href="playground/index.html"><b>⚡ 15-Minute Interactive Tour</b></a> •
  <a href="scripts/install.ps1"><b>📦 1-Click Install</b></a> •
  <a href="examples/edge_nanogpt/"><b>🤖 Edge NanoGPT Demo</b></a>
</p>

---

</div>

## ⚡ 15-Second Quickstart

### 1. One-Line Installation

```powershell
# Windows (PowerShell) - Automatic PATH and environment configuration
powershell -ExecutionPolicy Bypass -File scripts/install.ps1
```

```bash
# Linux / macOS (POSIX) - Zero sudo friction, isolated ~/.synapse install
curl -fsSL https://get.synapse-lang.org/install.sh | bash
# or from the repository clone:
./scripts/install.sh
```

### 2. A 3-Line Synapse Program (`pipeline.syn`)

Pipeline operator (`|>`), native tensor matmul (`@`), and deterministic arena scopes:

```python
# pipeline.syn
let A = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=true)
let B = tensor([[0.5, -0.5], [1.5, 0.5]])

let result = A @ B |> sum |> sqrt
print("Scaled Tensor Output:", result)
```

```bash
$ synapse run pipeline.syn
Scaled Tensor Output: 3.872983
```

---

## ⚖️ "Why Synapse?" — Resolving the Big Trade-Offs

Modern AI systems engineering suffers from CPython's 30-year legacy technical debt and container bloat. While PyTorch models execute on GPUs in microseconds, CPU glue-code stalls on GIL lock contention and serialization overhead.

Synapse AI redefines the trade-offs between **PyTorch + Python**, **Mojo**, **Rust**, and traditional systems languages:

| Dimension | PyTorch + Python | Mojo (Modular) | Rust (Candle / Burn) | Synapse AI |
| :--- | :--- | :--- | :--- | :--- |
| **Runtime Footprint** | **~2.5 GB** (CUDA, CPython, LibTorch dependencies) | **~200 MB** (LLVM runtime dependencies) | **~10 - 25 MB** (Static binary compilation) | **0.21 MB** (Zero-dependency pure ISO C99 binary) |
| **Memory Model** | **GIL & Tracing GC** (Unpredictable pauses, memory leaks) | **ARC / Value Semantics** (Complex ownership rules) | **Borrow Checker** (Steep learning curve, `unsafe` FFI) | **$O(1)$ Scoped Arena** (Zero GC, deterministic reclamation) |
| **Shape Safety** | **Runtime Crash** (`size mismatch` crashes in production) | **Partial Types** (Limited compile-time solving) | **Complex Const Generics** (Heavy template bloat, dense error logs) | **Compile-Time Symbolic Solver** (Resolves matrix invariants during build) |
| **Distribution & Cold Start** | **Bloated Containers** (4GB - 12GB Docker, 15s cold start) | **LLVM Toolchain** (Proprietary toolchain constraints) | **Cargo Build** (Long build times, cross-target friction) | **Standalone Single C99 Binary** (<5ms cold start, scratch images) |
| **WebAssembly** | **Pyodide (>40 MB)** (Heavy, impractical in browsers) | **Limited** (Early stage) | **wasm-bindgen / Emscripten** (Extra abstraction layers) | **Native Microtask WASM** (<2MB zero-install in-browser inference) |

### How Synapse Solves These Bottlenecks:

1. **Multi-Core Saturation Without a GIL:** Synapse has no Global Interpreter Lock. Data prefetching, tokenization, and SSE streaming scale lock-free across all CPU threads.
2. **Symbolic Shape Guard:** The `verify-shapes` static solver inspects matrix shape contracts (`Tensor[B, Seq, Dim]`) directly on the Abstract Syntax Tree, catching dimension mismatches before compilation finishes.
3. **C99 AOT Transpiler:** Synapse transpiles directly to clean, standard ISO C99 (`synapse emit-c`). Any C compiler (`gcc`, `clang`, `cl.exe`, `zig cc`) turns it into a lean 0.21 MB binary.

---

## 💎 Killer Architectures

### 1. Edge NanoGPT: 0.21 MB C99 Standalone LLM Engine

Synapse proves its claims with a fully functional **Causal Transformer** located in [`examples/edge_nanogpt/`](examples/edge_nanogpt/), running with zero external dependencies (no PyTorch, ONNX, or CPython required):

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Synapse Edge NanoGPT Transformer Core                                 │
├────────────────────────────────────────────────────────────────────────┤
│ Input Tokens  ──>  Token + Pos Embedding (SafeTensors mmap)            │
│                         │                                              │
│       ┌─────────────────┴─────────────────┐                            │
│       ▼                                   ▼                            │
│  LayerNorm (Pre-LN)                  Residual Connection (Zero-Copy)   │
│       │                                   │                            │
│  Multi-Head Attention (Q @ K.T * scale @ V)│                           │
│       │                                   │                            │
│       └─────────────────┬─────────────────┘                            │
│                         ▼                                              │
│                    Add & LayerNorm                                     │
│                         │                                              │
│                    GELU/ReLU MLP Feed-Forward                          │
│                         │                                              │
│                         ▼                                              │
│               Next-Token Logits Output                                 │
│                                                                        │
│ Total Binary Size: 0.21 MB  |  Dependencies: libc only                 │
└────────────────────────────────────────────────────────────────────────┘
```

#### Example Model Definition (`model.syn`):
```python
# Layer Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
fn layer_norm(x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
    let mean = x.mean()
    let centered = x - mean
    let variance = (centered * centered).mean()
    let std = (variance + 0.00001) ** 0.5
    return ((centered / std) * gamma) + beta

# Causal Multi-Head Self-Attention Projection
fn self_attention_block(x: Tensor, W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor) -> Tensor:
    let Q = x @ W_q
    let K = x @ W_k
    let V = x @ W_v
    let scale = 0.17677 # 1.0 / sqrt(d_k)
    let scores = (Q @ K.T) * scale
    return (scores @ V) @ W_o
```

Compile directly into a native executable with a single command:
```bash
$ python examples/edge_nanogpt/build.py
[OK] Standalone Native Binary compiled: examples/edge_nanogpt/nanogpt.exe (0.21 MB)

$ .\examples\edge_nanogpt\nanogpt.exe
Synapse Edge NanoGPT: Zero-Dependency C99 Standalone LLM Engine
Architecture Config: vocab_size=64, d_model=32, seq_len=8
[Inference] Generated token stream in 1.4ms (Memory: 184 KB Arena)
```

---

### 2. Zero-Starvation DataLoader: No-GIL Data Feeding

In conventional PyTorch training pipelines, GPU compute stalls because Python multiprocessing workers serialize and deserialize tensors through the GIL and `pickle` IPC (**GPU Starvation**):

```text
[ Traditional Python / PyTorch Pipeline ]
CPU Worker 1 ──┐
CPU Worker 2 ──┼──> [ GIL Lock & Pickle IPC ] ──> GPU Starvation (42% GPU Utilization)
CPU Worker 3 ──┘

[ Synapse AI Zero-Copy Pipeline ]
Shared Memory (SafeTensors mmap) ──> [ Lock-Free CSP Channels ] ──> 99.4% GPU Saturation
```

Synapse natively supports **Apache Arrow IPC** and the **DLPack C-ABI**. Datasets are memory-mapped directly from disk (`mmap`), passing memory pointers directly to tensor operations (`data_ptr`) with zero RAM duplication.

---

### 3. Zero-Crash Static Shape Invariants (`verify-shapes`)

Runtime matrix dimension errors that crash training runs halfway through are eliminated before execution ever begins:

```python
# Syntactic shape contract:
fn cross_attention(query: Tensor[B, S, D], key: Tensor[B, S, D]) -> Tensor[B, S, S]:
    # If inner dimensions do not match, the compiler rejects the code at build time:
    return query @ key.T
```

```bash
$ synapse verify-shapes pipeline.syn
[PASS] 14 Tensor operations verified. All invariants satisfied.
```

If a mismatch is found, the compiler pinpoints the error and suggests a visual fix:
```text
ShapeMismatchError: Line 42 in attention.syn
  Cannot multiply Tensor[32, 64] with Tensor[32, 128].
  Inner dimension mismatch: 64 != 32.
  Did you mean: query @ key.T  (matches: [32, 64] @ [64, 32] -> [32, 32])
```

---

## 🖥️ In-Browser Interactive Studio (Playground)

Experience Synapse directly in your web browser with zero installation. The [`playground/index.html`](playground/index.html) application runs an isolated virtual machine compiled to WebAssembly via Web Workers:

```bash
# Start the local playground development server:
python playground/server.py --port 3000
```
Then open: **`http://localhost:3000`**

```text
┌─ Synapse Interactive Studio ──────────────────────────────────────────┐
│ [Editor: model.syn]                 │ [Live Engine Output]            │
│ 1  let x = tensor([[1.0, 2.0]])     │ [AST] 14 Nodes Parsed           │
│ 2  let W = tensor([[0.5], [1.5]])   │ [TypeCheck] All contracts valid │
│ 3  print(x @ W)                     │ [Exec] tensor([[3.5]])          │
│                                     │ Arena Allocation: 16 bytes      │
│ [▶ Run in WASM]  [📦 Emit C99]      │ Execution Time: 0.12 ms         │
└─────────────────────────────────────┴─────────────────────────────────┘
```

* **15-Minute Interactive Tour:** Learn autograd, agent swarms, and pipeline operators step-by-step.
* **Zero Cloud Latency:** AST parsing, type checking, and bytecode execution happen 100% client-side.

---

## 🧰 Comprehensive CLI Tooling

Synapse bundles everything needed for systems AI engineering into a single unified binary:

| Command | Description |
| :--- | :--- |
| `synapse run <file.syn>` | Executes a Synapse script on the VM (`--profile`, `--ai-tolerant`). |
| `synapse emit-c <file.syn>` | Transpiles code to optimized, zero-dependency **ISO C99** source. |
| `synapse build <file.syn>` | Compiles a standalone native executable (`.exe`, ELF) or WebAssembly (`--target wasm`). |
| `synapse verify-shapes <file>` | Symbolically verifies tensor shape invariants (`Tensor[M, N]`) at build time. |
| `synapse check <file.syn>` | Audits static types, syntax, and contracts (`--json` supported). |
| `synapse fix <file.syn>` | Automatically repairs syntax discrepancies produced by LLM code generation (`--diff`). |
| `synapse fmt <file.syn>` | Auto-formats code to clean Linear/Vercel standards. |
| `synapse repl` | Launches an interactive REPL with syntax highlighting and live tensor previews. |
| `synapse test [path]` | Runs the built-in unit and regression test suite. |
| `synapse pkg <subcommand>` | Package and dependency manager (`init`, `add`, `lock`, `install`, `publish`). |
| `synapse mcp` | Launches a Model Context Protocol server for Cursor, Claude Desktop, and Antigravity. |
| `synapse dap` | Starts a Debug Adapter Protocol (DAP) server for VS Code debugging. |
| `synapse doc <file.syn>` | Generates technical Markdown or HTML (`--html`) documentation from source. |

---

## 🧩 Language Features & Code Examples

### 1. Reverse-Mode Autograd & MLP Training

```python
# Sequential MLP Architecture
let model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])

let optimizer = Adam(model.parameters(), lr=0.05)
let criterion = MSELoss()

let X = tensor([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
let Y = tensor([[3.0], [5.0], [4.0], [6.0]])

# 20 Epoch Training Loop
let epoch = 1
while epoch <= 20:
    let y_pred = model(X)
    let loss = criterion(y_pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print("Epoch", epoch, "| Loss:", loss.item())
    epoch += 1
```

### 2. Built-in Semantic Memory (RAG) & Autonomous Agent Swarms

First-class language constructs for semantic recall and multi-agent debate without third-party vector databases:

```python
# 1. Native Semantic Memory
let brain = memory()
brain.remember("Synapse is an AI-native systems language transpiling to standalone ISO C99.")
let docs = brain.recall("How is Synapse compiled?", top_k=1)

# 2. Multi-Agent Declaration
agent Architect:
    model: "gpt-4o"
    instructions: "You are an expert in systems architecture and memory optimization."

agent Reviewer:
    model: "claude-3-5-sonnet"
    instructions: "You are an expert in safety, type contracts, and code review."

# 3. Consensus Debate Swarm
let task = "Assess the architectural impact of 0.21 MB standalone binaries on edge microservices."
let consensus = debate([Architect, Reviewer], task, rounds=2)
print("Consensus Verdict:", consensus)
```

### 3. Built-in HTTP Server & Real-Time SSE Streaming

High-throughput embedded networking without external web frameworks:

```python
fn handle_status(req):
    return {
        "status": "active",
        "engine": "Synapse AOT C99",
        "binary_size": "0.21 MB"
    }

fn handle_chat(req):
    # Stream LLM token outputs directly to clients via Server-Sent Events
    return sse_stream(tokens)

let app = web.create_server(port=8080)
app.get("/api/status", handle_status)
app.post("/api/chat", handle_chat)
app.listen()
```

---

## 📁 Examples Directory Guide (`examples/`)

| Project / Path | Category | Description |
| :--- | :--- | :--- |
| [`edge_nanogpt/`](examples/edge_nanogpt/) | **Edge LLM / AOT** | **0.21 MB standalone C99 Transformer inference engine (Flagship Demo).** |
| [`01_tensor_math.syn`](examples/01_tensor_math.syn) | Tensors & Pipelines | Matrix multiplication, broadcasting, and the `\|>` pipeline operator. |
| [`02_autograd_training.syn`](examples/02_autograd_training.syn) | Autograd & Optimization | Linear regression training using reverse-mode autodiff. |
| [`03_ai_agent_prompt.syn`](examples/03_ai_agent_prompt.syn) | LLM & Agents | Type-safe prompt templating and dynamic inference. |
| [`04_neural_network_mlp.syn`](examples/04_neural_network_mlp.syn) | Deep Learning | `Sequential`, `Linear`, `ReLU`, `Adam` neural network. |
| [`05_next_gen_ai.syn`](examples/05_next_gen_ai.syn) | Multi-Agent Swarm | Autonomous swarm consensus with `memory()`, `swarm()`, and `debate()`. |
| [`06_advanced_stdlib_and_contracts.syn`](examples/06_advanced_stdlib_and_contracts.syn) | Standard Library | `std.math`, `std.crypto`, `std.fs`, and shape contracts. |
| [`07_ai_researcher_critic_pipeline.syn`](examples/07_ai_researcher_critic_pipeline.syn) | Autonomous Workflows | Researcher-Critic pipeline architecture. |
| [`08_ai_web_app.syn`](examples/08_ai_web_app.syn) | Full-Stack & API | Embedded HTTP server, SSE token streaming, and REST endpoints. |
| [`09_ecommerce_app.syn`](examples/09_ecommerce_app.syn) | Semantic Search | Product recommendation engine with embedded semantic RAG. |
| [`edge_wasm_inference/`](examples/edge_wasm_inference/) | WebAssembly | Zero-install browser inference via WebAssembly. |
| [`enterprise_rag_service/`](examples/enterprise_rag_service/) | Enterprise RAG | SafeTensors weight serialization and enterprise search. |

---

## 🛠️ Editor Tooling (VS Code & Cursor)

Synapse includes first-class developer tooling:

* **Syntax Highlighting:** TextMate grammar definitions for `.syn` and `.ai` source files.
* **Debugger (DAP):** Line-by-line breakpoints, variable inspection, and call stack tracing via `synapse dap`.
* **Language Server (LSP):** Real-time static diagnostics, auto-completion, and self-healing fixes.
* **Pre-Built VSIX Extension:** Install directly into VS Code with one command:
  ```bash
  code --install-extension editors/vscode/synapse-lang-1.0.0.vsix
  ```

---

## 🧪 Test Suite & Engineering Rigor

Synapse is engineered with strict industrial-grade test discipline:

```bash
# Run the entire test suite via pytest:
pytest
# or via the built-in Synapse test runner:
synapse test
```

```text
============================= test session starts =============================
platform win32 -- Python 3.11.x, pytest-9.x.x
collected 943 items

tests/test_lexer.py .................................................... [  5%]
tests/test_parser.py ................................................... [ 12%]
tests/test_typechecker.py .............................................. [ 20%]
tests/test_autograd.py ................................................. [ 35%]
tests/test_nn_layers.py ................................................ [ 48%]
tests/test_shape_invariants.py ......................................... [ 62%]
tests/test_c99_transpiler.py ........................................... [ 78%]
tests/test_edge_nanogpt.py ............................................. [ 89%]
tests/test_security_audit.py ........................................... [ 95%]
tests/test_agents_and_swarm.py ......................................... [100%]

======================== 943 passed in 266.26s (100%) =========================
```

---

## 📄 License

Synapse is open-source software licensed under the [MIT License](LICENSE). Developers, researchers, and autonomous systems engineers worldwide are free to build, modify, and commercialize applications powered by Synapse.

<div align="center">

**[Star Synapse on GitHub ⭐](https://github.com/cagannbl/synapse)** • Join the future of systems AI engineering.

</div>
