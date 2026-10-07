<div align="center">

# Synapse

**A Python-like language for tensor code that checks shapes before your program runs, and compiles to small native C99 binaries.**

<p>
  <b>English</b> • <a href="README.tr.md"><b>Türkçe</b></a>
</p>

<p>
  <a href="https://github.com/cagannbl/synapse/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/cagannbl/synapse/tests.yml?branch=main&label=tests" alt="Tests" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License" /></a>
</p>

</div>

```python
fn project(x: Tensor[N, D], w: Tensor[K, M]) -> Tensor[N, M]:
    return x @ w    # D and K are different dimensions: rejected before anything runs
```

```text
$ synapse verify-shapes project.syn
[Shape Guard] FAILED: Found 1 shape violation in 'project.syn':

CompileTimeShapeMismatchError: Cannot multiply tensor of shape ('N', 'D') with tensor of shape ('K', 'M'). Inner dimensions must match: D != K.
  --> Line 2, Column 14
```

> **Status:** Synapse is a young, single-maintainer project. The core language, interpreter, shape checker and C backend are tested on Linux and macOS in CI; other parts are experimental (see [Module stability](#module-stability)). Expect breaking changes.

---

## Quickstart

```bash
git clone https://github.com/cagannbl/synapse.git
cd synapse
pip install -e .            # Python 3.10+, depends only on numpy
```

Or install into `~/.synapse` without touching your Python environment:

```bash
curl -fsSL https://raw.githubusercontent.com/cagannbl/synapse/main/scripts/install.sh | bash
```

A first program, using the pipeline operator (`|>`) and matrix multiplication (`@`):

```python
# pipeline.syn
let A = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=true)
let B = tensor([[0.5, -0.5], [1.5, 0.5]])

let result = A @ B |> sum |> sqrt
print("Scaled Tensor Output:", result)
```

```bash
$ synapse run pipeline.syn
Scaled Tensor Output: tensor(3.4641016151377544, requires_grad=True)
```

Prefer the browser? `python playground/server.py` starts a local playground at `http://127.0.0.1:3000` that runs your code with the real interpreter.

---

## What Synapse does today

| | |
| :--- | :--- |
| **Static shape checking** | `synapse verify-shapes` checks matmul and broadcasting against shapes declared in signatures (`Tensor[B, S, D]`), including symbolic dimensions and return types. The same check runs before `synapse run`. |
| **Two engines** | A bytecode VM (written in Python) for development, and a C99 backend (`synapse emit-c`, `synapse build`) for native binaries. Code the C backend can't handle falls back to the VM. |
| **Tensors & autograd** | N-d tensors on NumPy, reverse-mode autodiff, `Linear` / `Sequential` / `Adam` / `MSELoss` and friends. |
| **Language features** | `match` / `case` with guards, `Result` / `Option` with the `?` operator, `let`, typed functions, structs, threads (`spawn`) and channels. |
| **Tooling** | REPL, formatter, linter, LSP, VS Code extension, DAP debugger, MCP server. |

### Native binaries

[`examples/edge_nanogpt/`](examples/edge_nanogpt/) is one pre-LN transformer block (attention + MLP) written in Synapse and compiled to C99. It uses toy 2×2 weights: it demonstrates the compilation pipeline, not a usable language model.

```bash
python examples/edge_nanogpt/build.py   # needs gcc, clang, cl.exe or zig cc
./examples/edge_nanogpt/nanogpt
```

On Linux x86_64 with clang the binary is **≈48 KB** and depends only on libc and libm. CI builds and runs it on Linux and macOS on every push and reports the measured size.

---

## Benchmarks

```bash
python benchmarks/run_all.py     # add --json for machine-readable output
```

Every number is measured when you run it; nothing is hardcoded. One run on Linux x86_64, Python 3.13:

| Benchmark | Result |
| :--- | :--- |
| Edge NanoGPT native binary | 47.6 KB (clang), runs |
| DataLoader: NumPy slicing (no copying, upper bound) | ~990 M samples/s |
| DataLoader: `SynapseFastDataLoader` | ~6.1 M samples/s |
| DataLoader: `multiprocessing` + pickle queue | ~1.9 M samples/s |
| Shape check of a mismatched matmul | detected in 0.22 ms, suggests `A @ B.T` |

The Synapse loader beats a process-and-pickle pipeline by roughly 3×, but it is far slower than plain NumPy slicing. Results vary a lot between machines.

---

## Language tour

All snippets below are executed by the test suite, so they stay in sync with the language.

### Shape contracts

```python
# Dimensions named in a signature must line up inside the body.
fn attention_scores(q: Tensor[B, S, D], k: Tensor[B, S, D]) -> Tensor[B, S, S]:
    return q @ k.T
```

```bash
$ synapse verify-shapes attention.syn
[Shape Guard] PASS: All tensor shape invariants and dimension contracts verified in 'attention.syn'
```

### Autograd & training

```python
let model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])

let optimizer = Adam(model.parameters(), lr=0.05)
let criterion = MSELoss()

let X = tensor([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]])
let Y = tensor([[3.0], [5.0], [4.0], [6.0]])

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

### Pattern matching & `?`

```python
fn parse_dim(text: str) -> Result[int, str]:
    if text == "":
        return Err("empty dimension")
    return Ok(int(text))

fn parse_shape(rows: str, cols: str) -> Result[list, str]:
    let r = parse_dim(rows)?   # returns the Err early if parsing fails
    let c = parse_dim(cols)?
    return Ok([r, c])

for args in [["3", "4"], ["3", ""]]:
    match parse_shape(args[0], args[1]):
        case Ok(shape) if shape[0] == shape[1]:
            print("Square shape:", shape)
        case Ok(shape):
            print("Shape:", shape)
        case Err(err):
            print("Invalid shape:", err)
```

### Threads & channels

```python
let ch = channel(capacity=4)

fn producer(count):
    for i in range(count):
        ch.send(i * i)
    ch.close()

spawn(producer, 5)
for value in ch:       # blocks until the channel is closed and drained
    print(value)
```

### Prompts, agents and memory (experimental)

Without `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, a built-in mock model answers, so these run offline.

```python
prompt Summarize(text: str):
    system: "You summarize technical text in one sentence."
    user: "Summarize: " + text

print(Summarize("Synapse checks tensor shapes before running and compiles to C99."))

let brain = memory()
brain.remember("Synapse compiles to ISO C99.")
print(brain.recall("How is Synapse compiled?", top_k=1))
```

### HTTP server (experimental)

```python
fn handle_status(req):
    return {
        "status": "active",
        "engine": "Synapse"
    }

fn handle_chat(req):
    # Stream tokens to the client as Server-Sent Events
    let tokens = ["Hello", " from", " Synapse"]
    return sse_response(tokens)

serve(port=8080, routes={
    "GET /api/status": handle_status,
    "POST /api/chat": handle_chat
})
```

More programs live in [`examples/`](examples/); each one is run by the test suite.

---

## Module stability

| Status | Modules |
|---|---|
| **Core** | `synapse/lexer`, `synapse/parser`, `synapse/analyzer`, `synapse/vm`, `synapse/codegen` (C99 backend), `synapse/runtime`, `synapse/core` (tensors & autograd), `synapse/nn`, `synapse/optim`, `synapse/cli.py`, `synapse/lsp` |
| **Experimental** | `synapse/ai` (agents, swarms, LLM providers), `synapse/web`, `synapse/orm`, `packages/`, `synapse/pkg`, `synapse/mcp_server.py`, WASM / CUDA interop |

---

## How it works

```text
source.syn ─▶ lexer ─▶ parser ─▶ AST ─▶ shape checker
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            bytecode compiler + VM                     C99 emitter (emit-c)
            (Python, used by run/repl)            ─▶ gcc / clang / cl / zig cc
                                                   ─▶ native binary (libc + libm)
```

The compiler is written in Python, like TorchInductor, Triton's frontend and Cython. Compiled binaries contain no Python: they are plain C99 linked against libc and libm, with an arena allocator for tensor memory.

---

## CLI

| Command | What it does |
| :--- | :--- |
| `synapse run <file>` | Check shapes, then run (C backend if possible, else the VM; `--vm` forces the VM). |
| `synapse verify-shapes <file>` | Static shape check only (`--json` available). |
| `synapse emit-c <file>` / `synapse build <file>` | Emit C99 source / build a native executable. |
| `synapse check`, `lint`, `fmt`, `fix` | Type checks, lints, formatting, automatic fixes for common LLM-generated mistakes. |
| `synapse repl` | Interactive REPL. |
| `synapse lsp`, `dap`, `mcp` | Language server, debug adapter, Model Context Protocol server. |
| `synapse demo` | Short showcases (`--preset nanogpt`, `matmul`, `tour`, `dataloader`). |

Run `synapse --help` for the full list.

---

## Development

```bash
pip install -e ".[test]"
pytest
```

The suite runs in about a minute and covers the language, both engines, the shape checker, the C runtime, and every code snippet in this README, the examples and the playground. See [CONTRIBUTING.md](CONTRIBUTING.md).

Editor support: TextMate grammar and a VS Code extension in [`editors/vscode/`](editors/vscode/):

```bash
python editors/vscode/build_vsix.py
code --install-extension editors/vscode/synapse-lang-1.0.0.vsix
```

## License

[MIT](LICENSE)
