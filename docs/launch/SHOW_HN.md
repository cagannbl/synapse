# Show HN draft

Every number in this draft was measured on Linux x86_64 (clang, Python 3.13) and can be reproduced with `python benchmarks/run_all.py`. Every Synapse code block is executed by `tests/test_docs_snippets.py`. Re-measure before posting if the code has changed.

**Submission title:**
```text
Show HN: Synapse – a Python-like tensor language that checks shapes before running
```

**Link:** `https://github.com/cagannbl/synapse`

---

## Submission text

Hi HN,

I've been building Synapse, a small Python-like language for tensor code. Its main idea is to treat tensor shapes as part of a function's type and check them before the program runs:

```python
fn project(x: Tensor[N, D], w: Tensor[K, M]) -> Tensor[N, M]:
    return x @ w
```

```text
$ synapse verify-shapes project.syn
[Shape Guard] FAILED: Found 1 shape violation in 'project.syn':

CompileTimeShapeMismatchError: Cannot multiply tensor of shape ('N', 'D') with tensor of shape ('K', 'M'). Inner dimensions must match: D != K.
  --> Line 2, Column 14
```

The same check runs before `synapse run`, so this never reaches a matmul at runtime. Dimensions named in a signature are rigid: `D` and `K` above are different even though nothing binds them to numbers yet, and declared return shapes are enforced too. Concrete mismatches come with a fix suggestion (e.g. `A @ B.T`).

**How it runs**

```text
source.syn → lexer → parser → AST → shape checker ─┬→ bytecode VM (Python)
                                                    └→ C99 emitter → gcc / clang / cl / zig cc
```

- The **VM** runs everything: tensors on NumPy, reverse-mode autograd, `nn`/`optim` (Linear, Sequential, Adam, ...), `match`/`case`, `Result`/`Option` with `?`, threads and channels.
- The **C99 backend** (`synapse build`) currently covers tensor math and plain functions, not yet `nn`/`optim` or the AI helpers. Binaries depend only on libc and libm and use an arena allocator for tensors.

Example: `examples/edge_nanogpt/` is one pre-LN transformer block (attention + MLP) on toy 2×2 weights. Built with clang on Linux x86_64:

| | Native binary | `synapse run --vm` |
| :--- | :--- | :--- |
| Size | 47.6 KB | (Python + NumPy) |
| Process start to exit | ~1.3 ms | ~220 ms for "hello world" |
| Peak RSS | ~2.2 MB | ~40 MB |

It's a demo of the pipeline, not a usable language model. CI builds and runs it on Linux and macOS on every push and reports the size.

**Why emit C instead of LLVM IR?** The output is readable (`synapse emit-c`), any C compiler works, and cross-compiling is a compiler flag away. The cost is that optimizations are left to the C compiler.

**Why is the compiler in Python?** Same reason TorchInductor and Triton's frontend are: iterating on the compiler is fast, and the compiled binaries contain no Python.

**Honest status**

- Young project, one maintainer, v3.0.0 with breaking changes likely.
- 1,043 tests run in CI on Linux and macOS (Python 3.10 and 3.13). Not tested on Windows in CI.
- Experimental: LLM prompt/agent helpers (OpenAI, Anthropic, Ollama, or an offline mock), an embedded HTTP/SSE server, ORM, WASM output (emits C for Emscripten; not tested end to end).
- A local playground (`python playground/server.py`) runs code in the real interpreter.

I'd especially like feedback on:
1. Rigid vs. inferred dimensions: should `Tensor[N, D] @ Tensor[K, M]` be an error, or should `D == K` become a constraint the caller must satisfy?
2. How far a C99 backend can go before LLVM becomes worth it.
3. Which shape errors you hit most often in practice that a checker like this should catch.

Code: https://github.com/cagannbl/synapse (MIT)
