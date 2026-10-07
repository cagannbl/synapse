# X (Twitter) thread draft

Numbers measured on Linux x86_64 with clang (`python benchmarks/run_all.py` reproduces the size). Code blocks are executed by `tests/test_docs_snippets.py`. Re-measure before posting if the code has changed.

---

### 1/6
Shape bugs in tensor code usually show up at runtime, often after a long model load.

I'm building Synapse: a small Python-like language where tensor shapes are part of the type, checked before anything runs.

Open source, MIT. 🧵

---

### 2/6
Dimensions in a signature are rigid. This is rejected before execution:

```python
fn project(x: Tensor[N, D], w: Tensor[K, M]) -> Tensor[N, M]:
    return x @ w
```

`Inner dimensions must match: D != K.` Declared return shapes are checked too.

---

### 3/6
Two engines:

• a bytecode VM (Python + NumPy) with autograd, nn/optim, match/case, Result + `?`, threads and channels
• a C99 backend: `synapse build` emits readable C, compiled by gcc / clang / cl / zig cc

The C backend covers tensor math and plain functions today; nn/optim is VM-only for now.

---

### 4/6
Demo: one transformer block (toy 2×2 weights) compiled to C99.

📦 47.6 KB binary, only libc + libm
⚡ ~1.3 ms from process start to exit
🧠 ~2.2 MB peak RSS

It shows the pipeline, not a usable LLM. CI rebuilds it on Linux and macOS on every push.

---

### 5/6
Also in the box, experimental:

• prompt/agent helpers (OpenAI, Anthropic, Ollama, or an offline mock)
• embedded HTTP + SSE server
• a local playground that runs your code in the real interpreter

---

### 6/6
It's young and a one-person project: 1,043 tests in CI on Linux + macOS, breaking changes likely.

Feedback on the shape checker is what I want most.

⭐ https://github.com/cagannbl/synapse
