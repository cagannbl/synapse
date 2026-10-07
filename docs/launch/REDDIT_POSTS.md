# Reddit drafts

Three drafts for **r/programming**, **r/MachineLearning** and **r/LocalLLaMA**. Numbers were measured on Linux x86_64 with clang and Python 3.13; code blocks are executed by `tests/test_docs_snippets.py`. Re-measure before posting if the code has changed.

---

## 1. r/programming

### Title
```text
Synapse: a Python-like tensor language that type-checks shapes and compiles to readable C99
```

### Body

I've been building Synapse, a small language for tensor code. Two parts might interest this sub: a shape checker that treats tensor dimensions as types, and a backend that emits plain C99 instead of LLVM IR.

**Shape checking.** Dimensions in a signature are rigid, so this fails before anything runs:

```python
fn project(x: Tensor[N, D], w: Tensor[K, M]) -> Tensor[N, M]:
    return x @ w    # D != K
```

**C99 output.** `synapse emit-c` produces readable C. For example, this Synapse function:

```python
fn affine(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b
```

becomes:

```c
syn_tensor_t* affine(syn_tensor_t* x, syn_tensor_t* w, syn_tensor_t* b) {
        return syn_add(syn_matmul(x, w), b);
}
```

plus `#line` directives that map C compiler errors back to the `.syn` source. Any C compiler builds it (gcc, clang, cl, zig cc). Tensor memory comes from an arena: allocation bumps an offset, and leaving a scope resets it.

**Numbers** for the bundled demo (one transformer block on toy weights), built with clang on Linux x86_64: a 47.6 KB binary, linked only against libc and libm, ~1.3 ms from process start to exit, ~2.2 MB peak RSS.

**Limits, honestly:** the C backend covers tensor math and plain functions. The `nn`/`optim` modules, agents and the web server only run on the bytecode VM (written in Python) for now. It's a young, one-person project; CI runs 1,043 tests on Linux and macOS.

Why C over LLVM IR: readable output, zero extra toolchain, trivial cross-compiling. The trade-off is depending on the C compiler's optimizer. I'd love to hear where people think that trade-off breaks down.

https://github.com/cagannbl/synapse (MIT)

---

## 2. r/MachineLearning

### Title
```text
[P] Synapse: static tensor shape contracts in a small Python-like language
```

### Body

A matmul shape mismatch typically surfaces at runtime, sometimes after a long checkpoint load:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x128 and 64x256)
```

Synapse makes shapes part of function signatures and checks them before running:

```python
fn attention_scores(q: Tensor[B, S, D], k: Tensor[B, S, D]) -> Tensor[B, S, S]:
    return q @ k.T
```

`synapse verify-shapes` checks matmul and broadcasting against these contracts, including symbolic dimensions (`Tensor[N, D] @ Tensor[K, M]` is rejected) and declared return shapes. For concrete mismatches it suggests a fix such as `A @ B.T`. The check takes ~0.2 ms for a small program.

There's also a reverse-mode autograd engine and the usual building blocks:

```python
let w = tensor([[0.5, -0.2], [1.1, 0.4]], requires_grad=true)
let x = tensor([[2.0], [1.0]])
let y_target = tensor([[1.0], [3.0]])

let y_pred = w @ x
let loss = ((y_pred - y_target) ** 2).sum()
loss.backward()

print("dL/dw:", w.grad)
```

Other pieces: `Linear`/`Sequential`/`Adam`, a DataFrame with `.to_tensor()`, DLPack import/export, and memory-mapped `.safetensors` loading.

**What it isn't:** a replacement for PyTorch or CUDA. Tensors run on NumPy (CPU), and training code runs on a Python bytecode VM. A C99 backend compiles tensor math to small native binaries, but not training code yet.

The part I'd most like feedback on is the shape checker's semantics. Rigid signature dimensions catch real bugs, but they also reject code that would be fine for the shapes it's actually called with. Should `D == K` instead become a constraint checked at call sites?

https://github.com/cagannbl/synapse (MIT, one maintainer, young project)

---

## 3. r/LocalLLaMA

### Title
```text
Synapse: a small scripting language with prompt templates, agents and an SSE server built in (works with Ollama)
```

### Body

Synapse is a Python-like language I'm building. It's mainly about tensor code (shape checking, a C99 backend), but it also has some experimental LLM features built into the language that might be useful for local setups.

Prompts are typed templates, and agents and an embedded vector memory are built-ins:

```python
prompt Summarize(text: str):
    system: "You summarize technical text in one sentence."
    user: "Summarize: " + text

print(Summarize("Synapse checks tensor shapes before running and compiles to C99."))

let brain = memory()
brain.remember("The home server has 16 GB of RAM.")
print(brain.recall("How much memory does the server have?", top_k=1))
```

Provider selection is automatic: OpenAI or Anthropic if their API keys are set, otherwise a local **Ollama** server if one is running (`llama3` by default), otherwise an offline mock so scripts still run.

There's also an embedded HTTP server with Server-Sent Events, so streaming an endpoint doesn't need FastAPI:

```python
fn handle_chat(req):
    let tokens = ["Hello", " from", " Synapse"]
    return sse_response(tokens)

serve(port=8080, routes={"POST /api/chat": handle_chat})
```

To be clear about the limits: these LLM features run on the Python-based VM, not in the small native binaries the C backend produces. Ollama is the only local backend so far; there are no vLLM or llama.cpp adapters yet. It's a young, one-person project, and this part is explicitly experimental.

If you run local models, I'd like to know which backend adapter would be most useful next.

https://github.com/cagannbl/synapse (MIT)
