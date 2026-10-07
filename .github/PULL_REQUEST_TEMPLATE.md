## 📌 Pull Request Overview

### Summary of Changes
<!-- Provide a concise explanation of what this PR does and why it was introduced. -->

### Related Issue
<!-- Link the corresponding issue here (e.g. Closes #123, Fixes #45). -->
Closes #

---

## 🏷️ Type of Change

- [ ] 🐛 **Bug Fix** (non-breaking change which fixes an issue)
- [ ] ✨ **New Feature** (non-breaking addition of syntax, kernel, or stdlib module)
- [ ] ⚡ **Performance Optimization** (SIMD, OpenMP, Arena memory reduction)
- [ ] 🛡️ **Safety & Shape Guard** (static analysis, dimension verification)
- [ ] 🔧 **Tooling / DevEx** (VS Code, DAP, LSP, MCP server)
- [ ] 📖 **Documentation** (architectural docs, examples, launch kits)
- [ ] 💥 **Breaking Change** (fix or feature that alters existing syntax or ABI)

---

## 🏛️ Subsystems Impacted

- [ ] **Lexer & Parser** (`synapse/lexer/`, `synapse/parser/`)
- [ ] **Shape Guard & Type Checker** (`synapse/analyzer/`)
- [ ] **C99 Emitter & Native Compiler** (`synapse/codegen/`)
- [ ] **Arena Runtime & Memory Allocator** (`synapse/runtime/`)
- [ ] **Autograd & Neural Ops** (`synapse/core/`, `synapse/nn/`, `synapse/optim/`)
- [ ] **Standard Library & Data** (`synapse/stdlib/`, `synapse/data/`)
- [ ] **CLI & Ecosystem Tools** (`synapse/cli.py`, `synapse/tools/`)

---

## 🔬 ISO C99 & Memory Safety Guarantees

If this PR touches `synapse/runtime/` or `synapse/codegen/c_emitter.py`:

- [ ] **Strict ISO C99:** Code builds cleanly with `gcc`, `clang`, `zig cc`, and `cl.exe` under `-Wall -Wextra -pedantic` without warnings.
- [ ] **Zero Unchecked Allocations:** Dynamic heap allocations are either arena-managed (`syn_arena_alloc`) or safely wrapped.
- [ ] **Arena Scope Balance:** Every `syn_arena_scope_enter` has a guaranteed, exception-safe `syn_arena_scope_leave`.
- [ ] **Binary Size Budget:** Standalone binary size remains lean (the CI binary-size job enforces a 512 KB budget for the Edge NanoGPT example).

---

## 🧪 Test Suite & Verification Proof

The full test suite must pass (CI runs it on Linux and macOS).

**Verification Proof:**
```text
# Paste the last line of your local `pytest` run here
```

**New Tests Added:**
- `tests/test_...py`: <!-- Name the test file and functions added -->

---

## ✅ Contributor Checklist

- [ ] My code adheres to the project's coding and style guidelines (`black`, `ruff`).
- [ ] I have verified that all existing tests pass without regressions (`pytest`).
- [ ] I have added thorough tests covering the new functionality or edge case.
- [ ] I have updated the documentation or example files where appropriate.
- [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guide.
