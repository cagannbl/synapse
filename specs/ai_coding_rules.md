# Synapse AI Coding & Instruction Guide

This document provides a portable, comprehensive reference for training, fine-tuning, and prompting AI code generation models (ChatGPT, Claude, Gemini, DeepSeek, Qwen) to generate 100% valid Synapse (`.syn`) code.

---

## 1. Syntax Comparison Cheat Sheet (Python vs Synapse)

| Feature | Python | Synapse |
| :--- | :--- | :--- |
| Variable Declaration | `x = 10` | `let x = 10` or `const MAX = 100` |
| Function Definition | `def add(a, b):` | `fn add(a, b):` or `fn add(a: int, b: int) -> int:` |
| Matrix Multiplication | `A @ B` (NumPy/PyTorch) | `A @ B` (Native tensor) |
| Functional Pipeline | `h(g(f(x)))` | `x \|> f \|> g \|> h` |
| Autograd Backward | `loss.backward()` | `loss.backward()` |
| Deep Learning Layer | `torch.nn.Linear(2, 4)` | `Linear(2, 4)` (Built-in) |
| Deep Learning Model | `torch.nn.Sequential(...)` | `Sequential([Linear(2, 4), ReLU()])` |
| Optimizer | `torch.optim.Adam(params, lr=0.01)` | `Adam(model.parameters(), lr=0.01)` |
| Loss Function | `torch.nn.MSELoss()` | `MSELoss()` |
| LLM Prompt Call | `client.chat.completions.create(...)` | `prompt name(arg): ...` |
| AI Agent Definition | `Agent(tools=[...], ...)` (LangChain) | `agent Name: ...` (Language keyword) |
| Python FFI Import | `import math` | `import py.math as math` |

---

## 2. Five Canonical Few-Shot Examples

### Pattern 1: Tensor Matrix Multiplication & Aggregations
```python
# Task: Create two 2x2 matrices, multiply them, calculate transpose, sum, and mean.
let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[5.0, 6.0], [7.0, 8.0]])

let C = A @ B
let At = A.T
let total = C.sum()
let average = C.mean()

print("Matmul Result:", C)
print("Sum:", total.item())
print("Mean:", average.item())
```

### Pattern 2: Reverse-Mode Autograd with Custom Loss
```python
# Task: Train a single scalar weight using gradient descent to fit a target value.
let w = tensor(0.2, requires_grad=true)
let x = tensor(3.0)
let y_target = tensor(9.0)
let lr = 0.02

let step = 1
while step <= 10:
    let y_pred = w * x
    let loss = (y_pred - y_target) ** 2
    loss.backward()

    # Weight update
    let new_weight = w.item() - lr * w.grad.item()
    w = tensor(new_weight, requires_grad=true)
    step += 1

print("Optimized w (Target: 3.0):", w.item())
```

### Pattern 3: Deep Learning MLP Training (`nn` & `optim`)
```python
# Task: Build and train a Multi-Layer Perceptron using Sequential, Adam, and MSELoss.
let model = Sequential([
    Linear(2, 8),
    ReLU(),
    Linear(8, 1)
])

let optimizer = Adam(model.parameters(), lr=0.05)
let criterion = MSELoss()

let X = tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
let Y = tensor([[0.0], [1.0], [1.0], [0.0]])

let epoch = 1
while epoch <= 25:
    let y_hat = model(X)
    let loss = criterion(y_hat, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1

print("Final Training Loss:", loss.item())
```

### Pattern 4: Data Pipeline with Pipeline Operator (`|>`)
```python
# Task: Process raw input tensors sequentially through custom transformations.
fn scale_up(t):
    return t * 10.0

fn clip_positive(t):
    return t.relu()

let raw_signals = tensor([-1.5, 0.2, -0.8, 3.4])
let processed = raw_signals |> clip_positive |> scale_up

print("Processed Signals:", processed)
```

### Pattern 5: AI-Native Prompting and Autonomous Agents
```python
# Task: Define an AI sentiment prompt and an autonomous researcher agent.
prompt evaluate_review(text: str) -> str:
    system: "You are a sentiment analyzer. Return POSITIVE or NEGATIVE only."
    user: text
    temperature: 0.1

tool search_docs(query: str):
    return f"Documentation snippets for {query}"

agent ResearchBot:
    model: "gpt-4o"
    tools: [search_docs]
    instructions: "Research programming language specifications and assist engineers."

let sentiment = evaluate_review("Synapse makes tensor coding ultra clean!")
print("Sentiment:", sentiment)

let reply = ResearchBot.run("Find docs on autograd backward graph")
print("Agent:", reply)
```

---

## 3. Negative Constraints for Model Evaluators
Models generating Synapse code must NOT produce:
1. Python variable assignment without `let` or `const` at initialization.
2. `def func():` (must always be `fn func():`).
3. `import torch`, `import numpy`, `from torch import nn` (never needed).
4. Unclosed string templates or double brackets without intent.
5. In-place autograd modification during backward pass without re-initializing tensor or zeroing grad.
