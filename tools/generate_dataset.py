#!/usr/bin/env python3
"""
Synapse Synthetic Dataset Generator (Execution-Backed Zero-Error Pipeline)
Generates, executes, validates, and serializes high-quality Synapse instruction datasets
for LLM fine-tuning (Unsloth, Hugging Face SFT, OpenAI Fine-Tuning).
"""

import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
import argparse
from typing import Optional, Tuple, Any

from synapse.lexer.lexer import Lexer, LexerError
from synapse.parser.parser import Parser, ParseError
from synapse.vm.compiler import Compiler
from synapse.vm.virtual_machine import VirtualMachine, VMRuntimeError
from synapse.ai.providers import AutoProvider, MockProvider, BaseLLMProvider


SYSTEM_PROMPT = """You are an expert Synapse programming language AI assistant.
Synapse is an AI-optimized, Python-like language with first-class tensors, built-in autograd, neural network modules (nn & optim), and native AI prompts.
Always write clean, working Synapse (.syn) code with 'let/const' variables and 'fn' functions."""


# =============================================================================
# Execution Verification Engine
# =============================================================================
def verify_synapse_code(code_str: str) -> Tuple[bool, str, Optional[VirtualMachine]]:
    """
    Executes candidate Synapse code through the entire compiler & VM pipeline.
    Guarantees 100% execution-backed accuracy for training samples.
    """
    try:
        tokens = Lexer(code_str).tokenize()
        ast = Parser(tokens).parse()
        compiled = Compiler(name="<synthetic_sample>").compile(ast)
        vm = VirtualMachine()
        vm.execute(compiled)
        return True, "Execution succeeded", vm
    except (LexerError, ParseError) as e:
        return False, f"Syntax error: {e}", None
    except VMRuntimeError as e:
        return False, f"Runtime error: {e}", None
    except Exception as e:
        return False, f"Execution failed: {e}", None


# =============================================================================
# Procedural Seed Tasks & Generators
# =============================================================================
SEED_TEMPLATES = [
    # 1. Tensor Matmul & Stats
    {
        "category": "tensor_math",
        "instruction": "Create two 2x2 matrices A and B, multiply them using matrix multiplication (@), and print the result, transpose, and total sum.",
        "code_fn": lambda: f"""let A = tensor([[1.0, 2.0], [3.0, 4.0]])
let B = tensor([[5.0, 6.0], [7.0, 8.0]])
let C = A @ B
let At = A.T
let total = C.sum()
print("Matrix C:", C)
print("Transpose of A:", At)
print("Sum of C:", total.item())
"""
    },
    # 2. Autograd Scalar Fitting
    {
        "category": "autograd",
        "instruction": "Write a Synapse script that uses reverse-mode automatic differentiation (loss.backward()) to optimize a weight parameter to predict target value 10.0.",
        "code_fn": lambda: f"""let w = tensor({random.uniform(0.1, 0.5):.2f}, requires_grad=true)
let x = tensor(2.0)
let y_target = tensor(10.0)
let lr = 0.05

let epoch = 1
while epoch <= 10:
    let pred = w * x
    let loss = (pred - y_target) ** 2
    loss.backward()
    let updated = w.item() - lr * w.grad.item()
    w = tensor(updated, requires_grad=true)
    epoch += 1

print("Optimized weight:", w.item())
"""
    },
    # 3. Functional Pipeline Operator
    {
        "category": "pipeline",
        "instruction": "Define functions to square numbers and double them, then process a tensor through a functional pipeline operator (|>).",
        "code_fn": lambda: f"""fn square(t):
    return t * t

fn double(t):
    return t * 2.0

let raw_data = tensor([-2.0, -1.0, 0.0, 2.0, 3.0])
let processed = raw_data |> relu |> square |> double
print("Processed result:", processed)
"""
    },
    # 4. Deep Learning Sequential Model
    {
        "category": "deep_learning",
        "instruction": "Construct a multi-layer neural network with Sequential, Linear, and ReLU layers, then train it using Adam and MSELoss.",
        "code_fn": lambda: f"""let model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])

let optimizer = Adam(model.parameters(), lr=0.05)
let criterion = MSELoss()

let X = tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
let Y = tensor([[3.0], [5.0], [7.0]])

let epoch = 1
while epoch <= 8:
    let y_pred = model(X)
    let loss = criterion(y_pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1

print("Final loss:", loss.item())
"""
    },
    # 5. Native AI Prompt Definition
    {
        "category": "ai_prompt",
        "instruction": "Define a native AI prompt to classify customer support inquiries as 'BILLING' or 'TECHNICAL', and call it.",
        "code_fn": lambda: f"""prompt classify_ticket(inquiry: str) -> str:
    system: "You are a customer ticket router. Return BILLING or TECHNICAL."
    user: inquiry
    temperature: 0.1

let ticket = "My invoice was charged twice this month."
let department = classify_ticket(ticket)
print("Assigned department:", department)
"""
    },
    # 6. Autonomous ReAct Agent with Tools
    {
        "category": "ai_agent",
        "instruction": "Define a custom tool and an autonomous agent that assists developers with architecture tasks.",
        "code_fn": lambda: f"""tool analyze_code(repo_name: str):
    return "Repository " + repo_name + " analyzed: 0 lint errors."

agent ArchitectBot:
    model: "gpt-4o"
    tools: [analyze_code]
    instructions: "You analyze codebases and architectural dependencies."

let answer = ArchitectBot.run("Inspect backend repository")
print("Agent reply:", answer)
"""
    },
    # 7. Python Interop (Math FFI)
    {
        "category": "interop",
        "instruction": "Import Python's standard math library using Synapse's FFI bridge and calculate trigonometric values.",
        "code_fn": lambda: f"""import py.math as math

let angle = math.pi / 3.0
let sine_val = math.sin(angle)
let cos_val = math.cos(angle)

print("Angle (pi/3):", angle)
print("Sine:", sine_val)
print("Cosine:", cos_val)
"""
    },
    # 8. Multi-Layer Perceptron with SGD
    {
        "category": "deep_learning",
        "instruction": "Build a binary classification MLP using Linear and Sigmoid layers trained with SGD optimizer.",
        "code_fn": lambda: f"""let net = Sequential([
    Linear(3, 4),
    Sigmoid(),
    Linear(4, 1)
])

let opt = SGD(net.parameters(), lr=0.1)
let criterion = MSELoss()

let inputs = tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
let targets = tensor([[0.0], [1.0]])

let step = 1
while step <= 5:
    let out = net(inputs)
    let loss = criterion(out, targets)
    opt.zero_grad()
    loss.backward()
    opt.step()
    step += 1

print("Training finished, loss:", loss.item())
"""
    },
    # 9. Dynamic Batch Normalization Simulation
    {
        "category": "tensor_math",
        "instruction": "Write a Synapse function that normalizes a 2D tensor by subtracting its mean and dividing by its standard deviation approximation.",
        "code_fn": lambda: f"""fn normalize_tensor(t):
    let mu = t.mean()
    let centered = t - mu
    let variance = (centered ** 2).mean()
    let denom = (variance.item() + 1e-5) ** 0.5
    return centered * (1.0 / denom)

let X = tensor([[10.0, 20.0], [30.0, 40.0]])
let norm_X = normalize_tensor(X)
print("Original X:", X)
print("Normalized X:", norm_X)
"""
    },
    # 10. Iterative Loss Convergence Loop
    {
        "category": "autograd",
        "instruction": "Train a quadratic polynomial curve fit in Synapse using autograd with a decreasing learning rate schedule.",
        "code_fn": lambda: f"""let a = tensor(0.1, requires_grad=true)
let x = tensor(2.0)
let y_target = tensor(4.0)
let lr = 0.04

let iter = 1
while iter <= 6:
    let y_pred = a * (x ** 2)
    let loss = (y_pred - y_target) ** 2
    loss.backward()
    let next_a = a.item() - lr * a.grad.item()
    a = tensor(next_a, requires_grad=true)
    iter += 1

print("Quadratic coefficient a:", a.item())
"""
    },
    # 11. Native Vector Memory & Semantic Recall
    {
        "category": "vector_memory",
        "instruction": "Initialize Synapse's in-memory semantic vector store, remember documentation facts, and query them using similarity search.",
        "code_fn": lambda: f"""let brain = memory()
brain.remember("Synapse has first-class tensors and autograd.")
brain.remember("Neural networks can be built with Sequential and Linear layers.")
brain.remember("Autonomous agents can cooperate using swarms and debates.")

let query = "How to build neural networks in Synapse?"
let matches = brain.recall(query, top_k=1)
print("Top match:", matches[0]["text"])
"""
    },
    # 12. Multi-Agent Swarm Synthesis
    {
        "category": "agent_swarm",
        "instruction": "Create two autonomous agents and coordinate them using the native swarm() primitive to solve a system architecture task.",
        "code_fn": lambda: f"""agent Architect:
    model: "mock-model"
    instructions: "Software architect specializing in compiler design."

agent Optimizer:
    model: "mock-model"
    instructions: "Performance engineer specializing in zero-copy execution."

let task = "Design an AI runtime"
let summary = swarm([Architect, Optimizer], task)
print("Swarm completed.")
"""
    }
]


# =============================================================================
# Synthetic Generation Engine
# =============================================================================
class DatasetGenerator:
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.provider = provider or AutoProvider().resolve_provider()

    def generate_procedural_sample(self) -> dict:
        """Picks a seed template and produces a unique, verified Synapse training sample."""
        template = random.choice(SEED_TEMPLATES)
        instruction = template["instruction"]
        code = template["code_fn"]()

        # Doğrula
        success, msg, _ = verify_synapse_code(code)
        if not success:
            raise RuntimeError(f"Seed template verification failed: {msg}\nCode:\n{code}")

        return {
            "instruction": instruction,
            "code": code,
            "category": template["category"],
            "verified": True
        }

    def generate_llm_sample(self, task_description: str) -> Optional[dict]:
        """Prompts an LLM to generate Synapse code for a task and verifies it."""
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Write complete, working Synapse (.syn) code for the following task:\n"
            f"Task: {task_description}\n\n"
            f"Output ONLY the Synapse code inside a markdown block:\n"
            f"```synapse\n...\n```"
        )
        response = self.provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.2
        )

        # Extract code from markdown block
        code = response
        if "```synapse" in response:
            code = response.split("```synapse")[1].split("```")[0].strip()
        elif "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()

        success, msg, _ = verify_synapse_code(code)
        if not success:
            return None

        return {
            "instruction": task_description,
            "code": code,
            "category": "llm_generated",
            "verified": True
        }

    def build_dataset(
        self,
        count: int = 20,
        mode: str = "procedural",
        format_type: str = "chatml"
    ) -> list[dict]:
        """Generates `count` verified Synapse training samples."""
        samples = []
        attempts = 0
        max_attempts = count * 4

        while len(samples) < count and attempts < max_attempts:
            attempts += 1
            sample = None
            if mode == "procedural":
                sample = self.generate_procedural_sample()
            elif mode == "llm":
                seed = random.choice(SEED_TEMPLATES)
                sample = self.generate_llm_sample(seed["instruction"])

            if sample and sample.get("verified"):
                formatted = self._format_sample(sample, format_type)
                samples.append(formatted)

        return samples

    def _format_sample(self, sample: dict, format_type: str) -> dict:
        instruction = sample["instruction"]
        code = sample["code"].strip()

        if format_type == "chatml":
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": f"```synapse\n{code}\n```"}
                ]
            }
        elif format_type == "alpaca":
            return {
                "instruction": instruction,
                "input": "",
                "output": f"```synapse\n{code}\n```"
            }
        elif format_type == "sharegpt":
            return {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": f"```synapse\n{code}\n```"}
                ]
            }
        else:
            return {
                "instruction": instruction,
                "code": code,
                "category": sample.get("category", "general")
            }


# =============================================================================
# CLI Interface
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Synapse Execution-Backed Synthetic Dataset Generator"
    )
    parser.add_argument(
        "-n", "--count", type=int, default=20, help="Number of verified samples to generate"
    )
    parser.add_argument(
        "-o", "--output", default="data/synapse_instruct_dataset.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "-m", "--mode", choices=["procedural", "llm"], default="procedural", help="Generation mode"
    )
    parser.add_argument(
        "-f", "--format", choices=["chatml", "alpaca", "sharegpt", "raw"], default="chatml", help="Serialization format"
    )
    parser.add_argument(
        "--verify-file", help="Path to an existing .syn file or .jsonl dataset to verify"
    )

    args = parser.parse_args()

    # Tek bir dosyayı doğrulama modu
    if args.verify_file:
        if args.verify_file.endswith(".syn"):
            with open(args.verify_file, "r", encoding="utf-8") as f:
                content = f.read()
            ok, msg, _ = verify_synapse_code(content)
            status = "PASSED" if ok else "FAILED"
            print(f"Verification of {args.verify_file}: [{status}] - {msg}")
            sys.exit(0 if ok else 1)
        elif args.verify_file.endswith(".jsonl"):
            passed, total = 0, 0
            with open(args.verify_file, "r", encoding="utf-8") as f:
                for line in f:
                    total += 1
                    data = json.loads(line)
                    code = ""
                    if "messages" in data:
                        code = data["messages"][-1]["content"]
                    elif "output" in data:
                        code = data["output"]
                    elif "code" in data:
                        code = data["code"]
                    if "```synapse" in code:
                        code = code.split("```synapse")[1].split("```")[0].strip()
                    elif "```" in code:
                        code = code.split("```")[1].split("```")[0].strip()
                    ok, _, _ = verify_synapse_code(code)
                    if ok:
                        passed += 1
            print(f"Dataset verification: {passed}/{total} samples passed ({(passed/total)*100:.1f}%)")
            sys.exit(0 if passed == total else 1)

    # Veri seti oluşturma modu
    print(f"=== Synapse Execution-Backed Dataset Generator ===")
    print(f"Generating {args.count} samples (Mode: {args.mode}, Format: {args.format})...")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    generator = DatasetGenerator()
    dataset = generator.build_dataset(count=args.count, mode=args.mode, format_type=args.format)

    with open(args.output, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Success! {len(dataset)} verified samples written to: {args.output}")


if __name__ == "__main__":
    main()
