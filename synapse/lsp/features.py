"""
Synapse Language Server Protocol (LSP) Features.
Provides rich Markdown documentation for hover tooltips and completions.
"""
from typing import Optional, Any


# =============================================================================
# Hover Documentation (Rich Markdown)
# =============================================================================

HOVER_DOCS: dict[str, str] = {
    # Keywords
    "let": """### `let` Variable Declaration
Declares a mutable variable within the current scope.

```synapse
let count = 0
count = count + 1
```
""",
    "const": """### `const` Constant Declaration
Declares an immutable constant. Once initialized, its value cannot be reassigned.

```synapse
const LEARNING_RATE = 0.001
const BATCH_SIZE = 32
```
""",
    "fn": """### `fn` Function Declaration
Defines a callable function in Synapse. Functions support arguments and return values.

```synapse
fn add(a, b):
    return a + b
```
""",
    "return": """### `return` Statement
Returns a value from a function call to the caller.

```synapse
fn square(x):
    return x * x
```
""",
    "if": """### `if` Conditional Statement
Executes a code block if the conditional expression evaluates to truthy.

```synapse
if x > 0:
    print("Positive")
elif x < 0:
    print("Negative")
else:
    print("Zero")
```
""",
    "elif": """### `elif` Conditional Statement
Else-if branch in conditional statements.
""",
    "else": """### `else` Conditional Statement
Fallback branch when preceding conditional checks fail.
""",
    "while": """### `while` Loop
Repeatedly executes a block while the condition evaluates to true.

```synapse
let i = 0
while i < 10:
    i = i + 1
```
""",
    "for": """### `for` Iteration Loop
Iterates over elements in a sequence, collection, or range.

```synapse
for x in range(0, 5):
    print(x)
```
""",
    "in": """### `in` Membership & Iteration Operator
Used in `for` loops to iterate sequences or check membership.
""",
    "pass": """### `pass` Statement
A null operation. Used as a placeholder where syntax requires a statement.
""",
    "break": """### `break` Statement
Terminates the innermost enclosing loop (`for` or `while`).
""",
    "continue": """### `continue` Statement
Skips the remainder of the current loop iteration and proceeds to the next.
""",
    "import": """### `import` Statement
Imports modules or functions from Synapse standard library, installed packages, or local files.

```synapse
import math
from synapse.core.tensor import tensor
```
""",
    "from": """### `from` Import Statement
Specifies the module from which items are imported.
""",
    "as": """### `as` Alias Keyword
Renames imported modules or identifiers to an alias.
""",

    # AI-Native Keywords
    "tensor": """### `tensor(data, requires_grad=False) -> Tensor`
Creates an AI-native n-dimensional tensor with automatic differentiation (autograd) support.

- **Parameters**:
  - `data`: List, nested list, number, or numpy array.
  - `requires_grad`: Boolean enabling gradient tracking for backpropagation.

```synapse
let w = tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
let x = tensor([[0.5], [1.5]])
let y = w @ x
```
""",
    "grad": """### `grad(fn) -> Callable` / Automatic Differentiation
Autograd gradient operator. Computes analytical gradients with respect to parameters.

```synapse
let w = tensor([2.0], requires_grad=True)
let loss = w * w + 3.0
loss.backward()
print(w.grad)  // tensor([4.0])
```
""",
    "prompt": """### `prompt Name(parameters):` Block
Defines a structured, template-driven AI Prompt.

- Integrates system instructions and dynamic user templates.
- Supports variables interpolated via `{param}` syntax.

```synapse
prompt CodeReviewer(code, lang):
    system: "You are a senior compiler architect."
    user: "Review this {lang} code:\\n{code}"
    temperature: 0.2

let review = CodeReviewer(my_code, "synapse")
```
""",
    "agent": """### `agent Name:` Block
Defines an autonomous AI Agent with system persona, model configuration, and tools.

```synapse
agent CoderAgent:
    model: "gpt-4o"
    system: "You write optimal Synapse code."
    temperature: 0.1
```
""",
    "tool": """### `tool` Keyword
Registers a function as an executable tool for AI Agents.
""",
    "memory": """### `memory(dim=128) -> VectorMemory`
Instantiates a vector-based semantic memory store for AI agents and RAG pipelines.

- **Methods**:
  - `mem.add(text, metadata)`: Stores embedding in vector space.
  - `mem.query(text, top_k=3)`: Semantic similarity vector search.

```synapse
let mem = memory()
mem.add("Synapse is an AI-native language.")
let results = mem.query("What is Synapse?")
```
""",
    "VectorMemory": """### `VectorMemory` Class
Semantic memory store with vector search indexing and cosine similarity.
""",
    "swarm": """### `swarm(agents, task, max_rounds=5)`
Orchestrates multi-agent swarm collaboration to solve complex goals.
""",
    "debate": """### `debate(agents, topic, rounds=3)`
Initiates a structured multi-agent debate to reach consensus on a topic.
""",

    # Built-in Standard Library
    "read_file": """### `read_file(path: str) -> str`
Reads entire file contents as a UTF-8 string.

```synapse
let content = read_file("./config.json")
```
""",
    "write_file": """### `write_file(path: str, content: str) -> int`
Writes text content to a file (UTF-8). Creates or overwrites the file. Returns written characters.

```synapse
write_file("./output.txt", "Execution successful.")
```
""",
    "append_file": """### `append_file(path: str, content: str) -> int`
Appends text content to an existing file (UTF-8). Returns number of characters written.
""",
    "file_exists": """### `file_exists(path: str) -> bool`
Checks whether a file or directory exists at the given path.
""",
    "json_parse": """### `json_parse(text: str) -> Any`
Parses a valid JSON string into Synapse dictionaries, lists, or primitives.

```synapse
let data = json_parse('{"status": "ok", "code": 200}')
print(data["status"])
```
""",
    "json_stringify": """### `json_stringify(obj: Any, indent=2) -> str`
Serializes a Synapse data structure into a formatted JSON string.
""",
    "sleep": """### `sleep(seconds: float) -> None`
Pauses the current execution thread for the specified duration in seconds.
""",
    "env": """### `env(key: str, default: str = "") -> str`
Retrieves the value of an environment variable. Returns `default` if not set.
""",
    "print": """### `print(*args)`
Prints representation of values to standard output.
""",
    "len": """### `len(obj) -> int`
Returns the number of elements in a string, list, or dictionary.
""",
    "range": """### `range(start, stop, step=1) -> range`
Generates an arithmetic progression of integers.
""",
    "zeros": """### `zeros(shape: list[int], requires_grad=False) -> Tensor`
Returns a new tensor filled with zeros according to the specified shape.

```synapse
let z = zeros([3, 3])
```
""",
    "ones": """### `ones(shape: list[int], requires_grad=False) -> Tensor`
Returns a new tensor filled with ones according to the specified shape.

```synapse
let o = ones([2, 4])
```
""",
    "randn": """### `randn(shape: list[int], requires_grad=False) -> Tensor`
Returns a new tensor filled with random samples from a standard normal distribution.
""",
    "relu": """### `relu(x) -> Tensor`
Rectified Linear Unit activation: `max(0, x)`.
""",
    "sigmoid": """### `sigmoid(x) -> Tensor`
Sigmoid activation function: `1 / (1 + exp(-x))`.
""",
    "Linear": """### `Linear(in_features: int, out_features: int) -> Layer`
Fully connected affine transformation layer: `y = x @ W.T + b`.
""",
    "Sequential": """### `Sequential(*layers) -> Model`
Sequentially cascades neural network layers.
""",
    "MSELoss": """### `MSELoss() -> Loss`
Mean Squared Error loss criterion for regression.
""",
    "CrossEntropyLoss": """### `CrossEntropyLoss() -> Loss`
Cross Entropy loss criterion for multi-class classification.
""",
    "Adam": """### `Adam(params, lr=0.001) -> Optimizer`
Adaptive Moment Estimation optimizer for neural network training.
""",
    "SGD": """### `SGD(params, lr=0.01) -> Optimizer`
Stochastic Gradient Descent optimizer.
""",

    # Tensor Methods and Properties
    "T": """### `tensor.T -> Tensor`
Returns the transpose of the tensor (reverses dimensions).

```synapse
let a = tensor([[1, 2], [3, 4]])
print(a.T)
```
""",
    "shape": """### `tensor.shape -> tuple`
Returns the dimensions of the tensor as a tuple of integers.
""",
    "ndim": """### `tensor.ndim -> int`
Returns the number of tensor dimensions (rank).
""",
    "size": """### `tensor.size -> int`
Returns the total count of scalar elements in the tensor.
""",
    "backward": """### `tensor.backward(gradient=None) -> None`
Computes the backward pass using autograd. Propagates gradients through computation graph.
""",
    "zero_grad": """### `tensor.zero_grad() -> None`
Resets the accumulated gradient (`tensor.grad`) to None.
""",
    "item": """### `tensor.item() -> float`
Extracts a standard scalar float from a single-element tensor.
""",
    "tolist": """### `tensor.tolist() -> list`
Converts tensor array data into a nested Python/Synapse list.
""",
    "sum": """### `tensor.sum(axis=None, keepdims=False) -> Tensor`
Computes the sum of all elements or along the specified axis.
""",
    "mean": """### `tensor.mean(axis=None, keepdims=False) -> Tensor`
Computes the arithmetic mean of tensor elements.
""",
}


# =============================================================================
# LSP Completion Items
# =============================================================================

# LSP CompletionItemKind constants
KIND_TEXT = 1
KIND_METHOD = 2
KIND_FUNCTION = 3
KIND_CONSTRUCTOR = 4
KIND_FIELD = 5
KIND_VARIABLE = 6
KIND_CLASS = 7
KIND_INTERFACE = 8
KIND_MODULE = 9
KIND_PROPERTY = 10
KIND_UNIT = 11
KIND_VALUE = 12
KIND_ENUM = 13
KIND_KEYWORD = 14
KIND_SNIPPET = 15


KEYWORDS_COMPLETIONS: list[dict[str, Any]] = [
    {"label": "let", "kind": KIND_KEYWORD, "detail": "Mutable variable", "insertText": "let "},
    {"label": "const", "kind": KIND_KEYWORD, "detail": "Immutable constant", "insertText": "const "},
    {"label": "fn", "kind": KIND_KEYWORD, "detail": "Function definition", "insertText": "fn ${1:name}(${2:params}):\n    ${0:pass}"},
    {"label": "return", "kind": KIND_KEYWORD, "detail": "Return statement", "insertText": "return "},
    {"label": "if", "kind": KIND_KEYWORD, "detail": "Conditional if", "insertText": "if ${1:condition}:\n    ${0:pass}"},
    {"label": "elif", "kind": KIND_KEYWORD, "detail": "Conditional elif", "insertText": "elif ${1:condition}:\n    ${0:pass}"},
    {"label": "else", "kind": KIND_KEYWORD, "detail": "Conditional else", "insertText": "else:\n    ${0:pass}"},
    {"label": "while", "kind": KIND_KEYWORD, "detail": "While loop", "insertText": "while ${1:condition}:\n    ${0:pass}"},
    {"label": "for", "kind": KIND_KEYWORD, "detail": "For in loop", "insertText": "for ${1:item} in ${2:iterable}:\n    ${0:pass}"},
    {"label": "in", "kind": KIND_KEYWORD, "detail": "In operator", "insertText": "in "},
    {"label": "pass", "kind": KIND_KEYWORD, "detail": "Pass statement", "insertText": "pass"},
    {"label": "break", "kind": KIND_KEYWORD, "detail": "Break loop", "insertText": "break"},
    {"label": "continue", "kind": KIND_KEYWORD, "detail": "Continue loop", "insertText": "continue"},
    {"label": "import", "kind": KIND_KEYWORD, "detail": "Import module", "insertText": "import "},
    {"label": "from", "kind": KIND_KEYWORD, "detail": "From module import", "insertText": "from "},
    {"label": "as", "kind": KIND_KEYWORD, "detail": "Import alias", "insertText": "as "},
    {"label": "and", "kind": KIND_KEYWORD, "detail": "Logical AND", "insertText": "and "},
    {"label": "or", "kind": KIND_KEYWORD, "detail": "Logical OR", "insertText": "or "},
    {"label": "not", "kind": KIND_KEYWORD, "detail": "Logical NOT", "insertText": "not "},
    {"label": "true", "kind": KIND_KEYWORD, "detail": "Boolean True", "insertText": "true"},
    {"label": "false", "kind": KIND_KEYWORD, "detail": "Boolean False", "insertText": "false"},
    {"label": "none", "kind": KIND_KEYWORD, "detail": "Null / None", "insertText": "none"},
    # AI Keywords
    {"label": "tensor", "kind": KIND_KEYWORD, "detail": "AI Tensor Factory", "insertText": "tensor(${1:data})"},
    {"label": "prompt", "kind": KIND_KEYWORD, "detail": "AI Prompt block", "insertText": "prompt ${1:Name}(${2:params}):\n    system: \"${3:You are an AI assistant.}\"\n    user: \"${0:{query}}\""},
    {"label": "agent", "kind": KIND_KEYWORD, "detail": "AI Agent block", "insertText": "agent ${1:Name}:\n    model: \"${2:gpt-4o}\"\n    system: \"${0:You are an AI agent.}\""},
    {"label": "tool", "kind": KIND_KEYWORD, "detail": "AI Tool declaration", "insertText": "tool "},
    {"label": "grad", "kind": KIND_KEYWORD, "detail": "Autograd operator", "insertText": "grad(${1:fn})"},
]

BUILTINS_COMPLETIONS: list[dict[str, Any]] = [
    {"label": "read_file", "kind": KIND_FUNCTION, "detail": "read_file(path: str) -> str", "insertText": "read_file(${1:path})"},
    {"label": "write_file", "kind": KIND_FUNCTION, "detail": "write_file(path: str, content: str) -> int", "insertText": "write_file(${1:path}, ${2:content})"},
    {"label": "append_file", "kind": KIND_FUNCTION, "detail": "append_file(path: str, content: str) -> int", "insertText": "append_file(${1:path}, ${2:content})"},
    {"label": "file_exists", "kind": KIND_FUNCTION, "detail": "file_exists(path: str) -> bool", "insertText": "file_exists(${1:path})"},
    {"label": "json_parse", "kind": KIND_FUNCTION, "detail": "json_parse(text: str) -> Any", "insertText": "json_parse(${1:text})"},
    {"label": "json_stringify", "kind": KIND_FUNCTION, "detail": "json_stringify(obj: Any) -> str", "insertText": "json_stringify(${1:obj})"},
    {"label": "sleep", "kind": KIND_FUNCTION, "detail": "sleep(seconds: float) -> None", "insertText": "sleep(${1:seconds})"},
    {"label": "env", "kind": KIND_FUNCTION, "detail": "env(key: str, default: str = '') -> str", "insertText": "env(\"${1:KEY}\")"},
    {"label": "print", "kind": KIND_FUNCTION, "detail": "print(*args) -> None", "insertText": "print(${1:val})"},
    {"label": "len", "kind": KIND_FUNCTION, "detail": "len(obj) -> int", "insertText": "len(${1:obj})"},
    {"label": "range", "kind": KIND_FUNCTION, "detail": "range(start, stop, step=1)", "insertText": "range(${1:start}, ${2:stop})"},
    {"label": "zeros", "kind": KIND_FUNCTION, "detail": "zeros(shape, requires_grad=False)", "insertText": "zeros([${1:shape}])"},
    {"label": "ones", "kind": KIND_FUNCTION, "detail": "ones(shape, requires_grad=False)", "insertText": "ones([${1:shape}])"},
    {"label": "randn", "kind": KIND_FUNCTION, "detail": "randn(shape, requires_grad=False)", "insertText": "randn([${1:shape}])"},
    {"label": "relu", "kind": KIND_FUNCTION, "detail": "relu(x: Tensor) -> Tensor", "insertText": "relu(${1:x})"},
    {"label": "sigmoid", "kind": KIND_FUNCTION, "detail": "sigmoid(x: Tensor) -> Tensor", "insertText": "sigmoid(${1:x})"},
    {"label": "memory", "kind": KIND_FUNCTION, "detail": "memory(dim=128) -> VectorMemory", "insertText": "memory()"},
    {"label": "VectorMemory", "kind": KIND_CLASS, "detail": "VectorMemory semantic index", "insertText": "VectorMemory()"},
    {"label": "swarm", "kind": KIND_FUNCTION, "detail": "swarm(agents, task) -> Any", "insertText": "swarm([${1:agents}], \"${2:task}\")"},
    {"label": "debate", "kind": KIND_FUNCTION, "detail": "debate(agents, topic) -> Any", "insertText": "debate([${1:agents}], \"${2:topic}\")"},
    {"label": "Linear", "kind": KIND_CLASS, "detail": "Linear(in_features, out_features)", "insertText": "Linear(${1:in_features}, ${2:out_features})"},
    {"label": "Sequential", "kind": KIND_CLASS, "detail": "Sequential(*layers)", "insertText": "Sequential(${1:layers})"},
    {"label": "MSELoss", "kind": KIND_CLASS, "detail": "MSELoss()", "insertText": "MSELoss()"},
    {"label": "CrossEntropyLoss", "kind": KIND_CLASS, "detail": "CrossEntropyLoss()", "insertText": "CrossEntropyLoss()"},
    {"label": "Adam", "kind": KIND_CLASS, "detail": "Adam(params, lr=0.001)", "insertText": "Adam(${1:model.parameters()}, lr=${2:0.001})"},
    {"label": "SGD", "kind": KIND_CLASS, "detail": "SGD(params, lr=0.01)", "insertText": "SGD(${1:model.parameters()}, lr=${2:0.01})"},
]

TENSOR_METHODS_COMPLETIONS: list[dict[str, Any]] = [
    {"label": "T", "kind": KIND_PROPERTY, "detail": "Tensor transpose", "documentation": "Returns transposed view of tensor."},
    {"label": "shape", "kind": KIND_PROPERTY, "detail": "tuple of dimensions", "documentation": "Returns tensor shape dimensions."},
    {"label": "ndim", "kind": KIND_PROPERTY, "detail": "int rank", "documentation": "Returns number of dimensions."},
    {"label": "size", "kind": KIND_PROPERTY, "detail": "int element count", "documentation": "Returns total count of scalar elements."},
    {"label": "data", "kind": KIND_PROPERTY, "detail": "numpy array data", "documentation": "Underlying ndarray buffer."},
    {"label": "grad", "kind": KIND_PROPERTY, "detail": "Tensor gradient", "documentation": "Accumulated gradient after backward pass."},
    {"label": "requires_grad", "kind": KIND_PROPERTY, "detail": "bool gradient flag", "documentation": "True if tensor requires autograd backprop."},
    {"label": "backward", "kind": KIND_METHOD, "detail": "backward(gradient=None)", "insertText": "backward()"},
    {"label": "zero_grad", "kind": KIND_METHOD, "detail": "zero_grad()", "insertText": "zero_grad()"},
    {"label": "item", "kind": KIND_METHOD, "detail": "item() -> float", "insertText": "item()"},
    {"label": "tolist", "kind": KIND_METHOD, "detail": "tolist() -> list", "insertText": "tolist()"},
    {"label": "sum", "kind": KIND_METHOD, "detail": "sum(axis=None, keepdims=False)", "insertText": "sum(${1:axis})"},
    {"label": "mean", "kind": KIND_METHOD, "detail": "mean(axis=None, keepdims=False)", "insertText": "mean(${1:axis})"},
    {"label": "relu", "kind": KIND_METHOD, "detail": "relu() -> Tensor", "insertText": "relu()"},
    {"label": "sigmoid", "kind": KIND_METHOD, "detail": "sigmoid() -> Tensor", "insertText": "sigmoid()"},
]


# =============================================================================
# Helper Queries
# =============================================================================

def get_hover(word: str) -> Optional[dict[str, Any]]:
    """
    Returns LSP Hover object for given word if documented.
    """
    if not word:
        return None
    cleaned = word.strip()
    if cleaned in HOVER_DOCS:
        return {
            "contents": {
                "kind": "markdown",
                "value": HOVER_DOCS[cleaned]
            }
        }
    return None


def get_completions(
    trigger_char: Optional[str] = None,
    line_prefix: str = ""
) -> list[dict[str, Any]]:
    """
    Returns list of LSP CompletionItems based on current trigger character or prefix.
    """
    stripped = line_prefix.rstrip()

    # Triggered by dot (.) -> tensor, object methods, or Python module stubs
    if trigger_char == "." or stripped.endswith("."):
        tokens = stripped.rstrip(".").split()
        if tokens:
            target_expr = tokens[-1]
            try:
                from synapse.interop.stubs import DynamicStubGenerator
                mod_target = target_expr[3:] if target_expr.startswith("py.") else target_expr
                py_stubs = DynamicStubGenerator.get_completions_for_lsp(mod_target)
                if py_stubs:
                    return py_stubs
            except Exception:
                pass
        return list(TENSOR_METHODS_COMPLETIONS)

    # Triggered by @ -> Matrix multiplication or decorators
    if trigger_char == "@" or stripped.endswith("@"):
        return [
            {"label": "T", "kind": KIND_PROPERTY, "detail": "Transpose for matrix multiplication", "insertText": "T"},
            {"label": "w @ x", "kind": KIND_SNIPPET, "detail": "Tensor Matmul snippet", "insertText": " @ "},
        ]

    # Triggered by : -> Block headers or type annotations
    if trigger_char == ":" or stripped.endswith(":"):
        return [
            {"label": "pass", "kind": KIND_KEYWORD, "detail": "Empty block body", "insertText": "\n    pass"},
            {"label": "return", "kind": KIND_KEYWORD, "detail": "Return statement", "insertText": "\n    return "},
        ]

    # Default: Return combined keywords, builtins, and general symbols
    return KEYWORDS_COMPLETIONS + BUILTINS_COMPLETIONS + TENSOR_METHODS_COMPLETIONS


# =============================================================================
# Advanced Language Features (Definition, Signature, Symbols, Format)
# =============================================================================

import re

def find_definition(text: str, word: str) -> Optional[dict[str, int]]:
    """
    Finds the definition line and character of a symbol in the given source text.
    Matches 'fn <name>', 'let <name>', 'const <name>', 'agent <name>', 'prompt <name>', 'enum <name>'.
    """
    if not word or not word.strip():
        return None
    target = word.strip()
    patterns = [
        rf"\bfn\s+{re.escape(target)}\b",
        rf"\blet\s+{re.escape(target)}\b",
        rf"\bconst\s+{re.escape(target)}\b",
        rf"\bagent\s+{re.escape(target)}\b",
        rf"\bprompt\s+{re.escape(target)}\b",
        rf"\benum\s+{re.escape(target)}\b",
        rf"\bclass\s+{re.escape(target)}\b",
    ]
    for line_idx, line in enumerate(text.splitlines()):
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                start_col = line.find(target, m.start())
                return {"line": line_idx, "character": max(0, start_col)}
    return None


def get_document_symbols(text: str) -> list[dict[str, Any]]:
    """
    Extracts all symbols (functions, variables, agents, enums) for LSP DocumentSymbol provider.
    SymbolKind: Function=12, Variable=13, Class/Agent=5, Enum=10
    """
    symbols: list[dict[str, Any]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m_fn = re.match(r"^\s*fn\s+([A-Za-z0-9_]+)\s*\((.*?)\)", line)
        if m_fn:
            name = m_fn.group(1)
            col = line.find(name)
            symbols.append({
                "name": name,
                "kind": 12,
                "detail": f"fn {name}({m_fn.group(2)})",
                "range": {"start": {"line": i, "character": 0}, "end": {"line": i, "character": len(line)}},
                "selectionRange": {"start": {"line": i, "character": col}, "end": {"line": i, "character": col + len(name)}}
            })
            continue

        m_ag = re.match(r"^\s*agent\s+([A-Za-z0-9_]+)", line)
        if m_ag:
            name = m_ag.group(1)
            col = line.find(name)
            symbols.append({
                "name": name,
                "kind": 5,
                "detail": f"agent {name}",
                "range": {"start": {"line": i, "character": 0}, "end": {"line": i, "character": len(line)}},
                "selectionRange": {"start": {"line": i, "character": col}, "end": {"line": i, "character": col + len(name)}}
            })
            continue

        m_let = re.match(r"^\s*(?:let|const)\s+([A-Za-z0-9_]+)\s*=", line)
        if m_let:
            name = m_let.group(1)
            col = line.find(name)
            symbols.append({
                "name": name,
                "kind": 13,
                "detail": name,
                "range": {"start": {"line": i, "character": 0}, "end": {"line": i, "character": len(line)}},
                "selectionRange": {"start": {"line": i, "character": col}, "end": {"line": i, "character": col + len(name)}}
            })
    return symbols


def get_signature_help(text: str, line_no: int, col_no: int) -> Optional[dict[str, Any]]:
    """
    Returns LSP SignatureHelp for the function being called at line_no, col_no.
    """
    lines = text.splitlines()
    if line_no < 0 or line_no >= len(lines):
        return None
    curr_line = lines[line_no][:col_no]
    open_paren = curr_line.rfind("(")
    if open_paren == -1:
        return None
    prefix = curr_line[:open_paren].rstrip()
    fn_name_match = re.search(r"([A-Za-z0-9_]+)$", prefix)
    if not fn_name_match:
        return None
    fn_name = fn_name_match.group(1)
    arg_part = curr_line[open_paren + 1:]
    param_idx = arg_part.count(",")

    for l in lines:
        m = re.match(rf"^\s*fn\s+{re.escape(fn_name)}\s*\((.*?)\)", l)
        if m:
            params_str = m.group(1)
            param_names = [p.strip() for p in params_str.split(",") if p.strip()]
            return {
                "signatures": [{
                    "label": f"{fn_name}({params_str})",
                    "documentation": f"Synapse user-defined function `{fn_name}`",
                    "parameters": [{"label": p} for p in param_names]
                }],
                "activeSignature": 0,
                "activeParameter": min(param_idx, max(0, len(param_names) - 1))
            }
    return None


def format_document(text: str) -> str:
    """
    Formats the document using fix_ai_drift and clean 4-space indentation.
    """
    from synapse.core.diagnostics import fix_ai_drift
    cleaned, _, _ = fix_ai_drift(text)
    return cleaned
