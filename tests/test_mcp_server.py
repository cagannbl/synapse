"""
Tests for Synapse Native MCP Server (Model Context Protocol) Tools
"""

import json
from synapse.mcp_server import create_mcp_server


def get_tool_fn(server, tool_name: str):
    for tool in server._tool_manager.list_tools():
        if tool.name == tool_name:
            return tool.fn
    raise ValueError(f"Tool {tool_name} not found on server")


def test_mcp_server_initialization():
    server = create_mcp_server()
    tools = [t.name for t in server._tool_manager.list_tools()]
    expected = [
        "validate_synapse_code",
        "execute_synapse",
        "fix_synapse_code",
        "transpile_to_c",
        "inspect_tensor_shapes",
        "get_ai_system_prompt"
    ]
    for exp in expected:
        assert exp in tools


def test_mcp_validate_synapse_code():
    server = create_mcp_server()
    validate_fn = get_tool_fn(server, "validate_synapse_code")

    valid_code = "let a = tensor([1.0, 2.0])\nprint(a)\n"
    res_valid = json.loads(validate_fn(valid_code))
    assert res_valid["status"] == "ok"

    invalid_code = "let a = tensor([1.0, 2.0]\n"
    res_invalid = json.loads(validate_fn(invalid_code))
    assert res_invalid["status"] == "error"
    assert "error_type" in res_invalid


def test_mcp_execute_synapse():
    server = create_mcp_server()
    exec_fn = get_tool_fn(server, "execute_synapse")

    code = "let x = 15\nlet y = 25\nprint(x + y)\n"
    res = json.loads(exec_fn(code))
    assert res["status"] == "success"
    assert "40" in res["stdout"]


def test_mcp_fix_synapse_code():
    server = create_mcp_server()
    fix_fn = get_tool_fn(server, "fix_synapse_code")

    drift_code = """import numpy as np
def multiply(a, b):
    result = a @ b
    return result
"""
    res = json.loads(fix_fn(drift_code))
    assert "fn multiply(a, b):" in res["fixed_code"]
    assert "let result = a @ b" in res["fixed_code"]
    assert "import numpy" not in res["fixed_code"]
    assert len(res["changes_applied"]) >= 2
    assert "diff" in res and "--- a/" in res["diff"]


def test_mcp_transpile_to_c():
    server = create_mcp_server()
    transpile_fn = get_tool_fn(server, "transpile_to_c")

    code = "let A = tensor([[1.0, 2.0], [3.0, 4.0]])\nlet B = tensor([[5.0, 6.0], [7.0, 8.0]])\nlet C = A @ B\n"
    res = json.loads(transpile_fn(code))
    assert res["status"] == "success"
    assert "int main" in res["c_code"]
    assert "syn_matmul" in res["c_code"]


def test_mcp_inspect_tensor_shapes():
    server = create_mcp_server()
    inspect_fn = get_tool_fn(server, "inspect_tensor_shapes")

    # Mismatch: [2, 3] @ [4, 5] (3 != 4)
    bad_code = "let A = zeros([2, 3])\nlet B = zeros([4, 5])\nlet C = A @ B\n"
    res_bad = json.loads(inspect_fn(bad_code))
    assert res_bad["status"] == "shape_mismatch"
    assert "Cannot multiply tensor of shape" in res_bad["report"]["message"]

    # Match: [2, 4] @ [4, 5] (4 == 4)
    good_code = "let A = zeros([2, 4])\nlet B = zeros([4, 5])\nlet C = A @ B\n"
    res_good = json.loads(inspect_fn(good_code))
    assert res_good["status"] == "ok"


def test_mcp_get_ai_system_prompt():
    server = create_mcp_server()
    prompt_fn = get_tool_fn(server, "get_ai_system_prompt")
    prompt = prompt_fn()
    assert "Synapse" in prompt
    assert "let x = 10" in prompt
    assert "fn add" in prompt
    assert len(prompt.split()) < 300  # Token-efficient guarantee
