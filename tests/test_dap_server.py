"""
Comprehensive Unit Tests for Synapse DAP (Debug Adapter Protocol) Server.
Tests message framing, lifecycle, capabilities, breakpoint hits, inspection, and stepping.
"""
from __future__ import annotations

import io
import json
import time
import pytest

from synapse.tools.dap_server import (
    DAPMessage,
    SynapseDAPServer,
    SynapseDebugSession,
    SessionState,
    StepMode,
    run_dap_server,
)


def _read_all_messages_from_stream(writer_stream: io.BytesIO) -> list[dict]:
    """Helper to parse all framed DAP messages from a BytesIO stream."""
    writer_stream.seek(0)
    data = writer_stream.read()
    messages: list[dict] = []
    offset = 0

    while offset < len(data):
        msg, consumed = DAPMessage.decode(data[offset:])
        if msg is not None and consumed > 0:
            messages.append(msg)
            offset += consumed
        else:
            break
    return messages


# =============================================================================
# Test Group 1: Wire Protocol Framing & Stream I/O
# =============================================================================

def test_dap_message_encode_decode():
    """Tests DAP wire protocol encoding and decoding with Content-Length framing."""
    sample = {
        "seq": 1,
        "type": "request",
        "command": "initialize",
        "arguments": {"clientID": "vscode", "adapterID": "synapse"}
    }
    encoded = DAPMessage.encode(sample)
    assert encoded.startswith(b"Content-Length:")
    assert b"\r\n\r\n" in encoded

    decoded, consumed = DAPMessage.decode(encoded)
    assert decoded is not None
    assert consumed == len(encoded)
    assert decoded["seq"] == 1
    assert decoded["command"] == "initialize"
    assert decoded["arguments"]["adapterID"] == "synapse"


def test_dap_message_stream_read_write():
    """Tests streaming read and write through BytesIO."""
    stream = io.BytesIO()
    payload1 = {"seq": 1, "type": "event", "event": "initialized", "body": {}}
    payload2 = {"seq": 2, "type": "response", "request_seq": 1, "command": "launch", "success": True}

    DAPMessage.write_to_stream(stream, payload1)
    DAPMessage.write_to_stream(stream, payload2)

    stream.seek(0)
    msg1 = DAPMessage.read_from_stream(stream)
    msg2 = DAPMessage.read_from_stream(stream)
    msg3 = DAPMessage.read_from_stream(stream)

    assert msg1 == payload1
    assert msg2 == payload2
    assert msg3 is None


def test_dap_message_malformed_and_partial():
    """Tests decoder resilience to partial and malformed data."""
    # Incomplete header
    decoded, consumed = DAPMessage.decode(b"Content-Length: 50")
    assert decoded is None
    assert consumed == 0

    # Incomplete body
    partial = b"Content-Length: 100\r\n\r\n{\"seq\": 1}"
    decoded, consumed = DAPMessage.decode(partial)
    assert decoded is None
    assert consumed == 0

    # Malformed JSON
    bad_json = b"Content-Length: 10\r\n\r\n{invalid:}"
    decoded, consumed = DAPMessage.decode(bad_json)
    assert decoded is None
    assert consumed > 0


# =============================================================================
# Test Group 2: Initialize & Capabilities Handshake
# =============================================================================

def test_dap_initialize_handshake():
    """Tests initialize request returns valid capabilities and emits initialized event."""
    reader = io.BytesIO()
    writer = io.BytesIO()
    server = SynapseDAPServer(reader=reader, writer=writer)

    req = {
        "seq": 1,
        "type": "request",
        "command": "initialize",
        "arguments": {"adapterID": "synapse", "clientID": "vscode"}
    }
    resp = server.handle_message(req)

    assert resp is not None
    assert resp["type"] == "response"
    assert resp["request_seq"] == 1
    assert resp["command"] == "initialize"
    assert resp["success"] is True
    caps = resp["body"]
    assert caps["supportsConfigurationDoneRequest"] is True
    assert caps["supportsFunctionBreakpoints"] is False
    assert caps["supportsEvaluateForHovers"] is True
    assert server.is_initialized is True

    # Check written messages (response + initialized event)
    messages = _read_all_messages_from_stream(writer)
    assert len(messages) >= 2
    assert messages[0]["type"] == "response"
    assert messages[1]["type"] == "event"
    assert messages[1]["event"] == "initialized"


# =============================================================================
# Test Group 3: Breakpoints Verification
# =============================================================================

def test_dap_set_breakpoints_with_lines():
    """Tests setting line breakpoints via lines array."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())

    req = {
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {
            "source": {"name": "app.syn", "path": "c:/test/app.syn"},
            "lines": [3, 7, 12]
        }
    }
    resp = server.handle_message(req)

    assert resp["success"] is True
    bps = resp["body"]["breakpoints"]
    assert len(bps) == 3
    assert [bp["line"] for bp in bps] == [3, 7, 12]
    assert all(bp["verified"] is True for bp in bps)


def test_dap_set_breakpoints_with_objects():
    """Tests setting breakpoints using breakpoint objects containing conditions."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())

    req = {
        "seq": 3,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {
            "source": {"name": "calc.syn", "path": "c:/test/calc.syn"},
            "breakpoints": [
                {"line": 5, "condition": "x > 10"},
                {"line": 10}
            ]
        }
    }
    resp = server.handle_message(req)

    assert resp["success"] is True
    bps = resp["body"]["breakpoints"]
    assert len(bps) == 2
    assert bps[0]["line"] == 5
    assert bps[1]["line"] == 10


# =============================================================================
# Test Group 4: Program Launch, Breakpoint Hit & StoppedEvent
# =============================================================================

def test_dap_launch_and_breakpoint_hit():
    """Tests launching Synapse code, hitting a line breakpoint, and emitting StoppedEvent."""
    writer = io.BytesIO()
    server = SynapseDAPServer(reader=io.BytesIO(), writer=writer)

    code = (
        "let a = 10\n"
        "let b = 20\n"
        "let c = a + b\n"
        "print(c)\n"
    )

    # 1. Initialize
    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})

    # 2. Set breakpoint at line 3 (let c = a + b)
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {
            "source": {"name": "main.syn", "path": "main.syn"},
            "lines": [3]
        }
    })

    # 3. Launch
    server.handle_message({
        "seq": 3,
        "type": "request",
        "command": "launch",
        "arguments": {
            "program": "main.syn",
            "source": code,
        }
    })

    # 4. ConfigurationDone (begins execution)
    server.handle_message({"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}})

    # Wait for session to hit breakpoint
    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.state == SessionState.STOPPED
    assert server.session.stop_reason == "breakpoint"
    assert server.session.current_line == 3

    # Verify stopped event was written
    messages = _read_all_messages_from_stream(writer)
    stopped_events = [m for m in messages if m.get("type") == "event" and m.get("event") == "stopped"]
    assert len(stopped_events) >= 1
    assert stopped_events[-1]["body"]["reason"] == "breakpoint"
    assert stopped_events[-1]["body"]["threadId"] == 1

    # Cleanup
    server.session.terminate()


# =============================================================================
# Test Group 5: StackTrace, Scopes, and Variables Inspection
# =============================================================================

def test_dap_stack_trace_scopes_variables_inspection():
    """Tests stackTrace, scopes, and variables inspection while paused at a breakpoint."""
    writer = io.BytesIO()
    server = SynapseDAPServer(reader=io.BytesIO(), writer=writer)

    code = (
        "let x = 42\n"
        "let y = 100\n"
        "let sum = x + y\n"
    )

    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {"source": {"name": "test.syn", "path": "test.syn"}, "lines": [3]}
    })
    server.handle_message({
        "seq": 3,
        "type": "request",
        "command": "launch",
        "arguments": {"program": "test.syn", "source": code}
    })
    server.handle_message({"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}})

    assert server.session.wait_for_stop(timeout=4.0) is True

    # 1. stackTrace request
    st_resp = server.handle_message({
        "seq": 5,
        "type": "request",
        "command": "stackTrace",
        "arguments": {"threadId": 1}
    })
    assert st_resp["success"] is True
    frames = st_resp["body"]["stackFrames"]
    assert len(frames) >= 1
    top_frame = frames[0]
    assert top_frame["line"] == 3
    assert top_frame["source"]["name"] == "test.syn"
    frame_id = top_frame["id"]

    # 2. scopes request
    scopes_resp = server.handle_message({
        "seq": 6,
        "type": "request",
        "command": "scopes",
        "arguments": {"frameId": frame_id}
    })
    assert scopes_resp["success"] is True
    scopes = scopes_resp["body"]["scopes"]
    assert len(scopes) == 2
    assert scopes[0]["name"] == "Locals"
    assert scopes[1]["name"] == "Globals"
    locals_ref = scopes[0]["variablesReference"]

    # 3. variables request
    vars_resp = server.handle_message({
        "seq": 7,
        "type": "request",
        "command": "variables",
        "arguments": {"variablesReference": locals_ref}
    })
    assert vars_resp["success"] is True
    variables = {v["name"]: v["value"] for v in vars_resp["body"]["variables"]}
    assert variables.get("x") == "42"
    assert variables.get("y") == "100"

    server.session.terminate()


# =============================================================================
# Test Group 6: Stepping (Next / Step Over) and Continue
# =============================================================================

def test_dap_next_step_over():
    """Tests next (step over) executes line by line and pauses with reason 'step'."""
    writer = io.BytesIO()
    server = SynapseDAPServer(reader=io.BytesIO(), writer=writer)

    code = (
        "let a = 1\n"
        "let b = 2\n"
        "let c = 3\n"
        "let d = 4\n"
    )

    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {"source": {"name": "step.syn", "path": "step.syn"}, "lines": [2]}
    })
    server.handle_message({
        "seq": 3,
        "type": "request",
        "command": "launch",
        "arguments": {"program": "step.syn", "source": code}
    })
    server.handle_message({"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}})

    # First stop at line 2
    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.current_line == 2

    # Step Over (next) -> should reach line 3
    writer.seek(0)
    writer.truncate(0)
    server.handle_message({"seq": 5, "type": "request", "command": "next", "arguments": {"threadId": 1}})

    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.stop_reason == "step"
    assert server.session.current_line == 3

    # Verify locals has 'b' populated after line 2 execution
    vars_resp = server.handle_message({
        "seq": 6,
        "type": "request",
        "command": "variables",
        "arguments": {"variablesReference": server.session.get_scopes(1000)[0]["variablesReference"]}
    })
    var_dict = {v["name"]: v["value"] for v in vars_resp["body"]["variables"]}
    assert var_dict.get("a") == "1"
    assert var_dict.get("b") == "2"

    server.session.terminate()


def test_dap_continue_resumes_to_exit():
    """Tests continue request resumes execution until program finishes cleanly."""
    writer = io.BytesIO()
    server = SynapseDAPServer(reader=io.BytesIO(), writer=writer)

    code = (
        "let a = 5\n"
        "let b = 10\n"
        "print('done')\n"
    )

    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {"source": {"name": "prog.syn", "path": "prog.syn"}, "lines": [2]}
    })
    server.handle_message({
        "seq": 3,
        "type": "request",
        "command": "launch",
        "arguments": {"program": "prog.syn", "source": code}
    })
    server.handle_message({"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}})

    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.current_line == 2

    # Resume to completion
    server.handle_message({"seq": 5, "type": "request", "command": "continue", "arguments": {"threadId": 1}})
    assert server.session.wait_for_termination(timeout=4.0) is True
    assert server.session.state == SessionState.TERMINATED

    messages = _read_all_messages_from_stream(writer)
    events = [m.get("event") for m in messages if m.get("type") == "event"]
    assert "terminated" in events
    assert "exited" in events


# =============================================================================
# Test Group 7: Threads, Evaluation, and Variable Modification
# =============================================================================

def test_dap_threads_request():
    """Tests threads request returns active thread."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())
    resp = server.handle_message({"seq": 1, "type": "request", "command": "threads", "arguments": {}})
    assert resp["success"] is True
    threads = resp["body"]["threads"]
    assert len(threads) >= 1
    assert threads[0]["id"] == 1
    assert threads[0]["name"] == "Main Thread"


def test_dap_evaluate_expression():
    """Tests live expression evaluation inside current frame context."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())
    code = "let x = 15\nlet y = 3\nlet z = x * y\n"

    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {"source": {"name": "test.syn", "path": "test.syn"}, "lines": [3]}
    })
    server.handle_message({
        "seq": 3,
        "type": "request",
        "command": "launch",
        "arguments": {"program": "test.syn", "source": code}
    })
    server.handle_message({"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}})

    assert server.session.wait_for_stop(timeout=4.0) is True

    # Evaluate x + y
    eval_resp1 = server.handle_message({
        "seq": 5,
        "type": "request",
        "command": "evaluate",
        "arguments": {"expression": "x + y", "frameId": 1000}
    })
    assert eval_resp1["success"] is True
    assert eval_resp1["body"]["result"] == "18"
    assert eval_resp1["body"]["type"] == "int"

    # Evaluate x * 2
    eval_resp2 = server.handle_message({
        "seq": 6,
        "type": "request",
        "command": "evaluate",
        "arguments": {"expression": "x * 2", "frameId": 1000}
    })
    assert eval_resp2["body"]["result"] == "30"

    server.session.terminate()


def test_dap_set_variable():
    """Tests modifying variable value dynamically during a debug pause."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())
    code = "let count = 1\nlet limit = 10\n"

    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {"source": {"name": "test.syn", "path": "test.syn"}, "lines": [2]}
    })
    server.handle_message({
        "seq": 3,
        "type": "request",
        "command": "launch",
        "arguments": {"program": "test.syn", "source": code}
    })
    server.handle_message({"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}})

    assert server.session.wait_for_stop(timeout=4.0) is True

    locals_ref = server.session.get_scopes(1000)[0]["variablesReference"]
    set_resp = server.handle_message({
        "seq": 5,
        "type": "request",
        "command": "setVariable",
        "arguments": {"variablesReference": locals_ref, "name": "count", "value": "99"}
    })
    assert set_resp["success"] is True
    assert set_resp["body"]["value"] == "99"

    # Check updated variable
    vars_resp = server.handle_message({
        "seq": 6,
        "type": "request",
        "command": "variables",
        "arguments": {"variablesReference": locals_ref}
    })
    vars_map = {v["name"]: v["value"] for v in vars_resp["body"]["variables"]}
    assert vars_map["count"] == "99"

    server.session.terminate()


# =============================================================================
# Test Group 8: Stop on Entry & Function Calls Stepping
# =============================================================================

def test_dap_stop_on_entry():
    """Tests launch with stopOnEntry=True immediately pauses on the first line."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())
    code = "let x = 1\nlet y = 2\n"

    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "launch",
        "arguments": {"program": "entry.syn", "source": code, "stopOnEntry": True}
    })
    server.handle_message({"seq": 3, "type": "request", "command": "configurationDone", "arguments": {}})

    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.stop_reason == "entry"
    assert server.session.current_line == 1

    server.session.terminate()


def test_dap_step_into_function():
    """Tests stepIn steps into a function body and displays nested stack frame."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())
    code = (
        "fn multiply(p, q):\n"
        "    let m = p * q\n"
        "    return m\n"
        "\n"
        "let res = multiply(3, 4)\n"
    )

    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})
    # Break at function call site (line 5)
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {"source": {"name": "fn.syn", "path": "fn.syn"}, "lines": [5]}
    })
    server.handle_message({
        "seq": 3,
        "type": "request",
        "command": "launch",
        "arguments": {"program": "fn.syn", "source": code}
    })
    server.handle_message({"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}})

    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.current_line == 5

    # Step In -> enters multiply() at line 2
    server.handle_message({"seq": 5, "type": "request", "command": "stepIn", "arguments": {"threadId": 1}})
    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.stop_reason == "step"

    # Verify stack trace shows call frame
    st_resp = server.handle_message({"seq": 6, "type": "request", "command": "stackTrace", "arguments": {"threadId": 1}})
    frames = st_resp["body"]["stackFrames"]
    assert len(frames) >= 2
    assert frames[0]["name"] == "multiply"
    assert frames[1]["name"] in ("fn.syn", "<module>")

    # Verify function arguments p and q in locals
    scopes_resp = server.handle_message({"seq": 7, "type": "request", "command": "scopes", "arguments": {"frameId": frames[0]["id"]}})
    fn_locals_ref = scopes_resp["body"]["scopes"][0]["variablesReference"]
    vars_resp = server.handle_message({"seq": 8, "type": "request", "command": "variables", "arguments": {"variablesReference": fn_locals_ref}})
    var_dict = {v["name"]: v["value"] for v in vars_resp["body"]["variables"]}
    assert var_dict.get("p") == "3"
    assert var_dict.get("q") == "4"

    server.session.terminate()


# =============================================================================
# Test Group 9: Simulated Execution Mode
# =============================================================================

def test_dap_simulated_execution_mode():
    """Tests DAP server operating in simulated stepping mode without real VM."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())

    server.handle_message({"seq": 1, "type": "request", "command": "initialize", "arguments": {}})
    server.handle_message({
        "seq": 2,
        "type": "request",
        "command": "setBreakpoints",
        "arguments": {"source": {"name": "sim.syn", "path": "sim.syn"}, "lines": [2]}
    })
    server.handle_message({
        "seq": 3,
        "type": "request",
        "command": "launch",
        "arguments": {
            "program": "sim.syn",
            "simulate": True,
            "simulated_lines": [1, 2, 3],
            "simulated_locals": {"counter": 100}
        }
    })
    server.handle_message({"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}})

    # Should pause at simulated line 2
    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.current_line == 2
    assert server.session.stop_reason == "breakpoint"

    # Step over to line 3
    server.handle_message({"seq": 5, "type": "request", "command": "next", "arguments": {"threadId": 1}})
    assert server.session.wait_for_stop(timeout=4.0) is True
    assert server.session.current_line == 3
    assert server.session.stop_reason == "step"

    # Verify simulated variables
    vars_resp = server.handle_message({
        "seq": 6,
        "type": "request",
        "command": "variables",
        "arguments": {"variablesReference": 1001}
    })
    var_map = {v["name"]: v["value"] for v in vars_resp["body"]["variables"]}
    assert var_map.get("counter") == "100"

    server.session.terminate()


# =============================================================================
# Test Group 10: Disconnect, Terminate & Unsupported Commands
# =============================================================================

def test_dap_disconnect_and_terminate():
    """Tests disconnect shuts down the server loop and terminates session."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())
    assert server.is_running is True

    resp = server.handle_message({"seq": 1, "type": "request", "command": "disconnect", "arguments": {}})
    assert resp["success"] is True
    assert server.is_running is False
    assert server.session.state == SessionState.TERMINATED


def test_dap_unsupported_command_handling():
    """Tests unsupported request gracefully returns success: False."""
    server = SynapseDAPServer(reader=io.BytesIO(), writer=io.BytesIO())
    resp = server.handle_message({
        "seq": 99,
        "type": "request",
        "command": "unknownFancyFeature",
        "arguments": {}
    })
    assert resp["success"] is False
    assert "Unsupported" in resp["message"]
