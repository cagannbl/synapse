import time
import json
import socket
import struct
import base64
import hashlib
import threading
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import pytest

from synapse.core.web import (
    serve,
    SynapseServer,
    SynapseRequest,
    SynapseResponse,
    SynapseSSEResponse,
    SynapseWebSocket,
    SynapseWebSocketRoute,
    _compute_ws_accept,
    WS_GUID,
)
from synapse.ai.agent_runtime import AgentRuntime
from synapse.ai.agent_swarm import (
    DistributedNode,
    SwarmMesh,
    swarm,
    debate,
)


# =============================================================================
# Helper: Raw WebSocket Client for RFC 6455 Testing
# =============================================================================
class SimpleWSClient:
    def __init__(self, host: str, port: int, path: str = "/ws"):
        self.host = host
        self.port = port
        self.path = path
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.handshake_key = base64.b64encode(b"0123456789abcdef").decode("ascii")

    def connect(self):
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {self.handshake_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        self.sock.sendall(req.encode("ascii"))
        resp = bytearray()
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(1024)
            if not chunk:
                break
            resp.extend(chunk)

        header_text = resp.decode("ascii", errors="replace")
        assert "101 Switching Protocols" in header_text
        assert "Upgrade: websocket" in header_text.lower() or "upgrade: websocket" in header_text.lower()
        expected_accept = _compute_ws_accept(self.handshake_key)
        assert f"Sec-WebSocket-Accept: {expected_accept}" in header_text

    def send_text(self, text: str):
        # Client frames MUST be masked
        payload = text.encode("utf-8")
        mask_key = b"\x1a\x2b\x3c\x4d"
        masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))

        b0 = 0x81  # FIN=1, opcode=1
        length = len(payload)
        if length <= 125:
            header = bytes([b0, 0x80 | length])
        elif length <= 65535:
            header = struct.pack("!BBH", b0, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", b0, 0x80 | 127, length)

        self.sock.sendall(header + mask_key + masked)

    def recv_text(self) -> str:
        # Server frames MUST NOT be masked
        head = self._read_exact(2)
        b0, b1 = head[0], head[1]
        opcode = b0 & 0x0F
        is_masked = bool(b1 & 0x80)
        length = b1 & 0x7F

        if length == 126:
            ext = self._read_exact(2)
            length = struct.unpack("!H", ext)[0]
        elif length == 127:
            ext = self._read_exact(8)
            length = struct.unpack("!Q", ext)[0]

        if is_masked:
            mask_key = self._read_exact(4)
            raw = self._read_exact(length)
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(raw))
        else:
            payload = self._read_exact(length)

        return payload.decode("utf-8")

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionResetError("Socket closed")
            buf.extend(chunk)
        return bytes(buf)

    def close(self):
        try:
            # Send close frame
            mask_key = b"\x00\x00\x00\x00"
            self.sock.sendall(bytes([0x88, 0x80]) + mask_key)
        except Exception:
            pass
        finally:
            self.sock.close()


# =============================================================================
# 1. Web Server Threading & Concurrency Tests
# =============================================================================
def test_threaded_server_concurrency():
    """Verify SynapseServer handles concurrent requests in parallel threads."""
    def slow_handler(req):
        time.sleep(0.3)
        return {"done": True, "path": req.path}

    server = serve(
        port=8111,
        routes={"/slow1": slow_handler, "/slow2": slow_handler},
        blocking=False
    )

    try:
        time.sleep(0.2)
        start_t = time.time()

        def fetch(url):
            with urllib.request.urlopen(url) as r:
                return json.loads(r.read().decode("utf-8"))

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(fetch, "http://127.0.0.1:8111/slow1")
            f2 = ex.submit(fetch, "http://127.0.0.1:8111/slow2")
            r1 = f1.result()
            r2 = f2.result()

        elapsed = time.time() - start_t
        assert r1["done"] is True
        assert r2["done"] is True
        # If sequential, it would take >= 0.6s. In threaded mode, both run concurrently in < 0.55s.
        assert elapsed < 0.55
    finally:
        server.stop()


# =============================================================================
# 2. SSE (Server-Sent Events) Streaming Tests
# =============================================================================
def test_sse_streaming_endpoint():
    """Verify SynapseSSEResponse streams events formatted as text/event-stream."""
    def sse_events_handler(req):
        def event_stream():
            yield "first-event"
            yield {"event": "status", "data": {"step": 2, "msg": "in-progress"}}
            yield {"data": "simple-json-data"}
            yield "data: custom-raw-line\n\n"

        return SynapseSSEResponse(event_stream())

    server = serve(
        port=8112,
        routes={"/api/events": sse_events_handler},
        blocking=False
    )

    try:
        time.sleep(0.2)
        req = urllib.request.Request("http://127.0.0.1:8112/api/events")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type")
            assert resp.headers.get("Cache-Control") == "no-cache"

            raw_stream = resp.read().decode("utf-8")
            assert "data: first-event\n\n" in raw_stream
            assert "event: status\n" in raw_stream
            assert '"step": 2' in raw_stream
            assert "data: simple-json-data\n\n" in raw_stream
            assert "data: custom-raw-line\n\n" in raw_stream
    finally:
        server.stop()


def test_generator_auto_sse_detection():
    """Verify generator returning route handler is automatically treated as SSE stream."""
    def direct_generator_handler(req):
        yield {"count": 1}
        yield {"count": 2}
        yield "completed"

    server = serve(
        port=8113,
        routes={"/api/gen": direct_generator_handler},
        blocking=False
    )

    try:
        time.sleep(0.2)
        with urllib.request.urlopen("http://127.0.0.1:8113/api/gen") as resp:
            assert "text/event-stream" in resp.headers.get("Content-Type")
            content = resp.read().decode("utf-8")
            assert 'data: {"count": 1}\n\n' in content
            assert 'data: {"count": 2}\n\n' in content
            assert "data: completed\n\n" in content
    finally:
        server.stop()


# =============================================================================
# 3. Native RFC 6455 WebSocket Tests
# =============================================================================
def test_websocket_echo_and_json():
    """Verify WebSocket handshake, masked client reception, and unmasked server echo."""
    received_messages = []

    def echo_ws(ws: SynapseWebSocket):
        for msg in ws:
            received_messages.append(msg)
            if msg == "ping":
                ws.send_text("pong")
            elif "hello" in msg:
                ws.send_json({"reply": f"echo:{msg}", "ack": True})
            elif msg == "bye":
                ws.send_text("goodbye")
                ws.close()
                break

    server = SynapseServer(port=8114)
    server.add_ws_route("/ws/echo", echo_ws)
    server.start(blocking=False)

    try:
        time.sleep(0.2)
        client = SimpleWSClient("127.0.0.1", 8114, "/ws/echo")
        client.connect()

        # 1. Send text frame
        client.send_text("ping")
        reply1 = client.recv_text()
        assert reply1 == "pong"

        # 2. Send text expecting JSON reply
        client.send_text("hello-synapse")
        reply2 = client.recv_text()
        data = json.loads(reply2)
        assert data["reply"] == "echo:hello-synapse"
        assert data["ack"] is True

        # 3. Close handshake
        client.send_text("bye")
        reply3 = client.recv_text()
        assert reply3 == "goodbye"

        client.close()
        time.sleep(0.1)
        assert "ping" in received_messages
        assert "hello-synapse" in received_messages
    finally:
        server.stop()


def test_websocket_route_wrapper_in_routes_dict():
    """Verify registering a SynapseWebSocketRoute directly in server routes dict."""
    def chat_ws(ws: SynapseWebSocket):
        msg = ws.recv_text()
        ws.send_text(f"Welcome, {msg}!")
        ws.close()

    server = serve(
        port=8115,
        routes={
            "/ws/chat": SynapseWebSocketRoute(chat_ws),
            "/api/health": lambda req: {"status": "up"}
        },
        blocking=False
    )

    try:
        time.sleep(0.2)
        # HTTP health works
        with urllib.request.urlopen("http://127.0.0.1:8115/api/health") as r:
            assert json.loads(r.read().decode("utf-8"))["status"] == "up"

        # WS endpoint works
        client = SimpleWSClient("127.0.0.1", 8115, "/ws/chat")
        client.connect()
        client.send_text("Alice")
        reply = client.recv_text()
        assert reply == "Welcome, Alice!"
        client.close()
    finally:
        server.stop()


# =============================================================================
# 4. DistributedNode Tests (Local & Remote RPC)
# =============================================================================
def test_distributed_node_local():
    """Verify DistributedNode running in-process wrapping AgentRuntime."""
    agent = AgentRuntime(name="MathAgent", instructions="Does math.")
    node = DistributedNode(
        name="LocalMathNode",
        role="calculator",
        capabilities=["math", "calculation"],
        weight=2.5,
        agent=agent
    )

    assert node.is_local is True
    assert node.has_capability("math") is True
    assert node.has_capability("nlp") is False
    assert node.ping() is True

    result = node.execute("Compute 40 + 2")
    assert len(str(result)) > 0

    info = node.to_dict()
    assert info["role"] == "calculator"
    assert info["weight"] == 2.5
    assert info["is_local"] is True


def test_distributed_node_remote_rpc():
    """Verify DistributedNode making remote HTTP RPC calls to a Synapse node server."""
    def rpc_handler(req):
        body = req.body or {}
        task = body.get("task", "")
        return {"result": f"REMOTE_PROCESSED: {task.upper()}"}

    def status_handler(req):
        return {"status": "healthy"}

    rpc_server = serve(
        port=8116,
        routes={
            "/api/task": rpc_handler,
            "/api/status": status_handler,
        },
        blocking=False
    )

    try:
        time.sleep(0.2)
        remote_node = DistributedNode(
            node_id="worker-node-1",
            name="RemoteWorker",
            host="127.0.0.1",
            port=8116,
            role="compute",
            capabilities=["heavy-compute", "transform"],
            agent=None,
            weight=1.0
        )

        assert remote_node.is_local is False
        assert remote_node.ping() is True
        assert remote_node.has_capability("heavy-compute") is True

        res = remote_node.execute("matrix-transpose")
        assert res == "REMOTE_PROCESSED: MATRIX-TRANSPOSE"

        # run alias
        res2 = remote_node.run("vector-dot")
        assert res2 == "REMOTE_PROCESSED: VECTOR-DOT"
    finally:
        rpc_server.stop()


# =============================================================================
# 5. SwarmMesh & Routing Tests
# =============================================================================
def test_swarm_mesh_registration_and_filtering():
    mesh = SwarmMesh(mesh_id="test_cluster")

    n1 = DistributedNode(node_id="n1", name="N1", capabilities=["code"], role="dev", agent=lambda t: f"Code: {t}")
    n2 = DistributedNode(node_id="n2", name="N2", capabilities=["code", "review"], role="reviewer", agent=lambda t: f"Review: {t}")
    n3 = DistributedNode(node_id="n3", name="N3", capabilities=["math"], role="calc", agent=lambda t: f"Math: {t}")

    mesh.register_node(n1)
    mesh.register_node(n2)
    mesh.register_node(n3)

    assert len(mesh) == 3
    code_nodes = mesh.get_nodes(capability="code")
    assert len(code_nodes) == 2
    math_nodes = mesh.get_nodes(capability="math")
    assert len(math_nodes) == 1
    assert math_nodes[0].name == "N3"

    reviewers = mesh.get_nodes(role="reviewer")
    assert len(reviewers) == 1
    assert reviewers[0].name == "N2"


def test_swarm_mesh_broadcast_and_routing():
    mesh = SwarmMesh()
    mesh.register_node(DistributedNode(node_id="a1", name="WorkerA", capabilities=["nlp"], agent=lambda t: f"A:{t}"))
    mesh.register_node(DistributedNode(node_id="a2", name="WorkerB", capabilities=["nlp"], agent=lambda t: f"B:{t}"))

    # Broadcast to all nlp nodes
    results = mesh.broadcast("Analyze sentiment", capability="nlp")
    assert len(results) == 2
    assert results["a1"] == "A:Analyze sentiment"
    assert results["a2"] == "B:Analyze sentiment"

    # Route task using round-robin
    r1 = mesh.route_task("Task 1", capability="nlp", strategy="round_robin")
    r2 = mesh.route_task("Task 2", capability="nlp", strategy="round_robin")
    assert (r1.startswith("A:") and r2.startswith("B:")) or (r1.startswith("B:") and r2.startswith("A:"))


# =============================================================================
# 6. SwarmMesh Consensus Mechanism Tests
# =============================================================================
def test_swarm_mesh_majority_consensus():
    """Verify majority consensus tallying across nodes."""
    mesh = SwarmMesh()

    # 3 nodes: 2 vote 'APPROVE', 1 votes 'REJECT'
    n1 = DistributedNode(node_id="v1", name="V1", agent=lambda t: "Verdict: APPROVE. Code is clean.")
    n2 = DistributedNode(node_id="v2", name="V2", agent=lambda t: "Verdict: REJECT. Missing tests.")
    n3 = DistributedNode(node_id="v3", name="V3", agent=lambda t: "Verdict: APPROVE. Good to merge.")

    mesh.register_node(n1)
    mesh.register_node(n2)
    mesh.register_node(n3)

    res = mesh.consensus("Evaluate pull request #42", strategy="majority", candidates=["APPROVE", "REJECT"])

    assert res["winner"] == "APPROVE"
    assert res["consensus_reached"] is True
    assert res["votes"]["v1"] == "APPROVE"
    assert res["votes"]["v2"] == "REJECT"
    assert res["votes"]["v3"] == "APPROVE"
    assert res["tally"]["APPROVE"] == 2.0
    assert res["tally"]["REJECT"] == 1.0
    assert res["confidence"] == pytest.approx(2.0 / 3.0, 0.01)


def test_swarm_mesh_weighted_consensus():
    """Verify weighted consensus where a higher weight node can dominate."""
    mesh = SwarmMesh()

    # Lead architect (weight=10.0) votes 'REJECT', two junior devs (weight=1.0 each) vote 'APPROVE'
    lead = DistributedNode(node_id="lead", name="Architect", weight=10.0, agent=lambda t: "REJECT")
    dev1 = DistributedNode(node_id="dev1", name="Dev1", weight=1.0, agent=lambda t: "APPROVE")
    dev2 = DistributedNode(node_id="dev2", name="Dev2", weight=1.0, agent=lambda t: "APPROVE")

    mesh.register_node(lead)
    mesh.register_node(dev1)
    mesh.register_node(dev2)

    # In simple majority: APPROVE would get 2 votes vs 1
    # In weighted: REJECT gets 10.0 vs 2.0 for APPROVE
    weighted_res = mesh.consensus("Approve architecture design", strategy="weighted", candidates=["APPROVE", "REJECT"])

    assert weighted_res["winner"] == "REJECT"
    assert weighted_res["consensus_reached"] is True
    assert weighted_res["tally"]["REJECT"] == 10.0
    assert weighted_res["tally"]["APPROVE"] == 2.0
    assert weighted_res["confidence"] == pytest.approx(10.0 / 12.0, 0.01)


# =============================================================================
# 7. Integration: swarm() and debate() with SwarmMesh
# =============================================================================
def test_swarm_delegating_through_mesh():
    """Verify swarm() accepts SwarmMesh directly or via mesh keyword."""
    mesh = SwarmMesh()
    mesh.register_node(DistributedNode(node_id="m1", name="Expert1", agent=lambda t: "Solution from Expert1"))
    mesh.register_node(DistributedNode(node_id="m2", name="Expert2", agent=lambda t: "Solution from Expert2"))

    # Pass mesh as agents argument
    synthesis = swarm(mesh, "Solve quantum distribution")
    assert "=== Swarm Synthesis" in synthesis
    assert "[Expert1]: Solution from Expert1" in synthesis
    assert "[Expert2]: Solution from Expert2" in synthesis

    # Pass mesh keyword
    synthesis2 = swarm(None, "Solve quantum distribution", mesh=mesh)
    assert "[Expert1]: Solution from Expert1" in synthesis2


def test_debate_delegating_through_mesh():
    """Verify debate() delegates turns through mesh nodes."""
    mesh = SwarmMesh()
    mesh.register_node(DistributedNode(node_id="d1", name="Alpha", role="debater", agent=lambda t: "I support Option A."))
    mesh.register_node(DistributedNode(node_id="d2", name="Beta", role="debater", agent=lambda t: "I support Option B."))

    debate_result = debate(mesh, "Architecture Selection", rounds=2)
    assert "=== Debate Consensus" in debate_result
    assert "Round 1" in debate_result
    assert "Round 2" in debate_result
    assert "Alpha: I support Option A." in debate_result
    assert "Beta: I support Option B." in debate_result
