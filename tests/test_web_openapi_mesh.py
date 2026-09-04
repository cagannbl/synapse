import time
import json
import urllib.request
import urllib.error
import pytest
import numpy as np

from synapse.core.web import (
    serve,
    SynapseServer,
    SynapseRequest,
    SynapseResponse,
    SynapseSSEResponse,
    generate_openapi_spec,
    render_swagger_ui_html,
)
from synapse.core.tensor import Tensor
from synapse.ai.agent_swarm import (
    DistributedNode,
    SwarmMesh,
    TensorMesh,
    DistributedTensor,
)


# =============================================================================
# 1. OpenAPI 3.1 Specification Generation Tests
# =============================================================================
def test_generate_openapi_spec_structure():
    """Verify generate_openapi_spec outputs valid OpenAPI 3.1.0 root schema."""
    server = SynapseServer(
        port=8201,
        title="Synapse Core API",
        version="2.0.0",
        description="Next-generation AI runtime API"
    )

    def get_status(req: SynapseRequest) -> dict:
        """System health and uptime status.
        Returns the current cluster health and active agent counts.
        """
        return {"status": "healthy", "version": "2.0.0"}

    def create_agent(name: str, priority: int = 1) -> dict:
        """Create a new agent instance.
        Registers an autonomous agent into the cluster mesh.
        """
        return {"agent_id": "ag_123", "name": name, "priority": priority}

    def update_model(model_id: str, body: dict) -> dict:
        """Update model configurations."""
        return {"model_id": model_id, "updated": True}

    def delete_item(item_id: str) -> str:
        """Remove an item from storage."""
        return "deleted"

    server.add_route("/api/status", get_status, method="GET")
    server.add_route("/api/agents", create_agent, method="POST")
    server.add_route("/api/models/{model_id}", update_model, method="PUT")
    server.add_route("/api/items/{item_id}", delete_item, method="DELETE")
    server.add_ws_route("/ws/stream", lambda ws: None)

    spec = generate_openapi_spec(server)

    # Root properties
    assert spec["openapi"] == "3.1.0"
    assert spec["info"]["title"] == "Synapse Core API"
    assert spec["info"]["version"] == "2.0.0"
    assert spec["info"]["description"] == "Next-generation AI runtime API"
    assert "paths" in spec
    assert "components" in spec

    # Verify GET /api/status
    assert "/api/status" in spec["paths"]
    get_op = spec["paths"]["/api/status"]["get"]
    assert get_op["summary"] == "System health and uptime status."
    assert "active agent counts" in get_op["description"]
    assert "200" in get_op["responses"]
    assert "application/json" in get_op["responses"]["200"]["content"]

    # Verify POST /api/agents parameters & request body
    assert "/api/agents" in spec["paths"]
    post_op = spec["paths"]["/api/agents"]["post"]
    assert post_op["summary"] == "Create a new agent instance."
    assert "requestBody" in post_op
    # Query parameters
    param_names = [p["name"] for p in post_op.get("parameters", [])]
    assert "name" in param_names
    assert "priority" in param_names

    # Verify PUT /api/models/{model_id} path param
    assert "/api/models/{model_id}" in spec["paths"]
    put_op = spec["paths"]["/api/models/{model_id}"]["put"]
    path_params = [p for p in put_op.get("parameters", []) if p.get("in") == "path"]
    assert len(path_params) == 1
    assert path_params[0]["name"] == "model_id"
    assert path_params[0]["required"] is True

    # Verify DELETE /api/items/{item_id}
    assert "/api/items/{item_id}" in spec["paths"]
    del_op = spec["paths"]["/api/items/{item_id}"]["delete"]
    assert "200" in del_op["responses"]
    assert "text/plain" in del_op["responses"]["200"]["content"]

    # Verify WebSocket route in paths
    assert "/ws/stream" in spec["paths"]
    ws_op = spec["paths"]["/ws/stream"]["get"]
    assert "WebSockets" in ws_op["tags"]
    assert "101" in ws_op["responses"]


def test_generate_openapi_spec_from_dict_and_annotations():
    """Verify spec generation directly from a routes dictionary with typing annotations."""
    def sse_handler() -> SynapseSSEResponse:
        """Stream events to client."""
        def event_gen():
            yield "data: ok\n\n"
        return SynapseSSEResponse(event_gen())

    routes_dict = {
        "GET /events": sse_handler
    }

    spec = generate_openapi_spec(routes_dict, title="SSE Server", version="1.1")
    assert spec["openapi"] == "3.1.0"
    assert spec["info"]["title"] == "SSE Server"
    assert "/events" in spec["paths"]
    op = spec["paths"]["/events"]["get"]
    assert "text/event-stream" in op["responses"]["200"]["content"]


# =============================================================================
# 2. Live HTTP Server Tests: /docs and /openapi.json
# =============================================================================
def test_swagger_ui_and_openapi_endpoints_live():
    """Test live SynapseServer serves /docs (200 OK HTML) and /openapi.json (200 OK JSON)."""
    server = serve(
        port=8202,
        routes={
            "/api/ping": lambda req: {"ping": "pong"},
            "POST /api/echo": lambda req: {"echo": req.body}
        },
        enable_docs=True,
        title="Live Test API",
        blocking=False
    )

    try:
        time.sleep(0.3)

        # 1. Test /openapi.json returns 200 OK with application/json
        json_url = "http://127.0.0.1:8202/openapi.json"
        with urllib.request.urlopen(json_url) as res:
            assert res.status == 200
            assert "application/json" in res.headers.get("Content-Type")
            spec = json.loads(res.read().decode("utf-8"))
            assert spec["openapi"] == "3.1.0"
            assert spec["info"]["title"] == "Live Test API"
            assert "/api/ping" in spec["paths"]
            assert "/api/echo" in spec["paths"]

        # 2. Test /docs returns 200 OK with text/html containing Swagger UI & fallback
        docs_url = "http://127.0.0.1:8202/docs"
        with urllib.request.urlopen(docs_url) as res:
            assert res.status == 200
            assert "text/html" in res.headers.get("Content-Type")
            html = res.read().decode("utf-8")
            assert "<!DOCTYPE html>" in html
            assert "swagger-ui" in html
            assert "swagger-ui-bundle.js" in html
            assert "Live Test API" in html
            assert "offline-fallback" in html

    finally:
        server.stop()


def test_enable_docs_false_returns_404():
    """Verify /docs and /openapi.json return 404 when enable_docs=False."""
    server = SynapseServer(port=8203, enable_docs=False)
    server.start(blocking=False)

    try:
        time.sleep(0.3)
        with pytest.raises(urllib.error.HTTPError) as exc_docs:
            urllib.request.urlopen("http://127.0.0.1:8203/docs")
        assert exc_docs.value.code == 404

        with pytest.raises(urllib.error.HTTPError) as exc_json:
            urllib.request.urlopen("http://127.0.0.1:8203/openapi.json")
        assert exc_json.value.code == 404
    finally:
        server.stop()


# =============================================================================
# 3. Distributed Tensor & TensorMesh Tests
# =============================================================================
def test_distributed_tensor_properties_and_math():
    """Verify DistributedTensor wrapping, properties, and arithmetic delegation."""
    t = Tensor([10.0, 20.0, 30.0])
    dt = DistributedTensor(
        local_tensor=t,
        mesh_id="mesh_test_1",
        node_id="node_0",
        shard_dim=0,
        shard_index=0,
        num_shards=3,
        global_shape=(3, 3)
    )

    assert dt.mesh_id == "mesh_test_1"
    assert dt.node_id == "node_0"
    assert dt.shard_dim == 0
    assert dt.shard_index == 0
    assert dt.num_shards == 3
    assert dt.is_sharded is True
    assert dt.shape == (3,)
    assert dt.global_shape == (3, 3)
    assert np.allclose(dt.numpy(), [10.0, 20.0, 30.0])

    # Operator delegation
    dt_add = dt + 5.0
    assert isinstance(dt_add, DistributedTensor)
    assert np.allclose(dt_add.data, [15.0, 25.0, 35.0])

    dt_mul = dt * 2.0
    assert np.allclose(dt_mul.data, [20.0, 40.0, 60.0])

    dt_sub = dt - 2.0
    assert np.allclose(dt_sub.data, [8.0, 18.0, 28.0])

    dt_div = dt / 2.0
    assert np.allclose(dt_div.data, [5.0, 10.0, 15.0])

    d = dt.to_dict()
    assert d["node_id"] == "node_0"
    assert d["shard_index"] == 0


def test_tensor_mesh_sharding_and_gathering():
    """Verify TensorMesh accurately shards a tensor along dimensions and gathers it back."""
    node_a = DistributedNode(node_id="worker_a", name="Worker-A")
    node_b = DistributedNode(node_id="worker_b", name="Worker-B")
    node_c = DistributedNode(node_id="worker_c", name="Worker-C")

    mesh = TensorMesh(nodes=[node_a, node_b, node_c])
    assert len(mesh) == 3

    # Shape (6, 4) tensor
    raw_data = np.arange(24.0).reshape(6, 4)
    original_t = Tensor(raw_data)

    # Shard along dimension 0
    shards_dim0 = mesh.shard_tensor(original_t, shard_dim=0)
    assert len(shards_dim0) == 3

    for idx, s in enumerate(shards_dim0):
        assert s.shape == (2, 4)
        assert s.shard_dim == 0
        assert s.shard_index == idx
        assert s.global_shape == (6, 4)

    # Gather along dim 0 and verify identity
    gathered_dim0 = mesh.gather(shards_dim0, shard_dim=0)
    assert np.allclose(gathered_dim0.data, original_t.data)

    # Gather directly from mesh node buffers
    gathered_from_buffers = mesh.gather(shard_dim=0)
    assert np.allclose(gathered_from_buffers.data, original_t.data)


def test_tensor_mesh_broadcast():
    """Verify TensorMesh broadcast copies full tensor to all node buffers."""
    nodes = [DistributedNode(node_id=f"node_{i}") for i in range(4)]
    mesh = TensorMesh(nodes=nodes)

    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    broadcast_shards = mesh.broadcast_tensor(t)
    assert len(broadcast_shards) == 4

    for node in nodes:
        buf = mesh.get_node_buffer(node.node_id)
        assert buf is not None
        assert np.allclose(buf.data, t.data)


def test_tensor_mesh_all_reduce_sum_mean_max():
    """Verify concurrent All-Reduce (sum, mean, max) across simulated nodes using ThreadPoolExecutor."""
    node1 = DistributedNode(node_id="n1", name="GPU-0")
    node2 = DistributedNode(node_id="n2", name="GPU-1")
    node3 = DistributedNode(node_id="n3", name="GPU-2")

    mesh = TensorMesh(nodes=[node1, node2, node3])

    # 1. All-Reduce Sum
    mesh.set_node_buffer("n1", Tensor([1.0, 2.0]))
    mesh.set_node_buffer("n2", Tensor([3.0, 4.0]))
    mesh.set_node_buffer("n3", Tensor([5.0, 6.0]))

    reduced_sum = mesh.all_reduce(op="sum")
    assert np.allclose(reduced_sum.data, [9.0, 12.0])

    # Verify local buffers on all nodes were updated concurrently
    for nid in ("n1", "n2", "n3"):
        buf = mesh.get_node_buffer(nid)
        assert buf is not None
        assert np.allclose(buf.data, [9.0, 12.0])
        assert np.allclose(mesh.get_node(nid).metadata["tensor_buffer"].data, [9.0, 12.0])

    # 2. All-Reduce Mean
    mesh.set_node_buffer("n1", Tensor([10.0, 20.0]))
    mesh.set_node_buffer("n2", Tensor([20.0, 40.0]))
    mesh.set_node_buffer("n3", Tensor([30.0, 60.0]))

    reduced_mean = mesh.all_reduce(op="mean")
    assert np.allclose(reduced_mean.data, [20.0, 40.0])
    for nid in ("n1", "n2", "n3"):
        assert np.allclose(mesh.get_node_buffer(nid).data, [20.0, 40.0])

    # 3. All-Reduce Max
    mesh.set_node_buffer("n1", Tensor([15.0, 3.0]))
    mesh.set_node_buffer("n2", Tensor([4.0, 99.0]))
    mesh.set_node_buffer("n3", Tensor([22.0, 50.0]))

    reduced_max = mesh.all_reduce(op="max")
    assert np.allclose(reduced_max.data, [22.0, 99.0])
    for nid in ("n1", "n2", "n3"):
        assert np.allclose(mesh.get_node_buffer(nid).data, [22.0, 99.0])


def test_swarm_mesh_create_tensor_mesh():
    """Verify SwarmMesh.create_tensor_mesh connects Swarm cluster to TensorMesh."""
    swarm_mesh = SwarmMesh(mesh_id="cluster_alpha")
    swarm_mesh.register_node(DistributedNode(node_id="agent_1", role="trainer"))
    swarm_mesh.register_node(DistributedNode(node_id="agent_2", role="trainer"))
    swarm_mesh.register_node(DistributedNode(node_id="agent_3", role="evaluator"))

    # Create TensorMesh covering all nodes
    all_tmesh = swarm_mesh.create_tensor_mesh()
    assert isinstance(all_tmesh, TensorMesh)
    assert len(all_tmesh) == 3
    assert all_tmesh.swarm_mesh is swarm_mesh

    # Create TensorMesh for subset of nodes
    subset_tmesh = swarm_mesh.create_tensor_mesh(node_ids=["agent_1", "agent_2"])
    assert len(subset_tmesh) == 2
    assert "agent_1" in subset_tmesh.nodes
    assert "agent_2" in subset_tmesh.nodes
    assert "agent_3" not in subset_tmesh.nodes

    # Test all_reduce on created tensor mesh
    subset_tmesh.set_node_buffer("agent_1", Tensor([1.0, 5.0]))
    subset_tmesh.set_node_buffer("agent_2", Tensor([3.0, 7.0]))
    res = subset_tmesh.all_reduce(op="sum")
    assert np.allclose(res.data, [4.0, 12.0])
