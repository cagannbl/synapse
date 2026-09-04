import uuid
import json
import random
import urllib.request
import urllib.error
import threading
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence, Optional, Union, Callable
import numpy as np
from synapse.ai.agent_runtime import AgentRuntime
from synapse.core.tensor import Tensor


class DistributedNode:
    """
    Represents an agent or compute node in a cluster.
    Can run in-process (wrapping an AgentRuntime or callable)
    or communicate with remote agents via lightweight HTTP/socket RPC.
    """
    def __init__(
        self,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        role: str = "worker",
        capabilities: Optional[Sequence[str]] = None,
        weight: float = 1.0,
        agent: Optional[Any] = None,
        rpc_url: Optional[str] = None,
        timeout: float = 10.0,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.name = name or f"Node-{self.node_id}"
        self.host = host
        self.port = port
        self.role = role
        self.capabilities = list(capabilities) if capabilities else []
        self.weight = float(weight)
        self.agent = agent
        self.rpc_url = rpc_url
        self.timeout = float(timeout)
        self.metadata = metadata or {}

    @property
    def is_local(self) -> bool:
        """Determines if the node runs within the local Python process."""
        return self.agent is not None or (self.port is None and self.rpc_url is None)

    def has_capability(self, cap: str) -> bool:
        """Checks if this node provides the specified capability (case-insensitive)."""
        return any(c.lower() == cap.lower() for c in self.capabilities)

    def execute(self, task: str, context: Optional[dict[str, Any]] = None) -> Any:
        """
        Executes a task on this node.
        Dispatches to local agent or remote RPC endpoint.
        """
        if self.is_local:
            return self._execute_local(task, context)
        return self._execute_remote(task, context)

    def _execute_local(self, task: str, context: Optional[dict[str, Any]] = None) -> Any:
        ag = self.agent
        if ag is None:
            return f"[{self.name}] No agent attached. Task acknowledged: {task}"

        if hasattr(ag, "run"):
            return ag.run(task)
        elif callable(ag):
            return ag(task)
        else:
            return str(ag)

    def _execute_remote(self, task: str, context: Optional[dict[str, Any]] = None) -> Any:
        url = self.rpc_url or f"http://{self.host}:{self.port}/api/task"
        payload = json.dumps({
            "node_id": self.node_id,
            "task": task,
            "context": context or {}
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Synapse-DistributedNode/1.0"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read().decode("utf-8")
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        return parsed.get("result", parsed.get("reply", parsed))
                    return parsed
                except Exception:
                    return data
        except Exception as e:
            return f"[RPC Error from {self.name}@{url}]: {str(e)}"

    def run(self, task: str) -> Any:
        """Drop-in compatibility with AgentRuntime interface."""
        return self.execute(task)

    def __call__(self, task: str) -> Any:
        """Callable interface for easy piping and invocation."""
        return self.execute(task)

    def ping(self) -> bool:
        """Health-checks the node connectivity."""
        if self.is_local:
            return True

        url = self.rpc_url or f"http://{self.host}:{self.port}/api/status"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Synapse-Ping/1.0"})
            with urllib.request.urlopen(req, timeout=min(2.0, self.timeout)) as resp:
                return resp.status < 400
        except Exception:
            # Fallback to TCP connect test if HTTP health endpoint isn't set up
            if self.port:
                import socket
                try:
                    with socket.create_connection((self.host, self.port), timeout=1.0):
                        return True
                except Exception:
                    return False
            return False

    def to_dict(self) -> dict[str, Any]:
        """Serializes the node configuration into a dictionary."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "weight": self.weight,
            "is_local": self.is_local,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        loc = "local" if self.is_local else f"remote:{self.host}:{self.port}"
        return f"<DistributedNode '{self.name}' id={self.node_id} role={self.role} {loc} weight={self.weight}>"


class SwarmMesh:
    """
    Manages a distributed network mesh of agent and compute nodes.
    Supports concurrent task broadcast, capability-based routing, and
    majority / weighted consensus mechanisms.
    """
    def __init__(self, mesh_id: Optional[str] = None):
        self.mesh_id = mesh_id or f"mesh_{uuid.uuid4().hex[:8]}"
        self.nodes: dict[str, DistributedNode] = {}
        self._rr_indices: dict[str, int] = {}
        self._lock = threading.Lock()

    def register_node(self, node: Union[DistributedNode, Any], **kwargs) -> DistributedNode:
        """Registers a node into the mesh. Can auto-wrap raw agents into DistributedNode."""
        if not isinstance(node, DistributedNode):
            node = DistributedNode(agent=node, **kwargs)

        with self._lock:
            self.nodes[node.node_id] = node
        return node

    def unregister_node(self, node_id: str) -> bool:
        """Removes a node from the mesh by its ID."""
        with self._lock:
            return self.nodes.pop(node_id, None) is not None

    def get_node(self, node_id: str) -> Optional[DistributedNode]:
        """Retrieves a node by its ID."""
        return self.nodes.get(node_id)

    def get_nodes(
        self,
        capability: Optional[str] = None,
        role: Optional[str] = None
    ) -> list[DistributedNode]:
        """Filters nodes by capability and/or role."""
        with self._lock:
            matched = list(self.nodes.values())

        if capability:
            matched = [n for n in matched if n.has_capability(capability)]
        if role:
            matched = [n for n in matched if n.role.lower() == role.lower()]
        return matched

    def broadcast(
        self,
        task: str,
        capability: Optional[str] = None,
        role: Optional[str] = None,
        context: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Executes a task concurrently across all matching nodes in the mesh.
        Returns a mapping of {node_id: result}.
        """
        targets = self.get_nodes(capability=capability, role=role)
        if not targets:
            return {}

        results: dict[str, Any] = {}
        max_workers = min(32, len(targets))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_node = {
                executor.submit(node.execute, task, context): node
                for node in targets
            }
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    results[node.node_id] = future.result()
                except Exception as e:
                    results[node.node_id] = f"[Execution Error]: {str(e)}"

        return results

    def route_task(
        self,
        task: str,
        capability: Optional[str] = None,
        role: Optional[str] = None,
        strategy: str = "round_robin",
        context: Optional[dict[str, Any]] = None
    ) -> Any:
        """
        Routes a task to a selected node based on the given strategy:
        'round_robin', 'random', 'weighted', or 'first_available'.
        """
        targets = self.get_nodes(capability=capability, role=role)
        if not targets:
            raise ValueError(f"No nodes in SwarmMesh match capability='{capability}', role='{role}'")

        if strategy == "round_robin":
            key = f"{capability}:{role}"
            with self._lock:
                idx = self._rr_indices.get(key, 0) % len(targets)
                self._rr_indices[key] = idx + 1
                selected_node = targets[idx]

        elif strategy == "random":
            selected_node = random.choice(targets)

        elif strategy == "weighted":
            weights = [max(0.001, n.weight) for n in targets]
            selected_node = random.choices(targets, weights=weights, k=1)[0]

        elif strategy == "first_available":
            selected_node = None
            for n in targets:
                if n.ping():
                    selected_node = n
                    break
            if selected_node is None:
                selected_node = targets[0]
        else:
            selected_node = targets[0]

        return selected_node.execute(task, context=context)

    def consensus(
        self,
        task: str,
        strategy: str = "majority",
        capability: Optional[str] = None,
        role: Optional[str] = None,
        candidates: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Evaluates a task across nodes to achieve consensus.
        Supports 'majority' (1 node = 1 vote) and 'weighted' (scaled by node.weight).
        """
        targets = self.get_nodes(capability=capability, role=role)
        if not targets:
            return {
                "winner": None,
                "strategy": strategy,
                "votes": {},
                "tally": {},
                "total_nodes": 0,
                "consensus_reached": False,
                "confidence": 0.0,
                "raw_responses": {},
                "summary": "No nodes available for consensus."
            }

        raw_responses = self.broadcast(task, capability=capability, role=role)
        votes: dict[str, str] = {}
        tally: dict[str, float] = collections.defaultdict(float)
        total_weight = 0.0

        for node in targets:
            node_id = node.node_id
            resp = raw_responses.get(node_id, "")
            verdict = self._extract_verdict(resp, candidates)
            votes[node_id] = verdict

            w = 1.0 if strategy == "majority" else node.weight
            tally[verdict] += w
            total_weight += w

        if tally:
            winner, max_score = max(tally.items(), key=lambda item: item[1])
            confidence = (max_score / total_weight) if total_weight > 0 else 0.0
            consensus_reached = confidence > 0.5
        else:
            winner = None
            confidence = 0.0
            consensus_reached = False

        summary = (
            f"Consensus [{strategy}]: Winner='{winner}' "
            f"({confidence:.1%} agreement across {len(targets)} nodes, reached={consensus_reached})"
        )

        return {
            "winner": winner,
            "strategy": strategy,
            "votes": votes,
            "tally": dict(tally),
            "total_nodes": len(targets),
            "consensus_reached": consensus_reached,
            "confidence": round(confidence, 4),
            "raw_responses": raw_responses,
            "summary": summary
        }

    def _extract_verdict(self, resp: Any, candidates: Optional[list[str]] = None) -> str:
        """Normalizes and extracts a discrete decision/vote from a node's output."""
        if isinstance(resp, dict):
            for k in ("verdict", "vote", "decision", "choice", "result"):
                if k in resp:
                    return str(resp[k]).strip()

        resp_str = str(resp).strip()
        if candidates:
            low = resp_str.lower()
            for cand in candidates:
                if cand.lower() in low:
                    return cand

        # If it's a short response or first line
        lines = [line.strip() for line in resp_str.splitlines() if line.strip()]
        if lines:
            first = lines[0]
            if len(first) <= 80:
                return first
            return first[:80] + "..."
        return "UNKNOWN"

    def create_tensor_mesh(self, node_ids: Optional[Sequence[str]] = None) -> "TensorMesh":
        """
        Creates a TensorMesh coordinating distributed tensor operations across mesh nodes.
        If node_ids is specified, only those nodes are included.
        """
        if node_ids is not None:
            target_nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        else:
            target_nodes = list(self.nodes.values())
        return TensorMesh(nodes=target_nodes, swarm_mesh=self)

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"<SwarmMesh id={self.mesh_id} nodes={len(self.nodes)}>"


class DistributedTensor:
    """
    Wraps a tensor with a mesh identifier, node assignment, and sharding specification.
    Allows seamless distributed operations while tracking global vs local shape and node ownership.
    """
    def __init__(
        self,
        local_tensor: Union[Tensor, Any],
        mesh_id: str,
        node_id: Optional[str] = None,
        shard_dim: Optional[int] = 0,
        shard_index: int = 0,
        num_shards: int = 1,
        global_shape: Optional[tuple[int, ...]] = None,
    ):
        if isinstance(local_tensor, Tensor):
            self.local_tensor = local_tensor
        elif isinstance(local_tensor, DistributedTensor):
            self.local_tensor = local_tensor.local_tensor
        else:
            self.local_tensor = Tensor(local_tensor)

        self.mesh_id = mesh_id
        self.node_id = node_id
        self.shard_dim = shard_dim
        self.shard_index = int(shard_index)
        self.num_shards = int(num_shards)
        self.global_shape = tuple(global_shape) if global_shape is not None else self.local_tensor.shape

    @property
    def data(self) -> np.ndarray:
        return self.local_tensor.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.local_tensor.shape

    @property
    def device(self) -> Any:
        return self.local_tensor.device

    @property
    def requires_grad(self) -> bool:
        return self.local_tensor.requires_grad

    @property
    def is_sharded(self) -> bool:
        return self.shard_dim is not None and self.num_shards > 1

    def numpy(self) -> np.ndarray:
        return self.local_tensor.data

    def to(self, device: Any) -> "DistributedTensor":
        new_local = self.local_tensor.to(device)
        return DistributedTensor(
            local_tensor=new_local,
            mesh_id=self.mesh_id,
            node_id=self.node_id,
            shard_dim=self.shard_dim,
            shard_index=self.shard_index,
            num_shards=self.num_shards,
            global_shape=self.global_shape,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_id": self.mesh_id,
            "node_id": self.node_id,
            "shard_dim": self.shard_dim,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "local_shape": self.shape,
            "global_shape": self.global_shape,
            "device": str(self.device),
        }

    def __repr__(self) -> str:
        shard_str = f"shard={self.shard_index}/{self.num_shards} (dim={self.shard_dim})" if self.is_sharded else "replicated"
        return (
            f"<DistributedTensor mesh='{self.mesh_id}' node='{self.node_id}' {shard_str} "
            f"shape={self.shape} global_shape={self.global_shape} device={self.device}>"
        )

    # Arithmetic operator overloads delegating to local_tensor
    def __add__(self, other: Any) -> "DistributedTensor":
        other_val = other.local_tensor if isinstance(other, DistributedTensor) else other
        res = self.local_tensor + other_val
        return DistributedTensor(
            res, self.mesh_id, self.node_id, self.shard_dim, self.shard_index, self.num_shards, self.global_shape
        )

    def __radd__(self, other: Any) -> "DistributedTensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "DistributedTensor":
        other_val = other.local_tensor if isinstance(other, DistributedTensor) else other
        res = self.local_tensor - other_val
        return DistributedTensor(
            res, self.mesh_id, self.node_id, self.shard_dim, self.shard_index, self.num_shards, self.global_shape
        )

    def __rsub__(self, other: Any) -> "DistributedTensor":
        other_val = other.local_tensor if isinstance(other, DistributedTensor) else other
        res = other_val - self.local_tensor
        return DistributedTensor(
            res, self.mesh_id, self.node_id, self.shard_dim, self.shard_index, self.num_shards, self.global_shape
        )

    def __mul__(self, other: Any) -> "DistributedTensor":
        other_val = other.local_tensor if isinstance(other, DistributedTensor) else other
        res = self.local_tensor * other_val
        return DistributedTensor(
            res, self.mesh_id, self.node_id, self.shard_dim, self.shard_index, self.num_shards, self.global_shape
        )

    def __rmul__(self, other: Any) -> "DistributedTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "DistributedTensor":
        other_val = other.local_tensor if isinstance(other, DistributedTensor) else other
        res = self.local_tensor / other_val
        return DistributedTensor(
            res, self.mesh_id, self.node_id, self.shard_dim, self.shard_index, self.num_shards, self.global_shape
        )

    def __matmul__(self, other: Any) -> "DistributedTensor":
        other_val = other.local_tensor if isinstance(other, DistributedTensor) else other
        res = self.local_tensor @ other_val
        return DistributedTensor(
            res, self.mesh_id, self.node_id, self.shard_dim, self.shard_index, self.num_shards, self.global_shape
        )

    def __len__(self) -> int:
        return len(self.local_tensor.data)

    def __getitem__(self, item: Any) -> Any:
        return self.local_tensor.data[item]


class TensorMesh:
    """
    Coordinates distributed tensor operations across a cluster of DistributedNode instances.
    Provides tensor sharding across nodes, gathering, broadcasting, and concurrent All-Reduce primitives.
    """
    def __init__(
        self,
        mesh_id: Optional[str] = None,
        nodes: Optional[Union[Sequence[DistributedNode], dict[str, DistributedNode]]] = None,
        swarm_mesh: Optional[SwarmMesh] = None,
        max_workers: int = 16,
    ):
        self.mesh_id = mesh_id or f"tmesh_{uuid.uuid4().hex[:8]}"
        self.swarm_mesh = swarm_mesh
        self.max_workers = max_workers
        self.nodes: dict[str, DistributedNode] = {}
        self.node_buffers: dict[str, Tensor] = {}
        self._lock = threading.Lock()

        if nodes:
            if isinstance(nodes, dict):
                for nid, n in nodes.items():
                    self.register_node(n)
            else:
                for n in nodes:
                    self.register_node(n)

    def register_node(self, node: Union[DistributedNode, Any], **kwargs) -> DistributedNode:
        """Registers a compute node into the tensor mesh."""
        if not isinstance(node, DistributedNode):
            node = DistributedNode(agent=node, **kwargs)
        with self._lock:
            self.nodes[node.node_id] = node
        return node

    def unregister_node(self, node_id: str) -> bool:
        """Removes a compute node from the tensor mesh."""
        with self._lock:
            self.node_buffers.pop(node_id, None)
            return self.nodes.pop(node_id, None) is not None

    def get_node(self, node_id: str) -> Optional[DistributedNode]:
        """Retrieves a node by its ID."""
        return self.nodes.get(node_id)

    def set_node_buffer(self, node_id: str, tensor: Union[Tensor, DistributedTensor, Any]) -> None:
        """Assigns or updates the local tensor buffer of a node in the mesh."""
        if isinstance(tensor, DistributedTensor):
            t = tensor.local_tensor
        elif isinstance(tensor, Tensor):
            t = tensor
        else:
            t = Tensor(tensor)

        with self._lock:
            self.node_buffers[node_id] = t
            node = self.nodes.get(node_id)
            if node:
                node.metadata["tensor_buffer"] = t

    def get_node_buffer(self, node_id: str) -> Optional[Tensor]:
        """Retrieves the current local tensor buffer stored on a node."""
        with self._lock:
            return self.node_buffers.get(node_id)

    def shard_tensor(
        self,
        tensor: Union[Tensor, DistributedTensor, Any],
        shard_dim: int = 0
    ) -> list[DistributedTensor]:
        """
        Shards a tensor along shard_dim across all participating nodes in the mesh.
        Returns a list of DistributedTensor instances assigned to individual nodes.
        """
        if isinstance(tensor, DistributedTensor):
            t = tensor.local_tensor
        elif isinstance(tensor, Tensor):
            t = tensor
        else:
            t = Tensor(tensor)

        node_list = list(self.nodes.values())
        num_nodes = len(node_list)
        if num_nodes == 0:
            raise ValueError("Cannot shard tensor: No nodes registered in TensorMesh.")

        # Split along shard_dim
        chunks = np.array_split(t.data, num_nodes, axis=shard_dim)

        distributed_shards: list[DistributedTensor] = []
        for idx, (node, chunk) in enumerate(zip(node_list, chunks)):
            local_t = Tensor(chunk, requires_grad=t.requires_grad, device=t.device)
            dt = DistributedTensor(
                local_tensor=local_t,
                mesh_id=self.mesh_id,
                node_id=node.node_id,
                shard_dim=shard_dim,
                shard_index=idx,
                num_shards=num_nodes,
                global_shape=t.shape,
            )
            self.set_node_buffer(node.node_id, local_t)
            distributed_shards.append(dt)

        return distributed_shards

    def gather(
        self,
        distributed_tensors: Optional[Sequence[DistributedTensor]] = None,
        shard_dim: Optional[int] = None
    ) -> Tensor:
        """
        Gathers sharded chunks across nodes back into a single consolidated Tensor.
        """
        if distributed_tensors is None:
            with self._lock:
                tensors = [self.node_buffers[nid] for nid in self.nodes if nid in self.node_buffers]
            if not tensors:
                raise ValueError("No tensor buffers available to gather in TensorMesh.")
            dim = shard_dim if shard_dim is not None else 0
            arrays = [t.data for t in tensors]
            gathered = np.concatenate(arrays, axis=dim)
            return Tensor(gathered, device=tensors[0].device)

        if not distributed_tensors:
            raise ValueError("Empty distributed_tensors provided to gather.")

        dim = shard_dim if shard_dim is not None else (
            distributed_tensors[0].shard_dim if distributed_tensors[0].shard_dim is not None else 0
        )
        sorted_shards = sorted(distributed_tensors, key=lambda d: d.shard_index)
        arrays = [d.data for d in sorted_shards]
        gathered = np.concatenate(arrays, axis=dim)
        return Tensor(gathered, device=distributed_tensors[0].device)

    def broadcast_tensor(self, tensor: Union[Tensor, DistributedTensor, Any]) -> list[DistributedTensor]:
        """
        Broadcasts a copy of the tensor to all nodes in the mesh.
        """
        if isinstance(tensor, DistributedTensor):
            t = tensor.local_tensor
        elif isinstance(tensor, Tensor):
            t = tensor
        else:
            t = Tensor(tensor)

        node_list = list(self.nodes.values())
        out: list[DistributedTensor] = []
        for node in node_list:
            local_t = Tensor(t.data.copy(), requires_grad=t.requires_grad, device=t.device)
            dt = DistributedTensor(
                local_tensor=local_t,
                mesh_id=self.mesh_id,
                node_id=node.node_id,
                shard_dim=None,
                shard_index=0,
                num_shards=1,
                global_shape=t.shape,
            )
            self.set_node_buffer(node.node_id, local_t)
            out.append(dt)
        return out

    def all_reduce(
        self,
        tensor: Optional[Union[Tensor, DistributedTensor, Sequence[Union[Tensor, DistributedTensor]], dict[str, Any]]] = None,
        op: str = "sum"
    ) -> Tensor:
        """
        Concurrently aggregates tensor chunks across nodes using ThreadPoolExecutor
        and updates local buffers.

        Parameters
        ----------
        tensor : Union[Tensor, DistributedTensor, Sequence, dict, None]
            - If Sequence of DistributedTensor: aggregates across these shard/node tensors.
            - If Sequence of Tensor/arrays: aggregates across elements corresponding to nodes.
            - If dict of {node_id: Tensor}: aggregates across specified node tensors.
            - If single DistributedTensor or Tensor: aggregates across nodes using provided/stored tensors.
            - If None: aggregates existing node buffers across all nodes.
        op : str
            Aggregation operation: 'sum', 'mean', 'max', 'min', or 'prod'.

        Returns
        -------
        Tensor
            The consolidated reduced Tensor.
        """
        op_norm = op.lower().strip()
        valid_ops = ("sum", "mean", "avg", "max", "min", "prod", "product")
        if op_norm not in valid_ops:
            raise ValueError(f"Unsupported all_reduce op '{op}'. Must be one of {valid_ops}.")

        # 1. Prepare items to aggregate per node
        node_items: list[tuple[str, Any]] = []

        if isinstance(tensor, dict):
            for nid, item in tensor.items():
                node_items.append((nid, item))
        elif isinstance(tensor, (list, tuple)):
            for idx, item in enumerate(tensor):
                if isinstance(item, DistributedTensor) and item.node_id:
                    nid = item.node_id
                elif idx < len(self.nodes):
                    nid = list(self.nodes.keys())[idx]
                else:
                    nid = f"node_{idx}"
                node_items.append((nid, item))
        elif isinstance(tensor, DistributedTensor):
            for nid, node in self.nodes.items():
                buf = self.get_node_buffer(nid)
                item = buf if buf is not None else tensor
                node_items.append((nid, item))
        elif isinstance(tensor, Tensor) or isinstance(tensor, np.ndarray):
            for nid, node in self.nodes.items():
                buf = self.get_node_buffer(nid)
                item = buf if buf is not None else tensor
                node_items.append((nid, item))
        else:
            for nid, node in self.nodes.items():
                buf = self.get_node_buffer(nid)
                if buf is None and "tensor_buffer" in node.metadata:
                    buf = node.metadata["tensor_buffer"]
                if buf is not None:
                    node_items.append((nid, buf))

        if not node_items:
            raise ValueError("No tensor chunks or node buffers found to execute all_reduce.")

        # 2. Concurrently fetch tensor data chunks using ThreadPoolExecutor
        def _fetch_data(node_id: str, item: Any) -> np.ndarray:
            if isinstance(item, DistributedTensor):
                return item.data
            elif isinstance(item, Tensor):
                return item.data
            elif isinstance(item, np.ndarray):
                return item
            elif isinstance(item, (int, float, list, tuple)):
                return np.array(item, dtype=np.float64)
            else:
                buf = self.get_node_buffer(node_id)
                if buf is not None:
                    return buf.data
                raise ValueError(f"Unable to extract tensor data for node '{node_id}'")

        workers = min(self.max_workers, max(1, len(node_items)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_nid = {
                executor.submit(_fetch_data, nid, item): nid
                for nid, item in node_items
            }
            chunks: list[np.ndarray] = []
            for future in as_completed(future_to_nid):
                chunks.append(future.result())

        # 3. Perform reduction
        if op_norm == "sum":
            reduced_arr = np.sum(chunks, axis=0)
        elif op_norm in ("mean", "avg"):
            reduced_arr = np.mean(chunks, axis=0)
        elif op_norm == "max":
            reduced_arr = np.maximum.reduce(chunks)
        elif op_norm == "min":
            reduced_arr = np.minimum.reduce(chunks)
        elif op_norm in ("prod", "product"):
            reduced_arr = np.prod(chunks, axis=0)
        else:
            reduced_arr = np.sum(chunks, axis=0)

        reduced_tensor = Tensor(reduced_arr)

        # 4. Concurrently update local buffers on all participating nodes
        all_target_nodes = list(self.nodes.values())
        if all_target_nodes:
            update_workers = min(self.max_workers, len(all_target_nodes))
            with ThreadPoolExecutor(max_workers=update_workers) as executor:
                def _update_node(n: DistributedNode):
                    self.set_node_buffer(n.node_id, reduced_tensor)

                list(executor.map(_update_node, all_target_nodes))

        # If tensor was a list of DistributedTensor, update their local_tensor reference too
        if isinstance(tensor, (list, tuple)):
            for it in tensor:
                if isinstance(it, DistributedTensor):
                    it.local_tensor = reduced_tensor
        elif isinstance(tensor, DistributedTensor):
            tensor.local_tensor = reduced_tensor

        return reduced_tensor

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"<TensorMesh id={self.mesh_id} nodes={len(self.nodes)} buffers={len(self.node_buffers)}>"



def swarm(
    agents: Optional[Union[Sequence[Any], SwarmMesh]] = None,
    task: str = "",
    mesh: Optional[SwarmMesh] = None
) -> str:
    """
    Birden fazla ajanın aynı görevi paralel veya sırayla inceleyip
    çıktılarını tek bir sentezde birleştirdiği çoklu ajan sürü (Swarm) primitifi.
    Supports optional delegation through SwarmMesh.
    """
    active_mesh = mesh or (agents if isinstance(agents, SwarmMesh) else None)

    if active_mesh is not None:
        raw = active_mesh.broadcast(task)
        responses = []
        for node_id, resp in raw.items():
            node = active_mesh.get_node(node_id)
            node_name = node.name if node else node_id
            responses.append(f"[{node_name}]: {resp}")
        synthesis = "\n\n".join(responses)
        return f"=== Swarm Synthesis ({len(raw)} Agents) ===\n{synthesis}"

    agent_list = list(agents) if agents else []
    if not agent_list:
        return f"No agents provided for task: {task}"

    responses = []
    for i, ag in enumerate(agent_list):
        name = getattr(ag, "name", f"Agent_{i+1}")
        if hasattr(ag, "run"):
            resp = ag.run(task)
        elif hasattr(ag, "execute"):
            resp = ag.execute(task)
        elif callable(ag):
            resp = ag(task)
        else:
            resp = str(ag)
        responses.append(f"[{name}]: {resp}")

    synthesis = "\n\n".join(responses)
    return f"=== Swarm Synthesis ({len(agent_list)} Agents) ===\n{synthesis}"


def debate(
    agents: Optional[Union[Sequence[Any], SwarmMesh]] = None,
    topic: str = "",
    rounds: int = 2,
    mesh: Optional[SwarmMesh] = None
) -> str:
    """
    İki veya daha fazla ajanın bir konuyu karşılıklı tartışarak (Debate)
    nihai bir konsensüse ulaştığı karar alma primitifi.
    Supports optional delegation through SwarmMesh.
    """
    active_mesh = mesh or (agents if isinstance(agents, SwarmMesh) else None)
    if active_mesh is not None:
        agent_list = active_mesh.get_nodes(role="debater") or list(active_mesh.nodes.values())
    else:
        agent_list = list(agents) if agents else []

    if not agent_list:
        return f"No agents available to debate: {topic}"

    transcript = []
    current_context = f"Topic for debate: {topic}"

    for r in range(1, rounds + 1):
        transcript.append(f"--- Round {r} ---")
        round_inputs = []
        for i, ag in enumerate(agent_list):
            name = getattr(ag, "name", f"Debater_{i+1}")
            turn_prompt = f"{current_context}\nProvide your perspective and address previous points."
            if hasattr(ag, "run"):
                resp = ag.run(turn_prompt)
            elif hasattr(ag, "execute"):
                resp = ag.execute(turn_prompt)
            elif callable(ag):
                resp = ag(turn_prompt)
            else:
                resp = str(ag)
            transcript.append(f"{name}: {resp}")
            round_inputs.append(f"{name} argued: {str(resp)[:120]}...")

        current_context = f"Topic: {topic}\nSummary of previous round:\n" + "\n".join(round_inputs)

    consensus_summary = (
        f"=== Debate Consensus ({len(agent_list)} Agents, {rounds} Rounds) ===\n"
        + "\n".join(transcript)
    )
    return consensus_summary
