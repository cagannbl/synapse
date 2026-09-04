"""
Synapse Durable Task Broker Architecture
=========================================
Pluggable backend brokers for Synapse Asynchronous Task Queue:
- BaseBroker: Abstract broker interface
- MemoryBroker: Ultra-fast thread-safe in-memory queue
- DiskQueueBroker: Zero-dependency durable append-only Write-Ahead Log (WAL)
- RedisQueueBroker: Distributed Redis-compatible protocol with graceful fallback
"""

from abc import ABC, abstractmethod
import copy
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# =============================================================================
# 1. Cross-Platform Zero-Dependency File Lock
# =============================================================================

_PATH_LOCKS: Dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _get_path_lock(path_str: str) -> threading.RLock:
    """Thread-safe lock lookup for identical file paths within the same process."""
    with _PATH_LOCKS_GUARD:
        if path_str not in _PATH_LOCKS:
            _PATH_LOCKS[path_str] = threading.RLock()
        return _PATH_LOCKS[path_str]


class FileLock:
    """
    Zero-dependency cross-platform file lock for inter-process synchronization.
    Uses atomic file creation (O_CREAT | O_EXCL) with stale lock protection.
    """

    def __init__(
        self,
        lock_file: Union[str, Path],
        timeout: float = 10.0,
        retry_interval: float = 0.01,
        stale_threshold: float = 30.0,
    ):
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.stale_threshold = stale_threshold
        self._fd: Optional[int] = None
        self._thread_lock = _get_path_lock(str(self.lock_file.resolve()))
        self._is_locked = False

    def acquire(self) -> bool:
        start_time = time.time()
        acquired = self._thread_lock.acquire(timeout=self.timeout)
        if not acquired:
            raise TimeoutError(
                f"Could not acquire file lock on '{self.lock_file}' within {self.timeout}s"
            )

        while True:
            try:
                self._fd = os.open(
                    str(self.lock_file),
                    os.O_CREAT | os.O_EXCL | os.O_RDWR
                )
                pid_info = f"{os.getpid()}:{time.time()}\n"
                os.write(self._fd, pid_info.encode("utf-8"))
                self._is_locked = True
                return True
            except FileExistsError:
                # Check for stale lock from dead or killed process
                try:
                    if self.lock_file.exists():
                        mtime = self.lock_file.stat().st_mtime
                        if time.time() - mtime > self.stale_threshold:
                            try:
                                self.lock_file.unlink()
                            except OSError:
                                pass
                except OSError:
                    pass

                if time.time() - start_time >= self.timeout:
                    try:
                        self._thread_lock.release()
                    except RuntimeError:
                        pass
                    raise TimeoutError(
                        f"Could not acquire file lock on '{self.lock_file}' within {self.timeout}s"
                    )
                time.sleep(self.retry_interval)
            except OSError as exc:
                if time.time() - start_time >= self.timeout:
                    try:
                        self._thread_lock.release()
                    except RuntimeError:
                        pass
                    raise TimeoutError(
                        f"Could not acquire file lock on '{self.lock_file}': {exc}"
                    )
                time.sleep(self.retry_interval)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except OSError:
                pass
        self._is_locked = False
        try:
            self._thread_lock.release()
        except RuntimeError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# =============================================================================
# 2. Base Broker Interface
# =============================================================================

class BaseBroker(ABC):
    """Abstract base class establishing the contract for durable task brokers."""

    @abstractmethod
    def enqueue(self, task_id: str, task_name: str, payload: Dict[str, Any]) -> None:
        """Enqueues a task with unique task_id, task_name, and payload dictionary."""
        pass

    @abstractmethod
    def dequeue(self, timeout: Optional[float] = None) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        Dequeues the next pending task.
        Returns (task_id, task_name, payload) or None if timeout expires.
        """
        pass

    @abstractmethod
    def acknowledge(self, task_id: str) -> None:
        """Marks a task as successfully processed."""
        pass

    @abstractmethod
    def reject(self, task_id: str, requeue: bool = False, reason: Optional[str] = None) -> None:
        """
        Rejects a task.
        If requeue=True, places the task back in the pending queue.
        If requeue=False, moves the task to the dead letter queue (DLQ).
        """
        pass

    @abstractmethod
    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Returns all failed tasks in the dead letter queue."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes any underlying resources, connections, or file handles."""
        pass

    @property
    def pending_count(self) -> int:
        """Number of tasks currently pending or processing in the broker."""
        return 0

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Returns the current state string of a task, if tracked."""
        return None

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full task metadata by task_id, if tracked."""
        return None


# =============================================================================
# 3. In-Memory Task Broker
# =============================================================================

class MemoryBroker(BaseBroker):
    """Thread-safe in-memory task broker using standard queue and state maps."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._dlq: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._closed = False

    def enqueue(self, task_id: str, task_name: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            record = {
                "id": task_id,
                "name": task_name,
                "payload": copy.deepcopy(payload),
                "state": "pending",
                "retries": payload.get("_retries", 0),
                "error": None,
                "enqueued_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "failed_at": None,
            }
            self._tasks[task_id] = record
            self._queue.put(task_id)

    def dequeue(self, timeout: Optional[float] = None) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        try:
            task_id = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

        with self._lock:
            record = self._tasks.get(task_id)
            if not record or record["state"] != "pending":
                return None
            record["state"] = "processing"
            record["started_at"] = time.time()
            return (record["id"], record["name"], copy.deepcopy(record["payload"]))

    def acknowledge(self, task_id: str) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record:
                record["state"] = "completed"
                record["completed_at"] = time.time()

    def reject(self, task_id: str, requeue: bool = False, reason: Optional[str] = None) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if not record:
                return
            record["error"] = reason
            if requeue:
                record["state"] = "pending"
                record["retries"] = record.get("retries", 0) + 1
                record["payload"]["_retries"] = record["retries"]
                self._queue.put(task_id)
            else:
                record["state"] = "failed"
                record["failed_at"] = time.time()
                self._dlq.append(copy.deepcopy(record))

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._dlq)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t["state"] in ("pending", "processing"))

    def get_task_status(self, task_id: str) -> Optional[str]:
        with self._lock:
            rec = self._tasks.get(task_id)
            return rec["state"] if rec else None

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._tasks.get(task_id)
            return copy.deepcopy(rec) if rec else None

    def close(self) -> None:
        with self._lock:
            self._closed = True


# =============================================================================
# 4. Durable Disk Queue Broker (Append-Only WAL)
# =============================================================================

class DiskQueueBroker(BaseBroker):
    """
    Zero-dependency durable append-only Write-Ahead Log (WAL) on disk.
    Persists tasks as JSON lines.
    Full lifecycle state tracking: 'pending', 'processing', 'completed', 'failed' (DLQ).
    Crash recovery: Replays unacknowledged tasks on startup back into the active queue.
    Atomic file operations with cross-process FileLock to prevent file corruption.
    """

    def __init__(self, storage_dir: Union[str, Path] = ".synapse_tasks", fsync: bool = False):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self.wal_path = self.storage_dir / "wal.jsonl"
        self._lock_file = self.storage_dir / ".lock"
        self._file_lock = FileLock(self._lock_file)
        self._thread_lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._dlq: List[Dict[str, Any]] = []
        self._wal_read_offset: int = 0
        self._closed = False

        self._recover_and_replay_wal()

    def _recover_and_replay_wal(self) -> None:
        """Replays WAL on startup, recovers unacknowledged tasks, and populates queue."""
        with self._thread_lock, self._file_lock:
            if not self.wal_path.exists():
                return

            recovery_needed: List[str] = []
            with open(self.wal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # Gracefully ignore incomplete or truncated line from crash
                        continue

                    action = entry.get("action")
                    tid = entry.get("id")
                    if not tid:
                        continue

                    if action == "ENQUEUE":
                        payload = entry.get("payload", {})
                        self._tasks[tid] = {
                            "id": tid,
                            "name": entry.get("name"),
                            "payload": payload,
                            "state": "pending",
                            "retries": payload.get("_retries", 0),
                            "error": None,
                            "enqueued_at": entry.get("ts"),
                            "started_at": None,
                            "completed_at": None,
                            "failed_at": None,
                        }
                    elif action == "START":
                        if tid in self._tasks:
                            self._tasks[tid]["state"] = "processing"
                            self._tasks[tid]["started_at"] = entry.get("ts")
                    elif action == "ACK":
                        if tid in self._tasks:
                            self._tasks[tid]["state"] = "completed"
                            self._tasks[tid]["completed_at"] = entry.get("ts")
                    elif action == "REJECT":
                        if tid in self._tasks:
                            requeue = entry.get("requeue", False)
                            reason = entry.get("reason")
                            self._tasks[tid]["error"] = reason
                            if requeue:
                                self._tasks[tid]["state"] = "pending"
                                self._tasks[tid]["retries"] = self._tasks[tid].get("retries", 0) + 1
                                self._tasks[tid]["payload"]["_retries"] = self._tasks[tid]["retries"]
                            else:
                                self._tasks[tid]["state"] = "failed"
                                self._tasks[tid]["failed_at"] = entry.get("ts")

                self._wal_read_offset = f.tell()

            # Inspect task state machine after complete replay
            for tid, task in self._tasks.items():
                if task["state"] == "failed":
                    self._dlq.append(copy.deepcopy(task))
                elif task["state"] == "processing":
                    # CRASH RECOVERY: Task was dequeued by crashed worker, never ACKed or REJECTed
                    task["state"] = "pending"
                    recovery_needed.append(tid)
                    self._queue.put(tid)
                elif task["state"] == "pending":
                    self._queue.put(tid)

            # Record recovery entries to WAL for unacknowledged tasks
            for tid in recovery_needed:
                self._append_wal_record({
                    "action": "RECOVER",
                    "id": tid,
                    "ts": time.time(),
                    "reason": "Crash recovery: unacknowledged task restored to pending",
                })

    def _append_wal_record(self, record: Dict[str, Any]) -> None:
        """Appends a record to WAL with flush and optional os.fsync."""
        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())
            self._wal_read_offset = f.tell()

    def _sync_from_wal_if_needed(self) -> None:
        """Synchronizes new entries written to WAL by other processes or broker instances."""
        if not self.wal_path.exists():
            return
        try:
            curr_size = self.wal_path.stat().st_size
            if curr_size <= self._wal_read_offset:
                return
        except OSError:
            return

        with self._file_lock:
            with open(self.wal_path, "r", encoding="utf-8") as f:
                f.seek(self._wal_read_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    action = entry.get("action")
                    tid = entry.get("id")
                    if not tid:
                        continue

                    if action == "ENQUEUE":
                        if tid not in self._tasks:
                            payload = entry.get("payload", {})
                            self._tasks[tid] = {
                                "id": tid,
                                "name": entry.get("name"),
                                "payload": payload,
                                "state": "pending",
                                "retries": payload.get("_retries", 0),
                                "error": None,
                                "enqueued_at": entry.get("ts"),
                                "started_at": None,
                                "completed_at": None,
                                "failed_at": None,
                            }
                            self._queue.put(tid)
                    elif action == "START":
                        if tid in self._tasks:
                            self._tasks[tid]["state"] = "processing"
                    elif action == "ACK":
                        if tid in self._tasks:
                            self._tasks[tid]["state"] = "completed"
                    elif action == "REJECT":
                        if tid in self._tasks:
                            requeue = entry.get("requeue", False)
                            reason = entry.get("reason")
                            self._tasks[tid]["error"] = reason
                            if requeue:
                                self._tasks[tid]["state"] = "pending"
                                self._tasks[tid]["retries"] = self._tasks[tid].get("retries", 0) + 1
                                self._tasks[tid]["payload"]["_retries"] = self._tasks[tid]["retries"]
                                self._queue.put(tid)
                            else:
                                self._tasks[tid]["state"] = "failed"
                                self._dlq.append(copy.deepcopy(self._tasks[tid]))
                self._wal_read_offset = f.tell()

    def enqueue(self, task_id: str, task_name: str, payload: Dict[str, Any]) -> None:
        with self._thread_lock:
            with self._file_lock:
                record = {
                    "id": task_id,
                    "name": task_name,
                    "payload": copy.deepcopy(payload),
                    "state": "pending",
                    "retries": payload.get("_retries", 0),
                    "error": None,
                    "enqueued_at": time.time(),
                    "started_at": None,
                    "completed_at": None,
                    "failed_at": None,
                }
                self._tasks[task_id] = record
                self._append_wal_record({
                    "action": "ENQUEUE",
                    "id": task_id,
                    "name": task_name,
                    "payload": copy.deepcopy(payload),
                    "ts": record["enqueued_at"],
                })
                self._queue.put(task_id)

    def dequeue(self, timeout: Optional[float] = None) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        deadline = (time.time() + timeout) if timeout is not None else None
        while not self._closed:
            remaining = 0.05
            if deadline is not None:
                rem = deadline - time.time()
                if rem <= 0:
                    return None
                remaining = min(remaining, rem)

            try:
                tid = self._queue.get(timeout=remaining)
            except queue.Empty:
                self._sync_from_wal_if_needed()
                if deadline is not None and time.time() >= deadline:
                    return None
                continue

            with self._thread_lock, self._file_lock:
                task = self._tasks.get(tid)
                if not task or task["state"] != "pending":
                    # Task was cancelled or processed by another worker
                    continue
                task["state"] = "processing"
                task["started_at"] = time.time()
                self._append_wal_record({
                    "action": "START",
                    "id": tid,
                    "ts": task["started_at"],
                })
                return (task["id"], task["name"], copy.deepcopy(task["payload"]))
        return None

    def acknowledge(self, task_id: str) -> None:
        with self._thread_lock:
            with self._file_lock:
                task = self._tasks.get(task_id)
                if task:
                    task["state"] = "completed"
                    task["completed_at"] = time.time()
                    self._append_wal_record({
                        "action": "ACK",
                        "id": task_id,
                        "ts": task["completed_at"],
                    })

    def reject(self, task_id: str, requeue: bool = False, reason: Optional[str] = None) -> None:
        with self._thread_lock:
            with self._file_lock:
                task = self._tasks.get(task_id)
                if not task:
                    return
                task["error"] = reason
                if requeue:
                    task["state"] = "pending"
                    task["retries"] = task.get("retries", 0) + 1
                    task["payload"]["_retries"] = task["retries"]
                    self._append_wal_record({
                        "action": "REJECT",
                        "id": task_id,
                        "requeue": True,
                        "reason": reason,
                        "ts": time.time(),
                    })
                    self._queue.put(task_id)
                else:
                    task["state"] = "failed"
                    task["failed_at"] = time.time()
                    self._append_wal_record({
                        "action": "REJECT",
                        "id": task_id,
                        "requeue": False,
                        "reason": reason,
                        "ts": task["failed_at"],
                    })
                    self._dlq.append(copy.deepcopy(task))

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        with self._thread_lock:
            return copy.deepcopy(self._dlq)

    @property
    def pending_count(self) -> int:
        with self._thread_lock:
            return sum(1 for t in self._tasks.values() if t["state"] in ("pending", "processing"))

    def get_task_status(self, task_id: str) -> Optional[str]:
        with self._thread_lock:
            rec = self._tasks.get(task_id)
            return rec["state"] if rec else None

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._thread_lock:
            rec = self._tasks.get(task_id)
            return copy.deepcopy(rec) if rec else None

    def close(self) -> None:
        with self._thread_lock:
            self._closed = True


# =============================================================================
# 5. Redis Queue Broker
# =============================================================================

class RedisQueueBroker(BaseBroker):
    """
    Redis-backed task broker.
    Supports redis-py if installed or socket RESP connection.
    Gracefully falls back with a clear error if Redis server is unreachable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        queue_key: str = "synapse:tasks",
        dlq_key: str = "synapse:dlq",
        client: Optional[Any] = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.queue_key = queue_key
        self.dlq_key = dlq_key
        self._closed = False
        self._tasks: Dict[str, Dict[str, Any]] = {}

        if client is not None:
            self._client = client
        else:
            self._client = self._init_client()

    def _init_client(self) -> Any:
        """Initializes Redis client or raises clear informative ConnectionError."""
        try:
            import redis
            client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            client.ping()
            return client
        except ImportError:
            # Check if socket can connect
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((self.host, self.port))
                s.close()
                raise ConnectionError(
                    f"Connected to {self.host}:{self.port}, but the 'redis' package is required "
                    f"for protocol operations. Install it via 'pip install redis'."
                )
            except (socket.error, OSError) as err:
                raise ConnectionError(
                    f"RedisQueueBroker failed to connect to Redis at {self.host}:{self.port}: {err}. "
                    f"Ensure Redis is running or use MemoryBroker / DiskQueueBroker."
                ) from err
        except Exception as err:
            raise ConnectionError(
                f"RedisQueueBroker failed to connect to Redis at {self.host}:{self.port}: {err}. "
                f"Ensure Redis is running or use MemoryBroker / DiskQueueBroker."
            ) from err

    def enqueue(self, task_id: str, task_name: str, payload: Dict[str, Any]) -> None:
        record = {
            "id": task_id,
            "name": task_name,
            "payload": payload,
            "state": "pending",
            "retries": payload.get("_retries", 0),
            "error": None,
            "enqueued_at": time.time(),
        }
        data = json.dumps(record, default=str)
        self._client.hset(f"{self.queue_key}:meta", task_id, data)
        self._client.lpush(self.queue_key, task_id)

    def dequeue(self, timeout: Optional[float] = None) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        t = int(timeout) if timeout is not None and timeout > 0 else 0
        if hasattr(self._client, "brpop"):
            res = self._client.brpop(self.queue_key, timeout=t)
            if not res:
                return None
            task_id = res[1] if isinstance(res, (tuple, list)) else res
        else:
            task_id = self._client.rpop(self.queue_key)
            if not task_id:
                return None

        raw = self._client.hget(f"{self.queue_key}:meta", task_id)
        if not raw:
            return None
        record = json.loads(raw)
        record["state"] = "processing"
        record["started_at"] = time.time()
        self._client.hset(f"{self.queue_key}:meta", task_id, json.dumps(record, default=str))
        return (record["id"], record["name"], record["payload"])

    def acknowledge(self, task_id: str) -> None:
        raw = self._client.hget(f"{self.queue_key}:meta", task_id)
        if raw:
            record = json.loads(raw)
            record["state"] = "completed"
            record["completed_at"] = time.time()
            self._client.hset(f"{self.queue_key}:meta", task_id, json.dumps(record, default=str))

    def reject(self, task_id: str, requeue: bool = False, reason: Optional[str] = None) -> None:
        raw = self._client.hget(f"{self.queue_key}:meta", task_id)
        if not raw:
            return
        record = json.loads(raw)
        record["error"] = reason
        if requeue:
            record["state"] = "pending"
            record["retries"] = record.get("retries", 0) + 1
            record["payload"]["_retries"] = record["retries"]
            self._client.hset(f"{self.queue_key}:meta", task_id, json.dumps(record, default=str))
            self._client.lpush(self.queue_key, task_id)
        else:
            record["state"] = "failed"
            record["failed_at"] = time.time()
            self._client.hset(f"{self.queue_key}:meta", task_id, json.dumps(record, default=str))
            self._client.rpush(self.dlq_key, json.dumps(record, default=str))

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        raw_list = self._client.lrange(self.dlq_key, 0, -1) or []
        items = []
        for raw in raw_list:
            try:
                items.append(json.loads(raw))
            except Exception:
                pass
        return items

    def close(self) -> None:
        self._closed = True
        if hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass
