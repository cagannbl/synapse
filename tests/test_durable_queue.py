"""
Tests for Synapse Durable & Distributed Task Queue (Faz 3)
==========================================================
Verifies:
- DiskQueueBroker: Write-Ahead Log (WAL), atomic operations, persistence across restarts
- Crash Recovery: Recovery of unacknowledged processing tasks on broker startup
- Dead Letter Queue (DLQ): Routing and tracking after max retries
- Retry Mechanism: Automatic retries with delay before DLQ routing
- Concurrency: Multi-threaded concurrent producers and consumers without corruption
- MemoryBroker: In-memory broker lifecycle and backward compatibility
- RedisQueueBroker: Graceful fallback and Redis protocol behavior
- FileLock: Zero-dependency cross-process file locking
"""

import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import unittest.mock as mock

import pytest

from synapse.tasks import (
    BaseBroker,
    DiskQueueBroker,
    FileLock,
    MemoryBroker,
    RedisQueueBroker,
    TaskQueue,
    TaskResult,
    TaskStatus,
)


# =============================================================================
# 1. DiskQueueBroker Lifecycle & Persistence Tests
# =============================================================================

def test_disk_broker_enqueue_dequeue_acknowledge(tmp_path):
    """Verify DiskQueueBroker enqueues, dequeues, transitions state, and acknowledges."""
    broker = DiskQueueBroker(storage_dir=tmp_path)

    task_id = "task-001"
    task_name = "math.square"
    payload = {"x": 8}

    broker.enqueue(task_id, task_name, payload)
    assert broker.pending_count == 1
    assert broker.get_task_status(task_id) == "pending"

    # Dequeue
    item = broker.dequeue(timeout=1.0)
    assert item is not None
    dequeued_id, dequeued_name, dequeued_payload = item
    assert dequeued_id == task_id
    assert dequeued_name == task_name
    assert dequeued_payload == payload
    assert broker.get_task_status(task_id) == "processing"

    # Acknowledge
    broker.acknowledge(task_id)
    assert broker.get_task_status(task_id) == "completed"
    assert broker.pending_count == 0

    broker.close()


def test_disk_broker_persistence_across_restart(tmp_path):
    """Verify tasks persisted to WAL are preserved and dequeued after broker restart."""
    broker1 = DiskQueueBroker(storage_dir=tmp_path)
    broker1.enqueue("task-1", "job.send_mail", {"to": "alice@synapse.tech"})
    broker1.enqueue("task-2", "job.send_mail", {"to": "bob@synapse.tech"})
    broker1.enqueue("task-3", "job.send_mail", {"to": "carol@synapse.tech"})

    # Process and ack only task-1 in first instance
    item1 = broker1.dequeue(timeout=1.0)
    assert item1 is not None and item1[0] == "task-1"
    broker1.acknowledge("task-1")
    broker1.close()

    # Start new broker instance on identical directory (simulating restart)
    broker2 = DiskQueueBroker(storage_dir=tmp_path)
    assert broker2.pending_count == 2
    assert broker2.get_task_status("task-1") == "completed"
    assert broker2.get_task_status("task-2") == "pending"
    assert broker2.get_task_status("task-3") == "pending"

    item2 = broker2.dequeue(timeout=1.0)
    assert item2 is not None and item2[0] == "task-2"
    broker2.acknowledge("task-2")

    item3 = broker2.dequeue(timeout=1.0)
    assert item3 is not None and item3[0] == "task-3"
    broker2.acknowledge("task-3")

    # Queue should now be empty
    empty_check = broker2.dequeue(timeout=0.05)
    assert empty_check is None
    assert broker2.pending_count == 0

    broker2.close()


def test_disk_broker_crash_recovery_unacknowledged_tasks(tmp_path):
    """
    Verify crash recovery:
    A task dequeued into 'processing' state without ACK or REJECT is automatically
    recovered back into 'pending' queue when a new broker starts.
    """
    broker1 = DiskQueueBroker(storage_dir=tmp_path)
    broker1.enqueue("task-crash-1", "heavy_etl", {"batch": 101})
    broker1.enqueue("task-crash-2", "heavy_etl", {"batch": 102})

    # Worker picks up task-crash-1 (now processing), and task-crash-2
    item1 = broker1.dequeue(timeout=1.0)
    assert item1 is not None and item1[0] == "task-crash-1"
    assert broker1.get_task_status("task-crash-1") == "processing"

    item2 = broker1.dequeue(timeout=1.0)
    assert item2 is not None and item2[0] == "task-crash-2"
    broker1.acknowledge("task-crash-2")  # task-2 completes normally

    # Crash! broker1 dies abruptly without acking or rejecting task-crash-1
    broker1.close()

    # Recovery: broker2 boots up
    broker2 = DiskQueueBroker(storage_dir=tmp_path)

    # task-crash-1 should have been recovered to pending!
    assert broker2.get_task_status("task-crash-1") == "pending"
    assert broker2.get_task_status("task-crash-2") == "completed"

    # task-crash-1 can now be successfully picked up and processed
    recovered_item = broker2.dequeue(timeout=1.0)
    assert recovered_item is not None
    assert recovered_item[0] == "task-crash-1"
    assert recovered_item[2]["batch"] == 101

    broker2.acknowledge("task-crash-1")
    assert broker2.get_task_status("task-crash-1") == "completed"
    broker2.close()


def test_disk_broker_reject_with_requeue(tmp_path):
    """Verify broker.reject(task_id, requeue=True) re-enqueues the task with incremented retry count."""
    broker = DiskQueueBroker(storage_dir=tmp_path)
    broker.enqueue("retry-1", "sync_data", {"shard": 4})

    # First attempt
    item = broker.dequeue(timeout=1.0)
    assert item is not None
    tid, _, payload = item
    assert payload.get("_retries", 0) == 0

    # Reject with requeue
    broker.reject(tid, requeue=True, reason="Network timeout")
    assert broker.get_task_status(tid) == "pending"

    # Second attempt: task is dequeued again with updated retry count
    item2 = broker.dequeue(timeout=1.0)
    assert item2 is not None
    assert item2[0] == "retry-1"
    assert item2[2].get("_retries") == 1

    broker.acknowledge(tid)
    broker.close()


def test_disk_broker_dead_letter_queue_on_final_reject(tmp_path):
    """Verify broker.reject(task_id, requeue=False) routes task to Dead Letter Queue (DLQ)."""
    broker = DiskQueueBroker(storage_dir=tmp_path)
    broker.enqueue("dlq-task-1", "parse_payload", {"data": "corrupt_binary"})

    item = broker.dequeue(timeout=1.0)
    assert item is not None

    # Reject permanently
    broker.reject("dlq-task-1", requeue=False, reason="Unparseable payload header")

    assert broker.get_task_status("dlq-task-1") == "failed"
    assert broker.pending_count == 0

    dlq = broker.get_dead_letter_queue()
    assert len(dlq) == 1
    assert dlq[0]["id"] == "dlq-task-1"
    assert dlq[0]["error"] == "Unparseable payload header"
    assert dlq[0]["state"] == "failed"

    broker.close()


# =============================================================================
# 2. TaskQueue Integration with DiskQueueBroker & Retries
# =============================================================================

def test_task_queue_with_disk_broker_success(tmp_path):
    """Verify TaskQueue running with DiskQueueBroker executes tasks and persists state."""
    disk_broker = DiskQueueBroker(storage_dir=tmp_path)
    queue = TaskQueue(num_workers=3, broker=disk_broker, name="disk-workers")

    @queue.task
    def multiply(a: int, b: int) -> int:
        return a * b

    try:
        results = []
        for i in range(5):
            res = multiply.delay(i, 10)
            results.append(res)

        values = [r.wait(timeout=3.0) for r in results]
        assert values == [0, 10, 20, 30, 40]
        assert all(r.status == TaskStatus.SUCCESS for r in results)
        assert queue.broker.pending_count == 0
    finally:
        queue.shutdown(wait=True)


def test_task_queue_retry_and_dlq_routing(tmp_path):
    """
    Verify TaskQueue retries a failing task up to max_retries,
    then marks it FAILED and routes it to the broker's Dead Letter Queue.
    """
    disk_broker = DiskQueueBroker(storage_dir=tmp_path)
    # 2 retries (total 3 attempts: attempt 0, retry 1, retry 2 -> DLQ)
    queue = TaskQueue(
        num_workers=2,
        broker=disk_broker,
        max_retries=2,
        retry_delay=0.05,
        name="retry-test-queue"
    )

    attempt_counts: Dict[str, int] = {"attempts": 0}

    @queue.task
    def consistently_failing_task():
        attempt_counts["attempts"] += 1
        raise RuntimeError(f"Failure on attempt {attempt_counts['attempts']}")

    try:
        task_res = consistently_failing_task.delay()

        # Wait for task to exhaust all retries and fail
        with pytest.raises(RuntimeError, match="Failure on attempt"):
            task_res.wait(timeout=4.0, raise_on_error=True)

        assert task_res.status == TaskStatus.FAILED
        assert task_res.is_failed is True
        assert attempt_counts["attempts"] == 3  # 1 initial + 2 retries

        # Check DLQ
        dlq = queue.get_dead_letter_queue()
        assert len(dlq) == 1
        assert dlq[0]["id"] == task_res.id
        assert "Failure on attempt 3" in dlq[0]["error"]
    finally:
        queue.shutdown(wait=True)


def test_task_queue_retry_eventual_success(tmp_path):
    """Verify task that fails on first attempts but succeeds within max_retries passes."""
    disk_broker = DiskQueueBroker(storage_dir=tmp_path)
    queue = TaskQueue(
        num_workers=2,
        broker=disk_broker,
        max_retries=3,
        retry_delay=0.05,
        name="flaky-task-queue"
    )

    call_count = {"count": 0}

    @queue.task
    def flaky_service_call() -> str:
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise ConnectionResetError("Transient network failure")
        return "Connection established"

    try:
        task_res = flaky_service_call.delay()
        result = task_res.wait(timeout=4.0)

        assert result == "Connection established"
        assert task_res.status == TaskStatus.SUCCESS
        assert call_count["count"] == 3

        # DLQ must be empty because task ultimately succeeded
        assert len(queue.get_dead_letter_queue()) == 0
    finally:
        queue.shutdown(wait=True)


# =============================================================================
# 3. Concurrency & Stress Tests
# =============================================================================

def test_disk_broker_concurrent_producers_and_consumers(tmp_path):
    """
    Stress test: Concurrent producer threads enqueue tasks while concurrent consumer
    threads dequeue and acknowledge tasks using DiskQueueBroker.
    Verifies zero data loss, atomic file locking, and WAL integrity.
    """
    broker = DiskQueueBroker(storage_dir=tmp_path)
    num_producers = 4
    tasks_per_producer = 25
    total_tasks = num_producers * tasks_per_producer

    consumed_task_ids = set()
    lock = threading.Lock()

    def producer_worker(producer_idx: int):
        for j in range(tasks_per_producer):
            task_id = f"p{producer_idx}-task-{j}"
            broker.enqueue(task_id, "compute_hash", {"producer": producer_idx, "seq": j})

    def consumer_worker():
        while True:
            item = broker.dequeue(timeout=0.1)
            if item is None:
                with lock:
                    if len(consumed_task_ids) >= total_tasks:
                        break
                continue

            tid, _, _ = item
            with lock:
                consumed_task_ids.add(tid)
            broker.acknowledge(tid)

    producer_threads = [
        threading.Thread(target=producer_worker, args=(i,))
        for i in range(num_producers)
    ]
    consumer_threads = [
        threading.Thread(target=consumer_worker)
        for _ in range(4)
    ]

    for t in producer_threads:
        t.start()
    for t in consumer_threads:
        t.start()

    for t in producer_threads:
        t.join(timeout=10.0)
    for t in consumer_threads:
        t.join(timeout=10.0)

    broker.close()

    assert len(consumed_task_ids) == total_tasks
    assert broker.pending_count == 0


def test_filelock_mutual_exclusion(tmp_path):
    """Verify FileLock enforces mutual exclusion and times out on conflict."""
    lock_file = tmp_path / "test.lock"
    lock1 = FileLock(lock_file, timeout=1.0)
    lock2 = FileLock(lock_file, timeout=0.1)

    assert lock1.acquire() is True
    assert lock_file.exists()

    # Second lock should fail with TimeoutError
    with pytest.raises(TimeoutError):
        lock2.acquire()

    lock1.release()
    assert not lock_file.exists()

    # Now lock2 should successfully acquire
    assert lock2.acquire() is True
    lock2.release()


# =============================================================================
# 4. MemoryBroker & RedisQueueBroker Tests
# =============================================================================

def test_memory_broker_full_lifecycle():
    """Verify MemoryBroker basic operations, retries, and DLQ."""
    broker = MemoryBroker()

    broker.enqueue("mem-1", "task.test", {"data": 42})
    assert broker.pending_count == 1
    assert broker.get_task_status("mem-1") == "pending"

    item = broker.dequeue(timeout=0.5)
    assert item is not None and item[0] == "mem-1"
    assert broker.get_task_status("mem-1") == "processing"

    # Reject with requeue
    broker.reject("mem-1", requeue=True, reason="retry later")
    assert broker.get_task_status("mem-1") == "pending"

    item2 = broker.dequeue(timeout=0.5)
    assert item2 is not None and item2[0] == "mem-1"

    # Reject permanently
    broker.reject("mem-1", requeue=False, reason="fatal")
    assert broker.get_task_status("mem-1") == "failed"
    assert broker.pending_count == 0

    dlq = broker.get_dead_letter_queue()
    assert len(dlq) == 1
    assert dlq[0]["id"] == "mem-1"
    assert dlq[0]["error"] == "fatal"

    broker.close()


def test_redis_broker_fallback_error():
    """Verify RedisQueueBroker raises informative ConnectionError when Redis is unreachable."""
    with pytest.raises(ConnectionError) as exc_info:
        # Port 59999 is typically unused and closed
        RedisQueueBroker(host="127.0.0.1", port=59999)

    error_msg = str(exc_info.value)
    assert "RedisQueueBroker failed to connect" in error_msg or "ConnectionRefused" in error_msg or "refused" in error_msg.lower() or "connect" in error_msg


def test_redis_broker_with_mock_client():
    """Verify RedisQueueBroker protocol methods work with an injected Redis-compatible client."""
    class MockRedis:
        def __init__(self):
            self.hash_store: Dict[str, Dict[str, str]] = {}
            self.list_store: Dict[str, List[str]] = {}

        def hset(self, key: str, field: str, value: str):
            if key not in self.hash_store:
                self.hash_store[key] = {}
            self.hash_store[key][field] = value

        def hget(self, key: str, field: str) -> Optional[str]:
            return self.hash_store.get(key, {}).get(field)

        def lpush(self, key: str, value: str):
            if key not in self.list_store:
                self.list_store[key] = []
            self.list_store[key].insert(0, value)

        def rpop(self, key: str) -> Optional[str]:
            lst = self.list_store.get(key, [])
            return lst.pop() if lst else None

        def rpush(self, key: str, value: str):
            if key not in self.list_store:
                self.list_store[key] = []
            self.list_store[key].append(value)

        def lrange(self, key: str, start: int, end: int) -> List[str]:
            return list(self.list_store.get(key, []))

    mock_client = MockRedis()
    broker = RedisQueueBroker(client=mock_client)

    broker.enqueue("r-1", "redis.task", {"val": "hello"})
    dequeued = broker.dequeue(timeout=0.1)
    assert dequeued is not None
    assert dequeued[0] == "r-1"
    assert dequeued[2]["val"] == "hello"

    broker.acknowledge("r-1")

    # Reject another task to DLQ
    broker.enqueue("r-2", "redis.task2", {})
    d2 = broker.dequeue(timeout=0.1)
    broker.reject("r-2", requeue=False, reason="Redis DLQ test")

    dlq = broker.get_dead_letter_queue()
    assert len(dlq) == 1
    assert dlq[0]["id"] == "r-2"
    assert dlq[0]["error"] == "Redis DLQ test"


def test_disk_broker_wal_corruption_recovery(tmp_path):
    """Verify DiskQueueBroker skips corrupted WAL lines without crashing on startup."""
    wal_file = tmp_path / "wal.jsonl"
    # Write valid entry, corrupted partial JSON, and another valid entry
    valid_line_1 = '{"action": "ENQUEUE", "id": "t-good-1", "name": "work", "payload": {"n": 1}, "ts": 100.0}\n'
    corrupted_line = '{"action": "ENQUEUE", "id": "t-broken", "name": "corrupt\n'
    valid_line_2 = '{"action": "ENQUEUE", "id": "t-good-2", "name": "work", "payload": {"n": 2}, "ts": 101.0}\n'

    wal_file.write_text(valid_line_1 + corrupted_line + valid_line_2, encoding="utf-8")

    broker = DiskQueueBroker(storage_dir=tmp_path)
    assert broker.pending_count == 2
    assert broker.get_task_status("t-good-1") == "pending"
    assert broker.get_task_status("t-good-2") == "pending"

    item1 = broker.dequeue(timeout=0.5)
    item2 = broker.dequeue(timeout=0.5)
    assert item1 is not None and item1[0] == "t-good-1"
    assert item2 is not None and item2[0] == "t-good-2"

    broker.close()


def test_disk_broker_empty_dequeue_timeout(tmp_path):
    """Verify dequeue returns None cleanly when timeout expires on empty queue."""
    broker = DiskQueueBroker(storage_dir=tmp_path)
    start = time.time()
    item = broker.dequeue(timeout=0.1)
    elapsed = time.time() - start

    assert item is None
    assert 0.08 <= elapsed <= 0.5
    broker.close()


def test_unregistered_task_rejection_to_dlq(tmp_path):
    """Verify dispatching an unregistered task name fails cleanly and routes to DLQ."""
    disk_broker = DiskQueueBroker(storage_dir=tmp_path)
    queue = TaskQueue(num_workers=1, broker=disk_broker, name="unregistered-test")

    try:
        task_res = queue.delay("non_existent_task_handler", x=123)
        with pytest.raises(RuntimeError, match="not registered"):
            task_res.wait(timeout=2.0, raise_on_error=True)

        assert task_res.status == TaskStatus.FAILED
        dlq = queue.get_dead_letter_queue()
        assert len(dlq) == 1
        assert "not registered" in dlq[0]["error"]
    finally:
        queue.shutdown(wait=True)


def test_task_queue_default_memory_broker_backward_compat():
    """Verify TaskQueue instantiated without arguments defaults to MemoryBroker."""
    queue = TaskQueue()
    try:
        assert isinstance(queue.broker, MemoryBroker)
        assert queue.num_workers == 2
        assert queue.max_retries == 3
        assert queue.retry_delay == 0.5

        @queue.task
        def echo(val):
            return val

        res = echo.delay("synapse")
        assert res.wait(timeout=2.0) == "synapse"
    finally:
        queue.shutdown(wait=True)

