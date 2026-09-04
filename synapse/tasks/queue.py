"""
Synapse Asynchronous Task Queue
===============================
High-performance, multi-worker task execution queue featuring:
- Pluggable backend brokers (MemoryBroker, DiskQueueBroker with WAL, RedisQueueBroker)
- Thread-safe worker pool bypassing GIL bottlenecks for I/O and compiled native operations
- @queue.task decorator with seamless .delay(*args, **kwargs) invocation
- Full-lifecycle TaskResult (PENDING, RUNNING, SUCCESS, FAILED) with .wait(timeout)
- Automatic retries, rejection, and Dead Letter Queue (DLQ) routing
- Graceful shutdown and execution monitoring
"""

from enum import Enum
import functools
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

from .broker import BaseBroker, MemoryBroker


# =============================================================================
# Global Task Registry
# =============================================================================

_GLOBAL_TASK_REGISTRY: Dict[str, Callable] = {}


# =============================================================================
# 1. Task Status Enum
# =============================================================================

class TaskStatus(str, Enum):
    """Execution status for tasks within the queue."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# 2. Task Result
# =============================================================================

class TaskResult:
    """Handle representing an asynchronous task in progress or completed."""

    def __init__(self, task_id: str):
        self.id = task_id
        self.status: TaskStatus = TaskStatus.PENDING
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self._event = threading.Event()

    def wait(self, timeout: Optional[float] = None, raise_on_error: bool = False) -> Any:
        """
        Blocks until the task completes or timeout is exceeded.
        Returns the task result if successful.
        Raises TimeoutError if timeout expires.
        Raises the underlying task error if raise_on_error is True.
        """
        completed = self._event.wait(timeout=timeout)
        if not completed:
            raise TimeoutError(f"Task '{self.id}' timed out after {timeout} seconds.")

        if raise_on_error and self.status == TaskStatus.FAILED and self.error:
            raise self.error

        return self.result

    @property
    def is_ready(self) -> bool:
        """True if task has finished executing (either SUCCESS or FAILED)."""
        return self.status in (TaskStatus.SUCCESS, TaskStatus.FAILED)

    @property
    def is_successful(self) -> bool:
        """True if task succeeded."""
        return self.status == TaskStatus.SUCCESS

    @property
    def is_failed(self) -> bool:
        """True if task failed with an exception."""
        return self.status == TaskStatus.FAILED

    @property
    def execution_time(self) -> Optional[float]:
        """Elapsed execution time in seconds, or None if not finished."""
        if self.started_at is not None and self.completed_at is not None:
            return self.completed_at - self.started_at
        return None

    def __repr__(self) -> str:
        return (
            f"<TaskResult id={self.id!r} status={self.status} "
            f"result={self.result!r} error={self.error!r}>"
        )


# =============================================================================
# 3. Task Wrapper
# =============================================================================

class TaskWrapper:
    """Wraps a target function to provide synchronous calls and .delay() asynchronous execution."""

    def __init__(
        self,
        queue_instance: "TaskQueue",
        fn: Callable,
        name: Optional[str] = None,
        **options: Any,
    ):
        self.queue = queue_instance
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "unnamed_task")
        self.options = options
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the function synchronously."""
        return self.fn(*args, **kwargs)

    def delay(self, *args: Any, **kwargs: Any) -> TaskResult:
        """Dispatches the function to the TaskQueue for asynchronous execution."""
        return self.queue.delay(self.fn, *args, **kwargs)

    def __repr__(self) -> str:
        return f"<TaskWrapper {self.name}>"


# =============================================================================
# 4. Task Queue
# =============================================================================

class TaskQueue:
    """Thread-safe multi-worker task execution queue backed by pluggable brokers."""

    def __init__(
        self,
        num_workers: int = 2,
        broker: Optional[BaseBroker] = None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        name: str = "default",
    ):
        self.num_workers = max(1, int(num_workers))
        self.name = name
        self.max_retries = max(0, int(max_retries))
        self.retry_delay = float(retry_delay)
        self.broker: BaseBroker = broker if broker is not None else MemoryBroker()
        self._tasks: Dict[str, TaskResult] = {}
        self._registry: Dict[str, Callable] = {}
        self._workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        self._start_workers()

    @property
    def _queue(self) -> Any:
        """Backward compatibility shim for code referencing internal _queue."""
        if hasattr(self.broker, "_queue"):
            return getattr(self.broker, "_queue")
        return None

    def _start_workers(self) -> None:
        """Initializes and starts daemon worker threads."""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"SynapseWorker-{self.name}-{i+1}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)

    def register_task(self, name: str, fn: Callable) -> None:
        """Registers a function in the queue's and global task registries."""
        with self._lock:
            self._registry[name] = fn
        _GLOBAL_TASK_REGISTRY[name] = fn

    def task(self, fn_or_name: Union[Callable, str, None] = None, **options: Any):
        """
        Decorator to register a task with support for .delay().
        Usage:
            @queue.task
            def my_job(x): ...

            @queue.task(name="custom")
            def my_job2(x): ...
        """
        if callable(fn_or_name):
            task_name = getattr(fn_or_name, "__name__", "unnamed_task")
            self.register_task(task_name, fn_or_name)
            return TaskWrapper(self, fn_or_name, name=task_name, **options)

        def decorator(fn: Callable) -> TaskWrapper:
            task_name = fn_or_name if isinstance(fn_or_name, str) else getattr(fn, "__name__", "unnamed_task")
            self.register_task(task_name, fn)
            return TaskWrapper(self, fn, name=task_name, **options)

        return decorator

    def delay(self, fn_or_name: Union[Callable, str], *args: Any, **kwargs: Any) -> TaskResult:
        """Enqueues an arbitrary callable or registered task name with arguments for background execution."""
        if callable(fn_or_name):
            fn = fn_or_name
            task_name = getattr(fn, "__name__", "unnamed_task")
            self.register_task(task_name, fn)
        else:
            task_name = str(fn_or_name)
            fn = self._registry.get(task_name) or _GLOBAL_TASK_REGISTRY.get(task_name)

        task_id = uuid.uuid4().hex
        task_res = TaskResult(task_id)

        with self._lock:
            self._tasks[task_id] = task_res

        payload = {
            "args": list(args),
            "kwargs": dict(kwargs),
            "_retries": 0,
        }
        self.broker.enqueue(task_id=task_id, task_name=task_name, payload=payload)
        return task_res

    def get_task(self, task_id: str) -> Optional[TaskResult]:
        """Retrieves a TaskResult by its task ID."""
        with self._lock:
            if task_id in self._tasks:
                return self._tasks[task_id]

        status = self.broker.get_task_status(task_id)
        if status:
            task_res = TaskResult(task_id)
            if status == "completed":
                task_res.status = TaskStatus.SUCCESS
                task_res._event.set()
            elif status == "failed":
                task_res.status = TaskStatus.FAILED
                task_res._event.set()
            elif status == "processing":
                task_res.status = TaskStatus.RUNNING
            else:
                task_res.status = TaskStatus.PENDING
            with self._lock:
                self._tasks[task_id] = task_res
            return task_res

        return None

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Retrieves all failed tasks in the broker's dead letter queue."""
        return self.broker.get_dead_letter_queue()

    def _worker_loop(self) -> None:
        """Internal loop executed by each worker thread."""
        while not self._shutdown_event.is_set():
            try:
                item = self.broker.dequeue(timeout=0.1)
            except Exception:
                time.sleep(0.02)
                continue

            if item is None:
                continue

            task_id, task_name, payload = item

            with self._lock:
                task_res = self._tasks.get(task_id)
                if task_res is None:
                    task_res = TaskResult(task_id)
                    self._tasks[task_id] = task_res

            task_res.status = TaskStatus.RUNNING
            if task_res.started_at is None:
                task_res.started_at = time.time()

            fn = self._registry.get(task_name) or _GLOBAL_TASK_REGISTRY.get(task_name)
            if fn is None:
                exc = RuntimeError(f"Task function '{task_name}' is not registered.")
                task_res.error = exc
                task_res.status = TaskStatus.FAILED
                task_res.completed_at = time.time()
                self.broker.reject(task_id, requeue=False, reason=str(exc))
                task_res._event.set()
                continue

            args = payload.get("args", [])
            kwargs = payload.get("kwargs", {})

            try:
                result = fn(*args, **kwargs)
                task_res.result = result
                task_res.status = TaskStatus.SUCCESS
                task_res.error = None
                task_res.completed_at = time.time()
                self.broker.acknowledge(task_id)
                task_res._event.set()
            except Exception as exc:
                retries = payload.get("_retries", 0)
                if retries < self.max_retries:
                    task_res.status = TaskStatus.PENDING
                    if self.retry_delay > 0:
                        time.sleep(self.retry_delay)
                    self.broker.reject(task_id, requeue=True, reason=str(exc))
                else:
                    task_res.error = exc
                    task_res.status = TaskStatus.FAILED
                    task_res.completed_at = time.time()
                    self.broker.reject(task_id, requeue=False, reason=str(exc))
                    task_res._event.set()

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """Blocks until all tasks currently in the queue have finished."""
        deadline = (time.time() + timeout) if timeout is not None else None
        while True:
            with self._lock:
                has_active_tasks = any(
                    t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                    for t in self._tasks.values()
                )
            broker_pending = getattr(self.broker, "pending_count", 0) > 0
            if not has_active_tasks and not broker_pending:
                return True
            if deadline and time.time() > deadline:
                return False
            time.sleep(0.02)

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Gracefully signals workers to stop and cleans up broker resources."""
        self._shutdown_event.set()
        if wait:
            for worker in self._workers:
                worker.join(timeout=timeout)
        self.broker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)

    def __repr__(self) -> str:
        return (
            f"<TaskQueue name={self.name!r} workers={self.num_workers} "
            f"broker={self.broker.__class__.__name__} pending={self.broker.pending_count}>"
        )
