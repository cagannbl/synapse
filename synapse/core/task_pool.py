from __future__ import annotations
import collections
import concurrent.futures
import os
import queue
import threading
import time
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

_CLOSE_SENTINEL = object()


class ChannelClosed(Exception):
    """Kanal kapatıldığında ve veri kalmadığında fırlatılır."""
    pass


class Channel(Generic[T]):
    """
    Go benzeri, thread-safe CSP (Communicating Sequential Processes) kanalı.
    İş parçacıkları ve asenkron görevler arasında sıfır sürtünmeyle mesaj aktarımı sağlar.
    Kanal kapatıldığında bekleyen alıcıları (receivers) anında uyandırır (deadlock-free).
    """
    def __init__(self, capacity: int = 0):
        self.capacity = capacity
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(capacity, 1 if capacity > 0 else 0))
        self._closed = False
        self._lock = threading.Lock()

    def send(self, item: T, timeout: Optional[float] = None) -> bool:
        """Kanala veri gönderir. Kanal kapalıysa ValueError fırlatır."""
        with self._lock:
            if self._closed:
                raise ValueError("Cannot send to a closed channel")
        try:
            self._queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False

    def recv(self, timeout: Optional[float] = None) -> T:
        """Kanalda veri bekler ve çeker. Kanal kapalıysa ChannelClosed fırlatır."""
        try:
            val = self._queue.get(timeout=timeout)
            if val is _CLOSE_SENTINEL:
                # Diğer bekleyen thread'ler veya sonraki recv çağrıları için nöbetçiyi geri koy
                self._queue.put(_CLOSE_SENTINEL)
                raise ChannelClosed("Channel is closed and empty")
            return val
        except queue.Empty:
            with self._lock:
                if self._closed:
                    raise ChannelClosed("Channel is closed and empty")
            raise TimeoutError("Channel recv timed out")

    def try_recv(self) -> Optional[T]:
        """Bloklamadan anında veri çekmeyi dener."""
        try:
            val = self._queue.get_nowait()
            if val is _CLOSE_SENTINEL:
                self._queue.put(_CLOSE_SENTINEL)
                return None
            return val
        except queue.Empty:
            return None

    def close(self):
        """Kanalı yeni gönderimlere kapatır ve bekleyen tüm alıcıları uyandırır."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(_CLOSE_SENTINEL)

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def __iter__(self) -> Iterator[T]:
        while True:
            try:
                yield self.recv()
            except (ChannelClosed, TimeoutError):
                break

    def __repr__(self) -> str:
        with self._lock:
            state = "closed" if self._closed else "open"
        return f"channel(capacity={self.capacity}, state='{state}')"


class TaskFuture(Generic[T]):
    """Arka planda çalışan görevin durumunu ve sonucunu temsil eder."""
    def __init__(self, future: concurrent.futures.Future[T]):
        self._future = future

    def result(self, timeout: Optional[float] = None) -> T:
        return self._future.result(timeout=timeout)

    def done(self) -> bool:
        return self._future.done()

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        return self._future.exception(timeout=timeout)

    def __repr__(self) -> str:
        status = "done" if self._future.done() else "running"
        return f"TaskFuture(status='{status}')"


class WorkStealingPool:
    """
    Sıfır-GIL & İş Çalma (Work-Stealing) Mimarili Görev Havuzu.
    """
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or max(4, os.cpu_count() or 4)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="SynapseWorker"
        )
        self._active = True

    def spawn(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> TaskFuture[T]:
        """Yeni bir iş parçacığında görevi asenkron olarak başlatır."""
        fut = self._executor.submit(fn, *args, **kwargs)
        return TaskFuture(fut)

    def parallel_map(self, fn: Callable[[Any], T], items: Iterable[Any], max_workers: Optional[int] = None) -> list[T]:
        """Tüm CPU çekirdeklerini kullanarak paralel veri haritalama yapar."""
        executor = self._executor if max_workers is None else concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            results = list(executor.map(fn, items))
            return results
        finally:
            if executor is not self._executor:
                executor.shutdown(wait=False)

    def shutdown(self, wait: bool = True):
        self._active = False
        self._executor.shutdown(wait=wait)


# Global Default Singleton Scheduler
_GLOBAL_POOL: Optional[WorkStealingPool] = None
_POOL_LOCK = threading.Lock()


def get_global_pool() -> WorkStealingPool:
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        with _POOL_LOCK:
            if _GLOBAL_POOL is None:
                _GLOBAL_POOL = WorkStealingPool()
    return _GLOBAL_POOL


def spawn(fn: Callable[..., T], *args: Any, **kwargs: Any) -> TaskFuture[T]:
    """
    Go benzeri asenkron iş parçacığı başlatıcı:
    let fut = spawn(compute_heavy, data)
    """
    return get_global_pool().spawn(fn, *args, **kwargs)


def channel(capacity: int = 0) -> Channel[Any]:
    """
    Yerleşik CSP iletişim kanalı oluşturur:
    let ch = channel(10)
    ch.send(item)
    let val = ch.recv()
    """
    return Channel(capacity=capacity)


def parallel_map(fn: Callable[[Any], T], items: Iterable[Any], max_workers: Optional[int] = None) -> list[T]:
    """
    Çok çekirdekli paralel işleme:
    let results = parallel_map(train_epoch, mini_batches)
    """
    return get_global_pool().parallel_map(fn, items, max_workers=max_workers)


def wait_all(futures: list[TaskFuture[Any]], timeout: Optional[float] = None) -> list[Any]:
    """Verilen tüm asenkron görevlerin tamamlanmasını bekler ve sonuçları döner."""
    raw_futs = [f._future for f in futures]
    done, not_done = concurrent.futures.wait(raw_futs, timeout=timeout)
    if not_done:
        raise TimeoutError(f"{len(not_done)} tasks did not complete within timeout")
    return [f.result() for f in futures]
