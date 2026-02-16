"""Rate limiting and budget primitives for parallel trace fetching.

Thread-safe utilities shared across fetch workers:
- TokenBucket: controls global request rate (tokens/sec) with burst capacity
- SharedBudget: enforces a global trace count limit across workers
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Dispenses tokens at a fixed rate. Workers call ``acquire()`` before
    each HTTP request.  When a 429 is received, any worker can call
    ``pause_until()`` to freeze *all* workers for the Retry-After period.

    Parameters
    ----------
    rate : float
        Tokens added per second.
    capacity : int
        Maximum burst size (also the initial token count).
    """

    def __init__(self, rate: float, capacity: int) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._lock = threading.Lock()
        self._rate = float(rate)
        self._capacity = int(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._paused_until: float = 0.0

    # -- internal helpers ---------------------------------------------------

    def _refill(self, now: float) -> None:
        """Add tokens accrued since last refill. Must hold ``_lock``."""
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now

    # -- public API ---------------------------------------------------------

    def acquire(self, timeout: float = 30.0) -> bool:
        """Block until a token is available or *timeout* seconds elapse.

        Returns ``True`` if a token was acquired, ``False`` on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                # Honour pause (429 propagation)
                if now < self._paused_until:
                    wait = min(self._paused_until - now, 0.05)
                else:
                    self._refill(now)
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return True
                    wait = 0.05
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(wait, max(deadline - time.monotonic(), 0)))

    def pause_until(self, resume_time: float) -> None:
        """Freeze all token acquisition until *resume_time* (monotonic clock).

        Multiple calls always advance the deadline — never retreat.
        """
        with self._lock:
            if resume_time > self._paused_until:
                self._paused_until = resume_time


class SharedBudget:
    """Thread-safe trace-count budget shared across fetch workers.

    Parameters
    ----------
    total : int
        Maximum number of traces the entire fetch may collect.
    """

    def __init__(self, total: int) -> None:
        if total < 0:
            raise ValueError(f"total must be >= 0, got {total}")
        self._lock = threading.Lock()
        self._remaining = int(total)

    def remaining(self) -> int:
        with self._lock:
            return self._remaining

    def consume(self, n: int) -> int:
        """Try to consume *n* from the budget.

        Returns the actual amount consumed (may be < *n* when budget is
        nearly exhausted, 0 when fully exhausted).
        """
        if n <= 0:
            return 0
        with self._lock:
            take = min(n, self._remaining)
            self._remaining -= take
            return take

    def exhausted(self) -> bool:
        with self._lock:
            return self._remaining <= 0
