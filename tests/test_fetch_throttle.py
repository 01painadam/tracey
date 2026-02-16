"""Tests for utils.fetch_throttle — TokenBucket and SharedBudget."""

from __future__ import annotations

import threading
import time

import pytest

from utils.fetch_throttle import SharedBudget, TokenBucket


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


class TestTokenBucketBasic:
    def test_burst_then_throttle(self):
        """Burst capacity is dispensed instantly; further tokens throttled."""
        bucket = TokenBucket(rate=10, capacity=2)
        # Two tokens available immediately (burst)
        assert bucket.acquire(timeout=0) is True
        assert bucket.acquire(timeout=0) is True
        # Third should not be available instantly
        assert bucket.acquire(timeout=0) is False

    def test_tokens_refill_over_time(self):
        bucket = TokenBucket(rate=20, capacity=1)
        assert bucket.acquire(timeout=0) is True  # drain it
        assert bucket.acquire(timeout=0) is False
        time.sleep(0.1)  # 20/s → ~2 tokens in 0.1s
        assert bucket.acquire(timeout=0) is True

    def test_acquire_timeout_returns_false(self):
        bucket = TokenBucket(rate=1, capacity=1)
        bucket.acquire(timeout=0)  # drain
        t0 = time.monotonic()
        result = bucket.acquire(timeout=0.15)
        elapsed = time.monotonic() - t0
        # At 1/s we need 1s to refill — 0.15s is not enough
        assert result is False
        assert elapsed < 0.5  # didn't hang

    def test_acquire_blocks_until_token(self):
        bucket = TokenBucket(rate=20, capacity=1)
        bucket.acquire(timeout=0)  # drain
        t0 = time.monotonic()
        assert bucket.acquire(timeout=1.0) is True
        elapsed = time.monotonic() - t0
        # Should take ~0.05s at 20 tokens/s
        assert elapsed < 0.3

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            TokenBucket(rate=0, capacity=1)
        with pytest.raises(ValueError):
            TokenBucket(rate=1, capacity=0)
        with pytest.raises(ValueError):
            TokenBucket(rate=-1, capacity=1)


class TestTokenBucketPause:
    def test_pause_until_blocks_acquire(self):
        bucket = TokenBucket(rate=100, capacity=5)
        pause_end = time.monotonic() + 0.3
        bucket.pause_until(pause_end)
        t0 = time.monotonic()
        assert bucket.acquire(timeout=2.0) is True
        elapsed = time.monotonic() - t0
        # Should have waited ~0.3s for the pause to lift
        assert elapsed >= 0.25

    def test_pause_until_takes_max(self):
        """Multiple pause_until calls keep the later deadline."""
        bucket = TokenBucket(rate=100, capacity=5)
        now = time.monotonic()
        bucket.pause_until(now + 0.5)
        bucket.pause_until(now + 0.2)  # earlier — should be ignored
        t0 = time.monotonic()
        assert bucket.acquire(timeout=2.0) is True
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.4  # still waited for the 0.5s deadline

    def test_pause_until_does_not_retreat(self):
        bucket = TokenBucket(rate=100, capacity=5)
        now = time.monotonic()
        bucket.pause_until(now + 1.0)
        bucket.pause_until(now + 0.1)  # try to shorten — should be ignored
        # Verify internal state
        assert bucket._paused_until >= now + 0.9


class TestTokenBucketConcurrent:
    def test_concurrent_access_no_over_dispense(self):
        """Multiple threads sharing a bucket don't exceed the rate."""
        rate = 50.0
        bucket = TokenBucket(rate=rate, capacity=3)
        total_acquired = 0
        lock = threading.Lock()
        tokens_per_thread = 10

        def worker():
            nonlocal total_acquired
            count = 0
            for _ in range(tokens_per_thread):
                if bucket.acquire(timeout=5.0):
                    count += 1
            with lock:
                total_acquired += count

        threads = [threading.Thread(target=worker) for _ in range(5)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.monotonic() - t0

        assert total_acquired == 50
        # 50 tokens at 50/s = 1s minimum (minus burst of 3)
        assert elapsed >= 0.8  # generous margin


# ---------------------------------------------------------------------------
# SharedBudget
# ---------------------------------------------------------------------------


class TestSharedBudgetBasic:
    def test_consume_and_remaining(self):
        b = SharedBudget(total=100)
        assert b.remaining() == 100
        assert b.exhausted() is False

        taken = b.consume(30)
        assert taken == 30
        assert b.remaining() == 70

    def test_consume_more_than_remaining(self):
        b = SharedBudget(total=100)
        b.consume(70)
        taken = b.consume(80)
        assert taken == 30  # only 30 left
        assert b.remaining() == 0
        assert b.exhausted() is True

    def test_consume_zero(self):
        b = SharedBudget(total=10)
        assert b.consume(0) == 0
        assert b.remaining() == 10

    def test_consume_negative(self):
        b = SharedBudget(total=10)
        assert b.consume(-5) == 0
        assert b.remaining() == 10

    def test_exhausted_returns_true_at_zero(self):
        b = SharedBudget(total=5)
        b.consume(5)
        assert b.exhausted() is True
        assert b.consume(1) == 0

    def test_invalid_total(self):
        with pytest.raises(ValueError):
            SharedBudget(total=-1)

    def test_zero_budget(self):
        b = SharedBudget(total=0)
        assert b.exhausted() is True
        assert b.consume(5) == 0


class TestSharedBudgetConcurrent:
    def test_concurrent_consume_exact(self):
        """Total consumed across threads equals the budget."""
        budget = SharedBudget(total=100)
        totals = [0] * 5
        barrier = threading.Barrier(5)

        def worker(idx: int):
            barrier.wait()
            while True:
                taken = budget.consume(7)
                if taken == 0:
                    break
                totals[idx] += taken

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert sum(totals) == 100
        assert budget.remaining() == 0
        assert budget.exhausted() is True
