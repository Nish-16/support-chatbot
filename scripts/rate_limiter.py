"""
Thread-safe sliding-window rate limiter.

Groq's limit on this key for openai/gpt-oss-20b is 8,000 tokens/MINUTE
(checked via the x-ratelimit-* response headers), which is the real
constraint for concurrent workers -- the 1,000 requests/day cap isn't
hit within one run. A classify_message() call (short prompt + a few
tokens of JSON reply) runs well under 400 tokens, so capping at
max_calls=15 per 60s period stays comfortably under the token budget
even with several workers in flight, while the existing 429 retry/
backoff in groq_lib absorbs any remaining burst.
"""
import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self.period - (now - self._calls[0])
            time.sleep(max(sleep_for, 0.05))
