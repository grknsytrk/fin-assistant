from app import cache as cache_module
from app.rate_limit import SharedRedisSlidingWindowRateLimiter, SlidingWindowRateLimiter


def test_sliding_window_limiter_returns_retry_after_without_resetting_the_window() -> None:
    limiter = SlidingWindowRateLimiter()

    assert limiter.consume("client", limit=2)[0] is True
    assert limiter.consume("client", limit=2)[0] is True
    allowed, retry_after = limiter.consume("client", limit=2)

    assert allowed is False
    assert retry_after >= 1


def test_shared_redis_limiter_uses_atomic_cache_window(monkeypatch) -> None:
    calls = []

    class FakeClient:
        def eval(self, script, key_count, key, *args):
            calls.append((script, key_count, key, args))
            return [0, 7]

    class FakeRedisCache:
        name = "redis"
        _client = FakeClient()

        @staticmethod
        def _prefixed(key: str) -> str:
            return f"ragfin:{key}"

    monkeypatch.setattr(cache_module, "get_cache", lambda: FakeRedisCache())
    result = SharedRedisSlidingWindowRateLimiter().consume("client:GET:/market/fx:127.0.0.1", limit=2)

    assert result == (False, 7)
    assert calls[0][1] == 1
    assert calls[0][2].startswith("ragfin:rate-limit:v1:")
