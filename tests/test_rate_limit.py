from app.rate_limit import SlidingWindowRateLimiter


def test_sliding_window_limiter_returns_retry_after_without_resetting_the_window() -> None:
    limiter = SlidingWindowRateLimiter()

    assert limiter.consume("client", limit=2)[0] is True
    assert limiter.consume("client", limit=2)[0] is True
    allowed, retry_after = limiter.consume("client", limit=2)

    assert allowed is False
    assert retry_after >= 1
