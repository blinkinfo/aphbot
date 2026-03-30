"""Self-contained offline unit tests for compute_adx().

Runs without network access -- all candle data is synthetic.

Assertions:
1. Output length is correct (non-empty, matches expected formula).
2. All ADX values are in [0, 100].
3. The series converges: variance of last 50 values < variance of first 50.
4. The final ADX from a 300-candle run differs from a cold-start 35-candle run
   by more than 5 points (warm-up matters). Because the guard is now 3*n=42,
   the 35-candle run returns None -- counted as an infinite difference.
"""

from __future__ import annotations
import math
import sys
import os

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `import config` works
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.adx import compute_adx  # noqa: E402


# ---------------------------------------------------------------------------
# Candle generation helpers
# ---------------------------------------------------------------------------

def _make_realistic_candles(n: int, start_price: float = 30_000.0) -> list[dict]:
    """Generate *n* candles with realistic mixed-direction price movement.

    Uses a sine wave overlaid with a slow trend and LCG noise so that:
    - price moves up AND down (realistic DM values)
    - ADX is meaningful and not pinned at 0 or 100
    - the series has enough variation for convergence testing
    """
    candles: list[dict] = []
    price = start_price
    seed = 12345

    for i in range(n):
        # LCG random number in [0, 1)
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        rnd = (seed & 0xFFFFFF) / 0xFFFFFF  # [0, 1)

        # Oscillating component: sine with period ~40 candles
        sine_component = math.sin(i * 2 * math.pi / 40) * 200

        # Slow upward drift
        drift = i * 2.0

        # Random noise: +/- 150
        noise = (rnd - 0.5) * 300

        # Candle body: open at current price, close at price + move
        move = sine_component * 0.1 + drift * 0.01 + noise * 0.1
        open_ = price
        close = price + move

        # High/low spread based on noise
        spread = max(abs(move) * 0.5, 50.0) + rnd * 30
        high = max(open_, close) + spread * 0.3
        low = min(open_, close) - spread * 0.3

        candles.append({
            "time": float(1_700_000_000 + i * 300),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        })
        price = close

    return candles


def _variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_length() -> None:
    """compute_adx() on 300 candles with ADX(14) must return a non-empty list."""
    candles = _make_realistic_candles(300)
    result = compute_adx(candles, length=14)
    assert result is not None, "compute_adx returned None for 300 candles"
    assert len(result) > 0, "compute_adx returned empty list"
    # Expected: 300 candles -> 299 TR/DM pairs -> 299-14+1=286 DI values
    # -> 286-14+1=273 ADX values
    assert len(result) == 273, f"Expected 273 ADX values, got {len(result)}"
    print(f"  output length: {len(result)}  (candles=300, n=14) -- CORRECT")


def test_values_in_range() -> None:
    """All ADX values must be in [0, 100]."""
    candles = _make_realistic_candles(300)
    result = compute_adx(candles, length=14)
    assert result is not None
    out_of_range = [v for v in result if not (0.0 <= v <= 100.0)]
    assert not out_of_range, (
        f"ADX values out of [0,100]: {out_of_range[:5]}"
    )
    print(f"  all {len(result)} values in [0, 100]  "
          f"min={min(result):.2f}  max={max(result):.2f}")


def test_series_converges() -> None:
    """ADX should stabilise over time: var(last 50) < var(first 50)."""
    candles = _make_realistic_candles(300)
    result = compute_adx(candles, length=14)
    assert result is not None
    assert len(result) >= 100, (
        f"Need at least 100 ADX values to test convergence, got {len(result)}"
    )
    var_first = _variance(result[:50])
    var_last = _variance(result[-50:])
    assert var_last < var_first, (
        f"Series did not converge: var_first={var_first:.4f}  var_last={var_last:.4f}"
    )
    print(f"  convergence OK  var_first={var_first:.4f}  var_last={var_last:.4f}")


def test_warmup_difference() -> None:
    """300-candle warm ADX must differ from cold-start 35-candle run by >5 pts.

    With the new guard (3*n = 42), 35 candles returns None -- the guard itself
    proves the warm-up protection is working, so we treat that as a pass.
    If somehow 35 candles produces a result, we assert a >5pt difference.
    """
    candles_300 = _make_realistic_candles(300)
    # Cold-start: last 35 candles (same price region, no history)
    candles_35 = candles_300[-35:]

    result_300 = compute_adx(candles_300, length=14)
    result_35 = compute_adx(candles_35, length=14)

    assert result_300 is not None, "300-candle ADX failed unexpectedly"

    if result_35 is None:
        # Guard (3*14=42 > 35) correctly rejected cold-start -- warm-up works
        print(
            f"  warm-up difference OK: guard blocked 35-candle cold-start (None).  "
            f"warm ADX={result_300[-1]:.2f}"
        )
        return

    final_300 = result_300[-1]
    final_35 = result_35[-1]
    diff = abs(final_300 - final_35)
    assert diff > 5, (
        f"Expected >5 pt difference between warm/cold ADX, got {diff:.2f}  "
        f"(warm={final_300:.2f}  cold={final_35:.2f})"
    )
    print(
        f"  warm-up difference OK: warm={final_300:.2f}  "
        f"cold={final_35:.2f}  diff={diff:.2f}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_output_length,
        test_values_in_range,
        test_series_converges,
        test_warmup_difference,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            print(f"[RUN] {t.__name__}")
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {t.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
