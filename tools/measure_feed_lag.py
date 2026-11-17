"""Measures how late the shim's feed thread wakes, and what the interpreter switch interval does to it.

    PYTHONPATH=. python3 tools/measure_feed_lag.py

WHAT IT IS FOR. ``extras.max_feed_lag_milliseconds`` is how the assignment proves the agent is not
distorting its own input: the shim feeds a recording in 20 ms chunks paced at real time, and if the
agent keeps the feed thread from waking on time, the audio it is measuring arrives late. The turn
loop sets ``sys.setswitchinterval`` to keep that from happening, and this is the measurement behind
that line, kept as a tool so the number in the report can be reproduced rather than asserted.

THE MECHANISM. The feed thread sleeps until its deadline and then needs the interpreter lock to
deliver the chunk. ``sys.setswitchinterval`` is how long ANOTHER thread may hold that lock before
it is asked to let go, so a main thread running Python code can hold the feed thread off for up to
one whole switch interval. The default is 5 ms, which is the same size as the budget the checker
allows for the entire lag.

That only bites when the main thread is running PYTHON. Time inside a native library that releases
the lock — which is where a faster-whisper decode spends almost all of its time — does not hold the
feed thread off this way, which is why the two cases below are measured separately. Comparing them
is the point: the difference between them IS the interpreter lock's contribution.
"""

from __future__ import annotations

import argparse
import sys
import threading

from voice_kit.audio import DEFAULT_CHUNK_MILLISECONDS
from voice_kit.clock import now_nanoseconds
from voice_kit.source import sleep_until

NANOSECONDS_PER_MILLISECOND = 1_000_000

# How often the busy main thread looks up to see whether the feed has finished. Small enough to stop
# promptly, large enough that the check is not itself most of the work.
BUSY_LOOP_BATCH = 2_000


def percentile_of(sorted_values: list[float], fraction: float) -> float:
    """Return a percentile of an already sorted list, by nearest rank."""
    index = min(int(fraction * len(sorted_values)), len(sorted_values) - 1)

    return sorted_values[index]


def measure(
    switch_interval_seconds: float,
    busy_main_thread: bool,
    chunk_milliseconds: int,
    chunk_count: int,
) -> list[float]:
    """Return the wake-up lag of every chunk, in milliseconds, under one condition.

    The feed thread does exactly what voice_kit.source.ReplaySource does — same clock, same
    sleep_until, same 20 ms deadlines — so the lag measured here is the lag that lands in the trace.
    """
    sys.setswitchinterval(switch_interval_seconds)

    lags: list[float] = []
    finished = threading.Event()

    def feed() -> None:
        start_nanoseconds = now_nanoseconds()

        for index in range(1, chunk_count + 1):
            deadline = start_nanoseconds + index * chunk_milliseconds * NANOSECONDS_PER_MILLISECOND
            sleep_until(deadline)
            lags.append((now_nanoseconds() - deadline) / NANOSECONDS_PER_MILLISECOND)

        finished.set()

    thread = threading.Thread(target=feed)
    thread.start()

    if busy_main_thread:
        # Pure Python arithmetic on purpose: it holds the interpreter lock, which is the whole thing
        # being measured. A numpy or a CTranslate2 workload would release the lock and show nothing.
        counter = 0

        while not finished.is_set():
            for _ in range(BUSY_LOOP_BATCH):
                counter += 1

    thread.join()

    return lags


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the settings of one measurement."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--chunks",
        type=int,
        default=250,
        help="How many 20 ms chunks to pace per condition (default: 250, which is 5 seconds).",
    )
    parser.add_argument(
        "--switch-intervals",
        type=float,
        nargs="+",
        default=[0.005, 0.0005],
        help="Switch intervals to compare, in seconds (default: 0.005 0.0005).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Measure every combination of switch interval and main-thread state, and print the table."""
    arguments = parse_arguments(argv)

    print(
        f"{arguments.chunks} chunks of {DEFAULT_CHUNK_MILLISECONDS} ms per condition "
        f"({arguments.chunks * DEFAULT_CHUNK_MILLISECONDS / 1000:.0f} s each)"
    )
    print("")
    print(f"{'main thread':<14}{'switch interval':>17}{'median':>10}{'p95':>10}{'worst':>10}")
    print("-" * 61)

    for busy in (False, True):
        for switch_interval in arguments.switch_intervals:
            lags = sorted(
                measure(
                    switch_interval, busy, DEFAULT_CHUNK_MILLISECONDS, arguments.chunks
                )
            )

            print(
                f"{'busy (Python)' if busy else 'idle':<14}"
                f"{switch_interval * 1000:>14.1f} ms"
                f"{percentile_of(lags, 0.5):>9.2f} ms"
                f"{percentile_of(lags, 0.95):>9.2f} ms"
                f"{lags[-1]:>9.2f} ms"
            )

    print("")
    print(
        "The checker fails a turn at 5 ms. Read the two 'busy' rows against each other: that\n"
        "difference is what the switch interval buys, and it is only ever paid by Python work on\n"
        "the main thread."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
