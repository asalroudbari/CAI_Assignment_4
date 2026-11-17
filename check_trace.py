"""Checks a trace with the same rules the TAs use, and prints the numbers for the report.

    PYTHONPATH=. python3 check_trace.py --trace run_output/trace.jsonl

Every record goes through voice_kit.record_problems. The percentiles are computed over the passing
records only, because a record with a missing event has no stage duration to take a percentile of.

Exit code is 0 only when there are no problems at all and the feed lag check passes; otherwise 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voice_kit import (
    read_trace,
    record_problems,
    summarize_stages,
    summarize_time_to_first_audio,
)

DEFAULT_MAX_FEED_LAG_MILLISECONDS = 5.0


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the settings of one check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("run_output/trace.jsonl"),
        help="The trace to check.",
    )
    parser.add_argument(
        "--max-feed-lag-ms",
        type=float,
        default=DEFAULT_MAX_FEED_LAG_MILLISECONDS,
        help="The most feed lag a turn may show before the run is suspect (default: 5 ms).",
    )

    return parser.parse_args(argv)


def heading(text: str) -> None:
    """Print a section heading."""
    print(f"\n{text}")
    print("-" * len(text))


def print_problems(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Print every problem of every record; return the passing records and the rejected turn ids."""
    heading("problems")

    passing_records = []
    rejected_turn_ids = []

    for record in records:
        turn_id = record.get("turn_id", "<no turn_id>")
        problems = record_problems(record)

        if problems:
            rejected_turn_ids.append(turn_id)

            for problem in problems:
                print(f"{turn_id}: {problem}")

        else:
            passing_records.append(record)

    if not rejected_turn_ids:
        print("none")

    return passing_records, rejected_turn_ids


def print_summary(
    records: list[dict], passing_records: list[dict], rejected_turn_ids: list[str]
) -> None:
    """Print how many turns there are and how many of them count."""
    heading("summary")
    print(f"records in trace   {len(records)}")
    print(f"passing            {len(passing_records)}")
    print(f"rejected           {len(rejected_turn_ids)}")

    if rejected_turn_ids:
        print(f"rejected turns     {', '.join(rejected_turn_ids)}")


def print_time_to_first_audio(passing_records: list[dict]) -> None:
    """Print the turn count and the two percentiles of the headline number."""
    heading("time to first audio (passing turns only)")

    if not passing_records:
        print("no passing turns, so there is nothing to summarize")
        return

    summary = summarize_time_to_first_audio(passing_records)
    print(f"turn_count         {summary['turn_count']:.0f}")
    print(f"p50_milliseconds   {summary['p50_milliseconds']:.1f}")
    print(f"p95_milliseconds   {summary['p95_milliseconds']:.1f}")


def print_stages(passing_records: list[dict]) -> None:
    """Print the p50 and p95 of every stage of the waterfall."""
    heading("stages (passing turns only)")

    if not passing_records:
        print("no passing turns, so there is nothing to summarize")
        return

    summary = summarize_stages(passing_records)
    print(f"{'stage':<20}{'p50 ms':>10}{'p95 ms':>10}")

    for name, percentiles in summary.items():
        print(
            f"{name:<20}"
            f"{percentiles['p50_milliseconds']:>10.1f}"
            f"{percentiles['p95_milliseconds']:>10.1f}"
        )


def print_feed_lag(records: list[dict], limit_milliseconds: float) -> bool:
    """Print the worst feed lag of the run; return whether every turn stayed under the limit."""
    heading(f"feed lag (limit {limit_milliseconds} ms)")

    lags = []

    for record in records:
        lag = record.get("extras", {}).get("max_feed_lag_milliseconds")

        if isinstance(lag, (int, float)) and not isinstance(lag, bool):
            lags.append((record.get("turn_id", "<no turn_id>"), float(lag)))

    if not lags:
        print("FAIL  no turn reports extras.max_feed_lag_milliseconds")
        return False

    worst_turn_id, worst_lag = max(lags, key=lambda item: item[1])
    over_limit = [item for item in lags if item[1] >= limit_milliseconds]

    print(f"worst              {worst_lag:.3f} ms on {worst_turn_id}")

    for turn_id, lag in over_limit:
        print(f"over limit         {turn_id}: {lag:.3f} ms")

    if over_limit:
        print(f"FAIL  {len(over_limit)} turn(s) at or over {limit_milliseconds} ms")
        return False

    print(f"PASS  every turn under {limit_milliseconds} ms")

    return True


def print_stage_implementations(records: list[dict]) -> None:
    """Print which code produced this trace, so a fake run can never pass for a real measurement."""
    heading("stage implementations")

    seen: dict[str, int] = {}

    for record in records:
        implementations = record.get("extras", {}).get("stage_implementations")
        key = json.dumps(implementations, sort_keys=True) if implementations else "not recorded"
        seen[key] = seen.get(key, 0) + 1

    for key, count in seen.items():
        print(f"{count:>4} turn(s)  {key}")

    if any("Fake" in key for key in seen):
        print("\n*** THIS TRACE WAS PRODUCED WITH FAKE STAGES — NOT A REAL MEASUREMENT ***")


def main(argv: list[str] | None = None) -> int:
    """Check a trace and print everything the report needs from it."""
    arguments = parse_arguments(argv)

    if not arguments.trace.exists():
        raise SystemExit(f"No trace at {arguments.trace}. Run run_agent.py first.")

    records = read_trace(arguments.trace)

    if not records:
        raise SystemExit(f"{arguments.trace} holds no records.")

    print(f"trace: {arguments.trace}")

    passing_records, rejected_turn_ids = print_problems(records)
    print_summary(records, passing_records, rejected_turn_ids)
    print_time_to_first_audio(passing_records)
    print_stages(passing_records)
    feed_lag_passed = print_feed_lag(records, arguments.max_feed_lag_ms)
    print_stage_implementations(records)

    everything_passed = not rejected_turn_ids and feed_lag_passed

    heading("verdict")
    print("PASS" if everything_passed else "FAIL")

    return 0 if everything_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
