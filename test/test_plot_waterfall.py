"""Sanity check: the waterfall figure draws exactly the numbers the TA checker computes.

    python3 test/test_plot_waterfall.py

Builds a three-turn trace by hand, so every expected value can be checked with a pencil:

    turn   asr   llm_start   speech_start   ttfa   llm_remaining
    1      40    300         100            440    200
    2      50    400         120            570    300
    3      60    500         140            700    400

p50 is the middle turn (linear interpolation over 3 values), so p50 asr = 50, llm_start = 400,
speech_start = 120, ttfa = 570. A fourth turn has no llm_first_token, so it must be rejected and
must not move any number. Needs matplotlib.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plot_waterfall  # noqa: E402

MS = 1_000_000


def make_record(turn: int, asr: int, llm_start: int, speech_start: int, remaining: int) -> dict:
    """Return one trace record whose stages last the given milliseconds."""
    end_of_speech = 10_000 * MS * turn
    asr_final = end_of_speech + asr * MS
    first_token = asr_final + llm_start * MS
    return {
        "turn_id": f"turn_{turn:04d}",
        "scenario_node": "test",
        "clock": "monotonic_ns",
        "events": {
            "end_of_user_speech": end_of_speech,
            "asr_final": asr_final,
            "llm_first_token": first_token,
            "llm_complete": first_token + remaining * MS,
            "tts_first_audio": first_token + speech_start * MS,
        },
        "extras": {
            "recording": f"test_{turn}.wav",
            "max_feed_lag_milliseconds": 0.1,
            "transcript": "hello",
            "reply": "hi",
        },
    }


def main() -> int:
    records = [
        make_record(1, 40, 300, 100, 200),
        make_record(2, 50, 400, 120, 300),
        make_record(3, 60, 500, 140, 400),
    ]
    broken = make_record(4, 999, 999, 999, 999)
    del broken["events"]["llm_first_token"]
    records.append(broken)

    numbers = plot_waterfall.waterfall_numbers(records)

    expected_p50 = {"asr": 50.0, "llm_start": 400.0, "speech_start": 120.0, "llm_remaining": 300.0}
    assert numbers["turn_count"] == 3, numbers["turn_count"]
    assert numbers["rejected_count"] == 1, numbers["rejected_count"]
    assert abs(numbers["ttfa_p50"] - 570.0) < 1e-6, numbers["ttfa_p50"]
    for stage, value in expected_p50.items():
        got = numbers["stages"][stage]["p50_milliseconds"]
        assert abs(got - value) < 1e-6, (stage, got, value)

    # p95 over 3 values: position 0.95 * 2 = 1.9, so 10% of the middle plus 90% of the largest.
    assert abs(numbers["stages"]["asr"]["p95_milliseconds"] - 59.0) < 1e-6
    assert abs(numbers["ttfa_p95"] - 687.0) < 1e-6, numbers["ttfa_p95"]

    # The LLM start parts: one turn carries the times inside LLM start (400 ms here), split
    # 60 + 150 + 40 + 100 + 50; the parts must add up to that turn's LLM start exactly.
    split = make_record(5, 50, 400, 120, 300)
    asr_final = split["events"]["asr_final"]
    for name, after in (("intent_first_reasoning", 60), ("intent_first_json", 210), ("intent_done", 250),
                        ("first_reasoning_token", 350)):
        split["extras"][f"{name}_nanoseconds"] = asr_final + after * 1_000_000
    parts = plot_waterfall.llm_start_parts([split])
    got = {key: parts[key]["p50_milliseconds"] for key, *_ in plot_waterfall.LLM_START_PARTS}
    assert got == {"intent_wait": 60.0, "intent_reasoning": 150.0, "intent_json": 40.0,
                   "act_and_prefill": 100.0, "reply_reasoning": 50.0}, got
    assert sum(got.values()) == 400.0
    numbers["llm_start_parts"] = parts

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "waterfall.png"
        plot_waterfall.draw(numbers, output, None)
        assert output.exists() and output.stat().st_size > 1000

    print("PASS  turn count 3 (1 rejected), TTFA p50 570.0 ms, p95 687.0 ms, stage p50s",
          {stage: numbers["stages"][stage]["p50_milliseconds"] for stage in expected_p50})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
