"""Draws the stacked latency waterfall of a trace: the p50 and p95 of every stage up to TTFA.

    PYTHONPATH=. python3 plot_waterfall.py --trace run_output_pace/<run>/trace.jsonl \
        --output figures/waterfall.png

Only passing turns are used (voice_kit.record_problems, the check the TAs run). Every number comes
from voice_kit.summarize_stages and voice_kit.summarize_time_to_first_audio, so the figure and
check_trace.py print the same values from the same trace.

How the bars are built:
  * The TTFA path is three stages that are back to back in every turn, so per turn they add up to
    the TTFA exactly:
        asr          = asr_final       - end_of_user_speech
        llm_start    = llm_first_token - asr_final
        speech_start = tts_first_audio - llm_first_token
  * Each bar stacks the per-stage percentile (p50 row: the p50 of each stage; p95 row: the p95 of
    each stage). Percentiles do not add, so the stacked end is not the TTFA percentile. The measured
    TTFA percentile is therefore drawn as its own black tick and printed on the right.
  * llm_remaining = llm_complete - llm_first_token runs in parallel with synthesis and playback, so
    it is not on the TTFA path. It is drawn as a thin hatched bar starting at llm_first_token, under
    the stacked bar, so the overlap is visible instead of being added to the total.
  * When the trace carries the times inside the LLM start stage (--llm-nlu llm writes them in
    extras), LLM start is split into five parts that are back to back and add up to it exactly per
    turn, drawn as thin rows under it (each row: that part's percentile):
        intent_wait      = intent first reasoning token - asr_final     (state, request, prefill)
        intent_reasoning = intent first JSON token - its first reasoning token
        intent_json      = intent JSON complete - intent first JSON token
        act_and_prefill  = reply first reasoning token - intent JSON complete
                                                    (actions run in Python, reply request, prefill)
        reply_reasoning  = llm_first_token - reply first reasoning token

Needs matplotlib (not part of the agent image; any Python with matplotlib works).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from voice_kit import (  # noqa: E402
    percentile,
    read_trace,
    record_problems,
    summarize_stages,
    summarize_time_to_first_audio,
)

# Stages on the TTFA path, in time order: key, row name, event span, fill colour.
TTFA_PATH_STAGES = (
    ("asr", "ASR", "end_of_user_speech → asr_final", "#2a78d6"),
    ("llm_start", "LLM start", "asr_final → llm_first_token", "#eb6834"),
    ("speech_start", "Speech start", "llm_first_token → tts_first_audio", "#1baf7a"),
)
OFF_PATH_STAGE = ("llm_remaining", "LLM reply start", "llm_first_token → llm_complete")
OFF_PATH_COLOUR = "#898781"

# The parts of the LLM start stage: key, row name, (start, end) time, colour. A time is an event
# name or an extras key; the parts are back to back from asr_final to llm_first_token.
LLM_START_PARTS = (
    ("intent_wait", "intent call: request + prefill", ("asr_final", "intent_first_reasoning_nanoseconds"), "#f6b08e"),
    ("intent_reasoning", "intent call: reasoning", ("intent_first_reasoning_nanoseconds", "intent_first_json_nanoseconds"), "#d9541a"),
    ("intent_json", "intent call: JSON", ("intent_first_json_nanoseconds", "intent_done_nanoseconds"), "#f29061"),
    ("act_and_prefill", "actions run + reply prefill", ("intent_done_nanoseconds", "first_reasoning_token_nanoseconds"), "#f6b08e"),
    ("reply_reasoning", "reply: reasoning", ("first_reasoning_token_nanoseconds", "llm_first_token"), "#d9541a"),
)
TTFA_COLOUR = "#3d3c38"
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#6b6a64"
GRID = "#e4e3dd"
BUDGET_MILLISECONDS = 1500.0

def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the settings of one plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", type=Path, required=True, help="The trace.jsonl to plot.")
    parser.add_argument(
        "--output", type=Path, default=Path("figures/waterfall.png"), help="PNG (or PDF/SVG) path."
    )
    parser.add_argument("--title", default=None, help="Figure title (default: built from the trace).")
    return parser.parse_args(argv)


def passing_records(records: list[dict]) -> list[dict]:
    """Return the records with no problem, the only ones that count."""
    return [record for record in records if not record_problems(record)]


def waterfall_numbers(records: list[dict]) -> dict:
    """Return every number the figure draws, from the passing records of a trace."""
    passing = passing_records(records)

    if not passing:
        raise SystemExit("No passing turns in the trace, so there is nothing to plot.")

    stages = summarize_stages(passing)
    ttfa = summarize_time_to_first_audio(passing)

    return {
        "turn_count": len(passing),
        "rejected_count": len(records) - len(passing),
        "stages": stages,
        "ttfa_p50": ttfa["p50_milliseconds"],
        "ttfa_p95": ttfa["p95_milliseconds"],
        "llm_start_parts": llm_start_parts(passing),
    }


def llm_start_parts(records: list[dict]) -> dict | None:
    """Return the p50 and p95 of each part of the LLM start stage, or None if the trace lacks them.

    Only turns that carry every time are used (a turn whose intent read fell back to the rules has
    none); ``turn_count`` says how many.
    """
    durations: dict[str, list[float]] = {key: [] for key, *_ in LLM_START_PARTS}

    for record in records:
        times = {**record["events"], **record.get("extras", {})}
        names = [name for _, _, span, _ in LLM_START_PARTS for name in span]

        if not all(isinstance(times.get(name), int) for name in names):
            continue

        for key, _, (start, end), _ in LLM_START_PARTS:
            durations[key].append((times[end] - times[start]) / 1e6)

    count = len(durations[LLM_START_PARTS[0][0]])

    if not count:
        return None

    return {
        "turn_count": count,
        **{
            key: {"p50_milliseconds": percentile(values, 0.5), "p95_milliseconds": percentile(values, 0.95)}
            for key, values in durations.items()
        },
    }


def draw(numbers: dict, output: Path, title: str | None) -> None:
    """Draw one waterfall panel per percentile (p50 on top, p95 below) and write the figure.

    Each panel has one row per stage, named on the left. A stage's bar starts where the previous
    stage's bar ends, so the TTFA-path bars step down and to the right like a waterfall; the last
    row is the measured TTFA percentile from 0. The LLM's remaining generation is hatched and
    starts at llm_first_token, so its overlap with speech start is visible.
    """
    parts = numbers.get("llm_start_parts")
    part_rows = [(name, "") for _, name, _, _ in LLM_START_PARTS] if parts else []
    rows = [(name, span) for _, name, span, _ in TTFA_PATH_STAGES]
    rows[2:2] = part_rows  # the parts sit right under the LLM start row
    rows.append((OFF_PATH_STAGE[1], OFF_PATH_STAGE[2]))
    rows.append(("TTFA (measured)", "end_of_user_speech → tts_first_audio"))
    path_rows = [0, 1, 2 + len(part_rows)]  # rows of ASR, LLM start, speech start

    panels = (("p50", "p50_milliseconds", numbers["ttfa_p50"]),
              ("p95", "p95_milliseconds", numbers["ttfa_p95"]))

    longest = 0.0
    for _, key, measured in panels:
        steps = sum(numbers["stages"][stage][key] for stage, *_ in TTFA_PATH_STAGES)
        start_remaining = numbers["stages"]["asr"][key] + numbers["stages"]["llm_start"][key]
        longest = max(longest, steps, measured, start_remaining + numbers["stages"]["llm_remaining"][key])
    x_max = max(longest * 1.15, 300.0)

    figure, all_axes = plt.subplots(2, 1, figsize=(8.5, 6.4 + 1.6 * bool(part_rows)), dpi=200, sharex=True)
    bar_height = 0.62

    for axes, (panel_name, key, measured) in zip(all_axes, panels):
        left = 0.0
        previous_row = None
        for row, (stage, _, _, colour) in zip(path_rows, TTFA_PATH_STAGES):
            width = numbers["stages"][stage][key]
            axes.barh(row, width, left=left, height=bar_height, color=colour)
            axes.text(left + width + 6, row, f"{width:.1f} ms", ha="left", va="center",
                      color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
            if previous_row is not None:
                axes.plot([left, left], [previous_row + bar_height / 2, row - bar_height / 2],
                          color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")

            if stage == "llm_start" and parts:
                # Each part's own percentile, stepped from where LLM start begins. Percentiles do not
                # add, so the steps need not end exactly where the LLM start bar ends.
                part_left = left
                for offset, (part, _, _, part_colour) in enumerate(LLM_START_PARTS, start=1):
                    part_width = parts[part][key]
                    axes.barh(row + offset, part_width, left=part_left, height=bar_height * 0.55,
                              color=part_colour)
                    axes.text(part_left + part_width + 6, row + offset, f"{part_width:.1f} ms",
                              ha="left", va="center", color=TEXT_SECONDARY, fontsize=8)
                    part_left += part_width

            previous_row = row
            left += width

        remaining_row = path_rows[-1] + 1
        remaining_start = numbers["stages"]["asr"][key] + numbers["stages"]["llm_start"][key]
        remaining = numbers["stages"]["llm_remaining"][key]
        axes.barh(remaining_row, remaining, left=remaining_start, height=bar_height * 0.7,
                  color="white", edgecolor=OFF_PATH_COLOUR, hatch="////", linewidth=0.8)
        axes.text(remaining_start + remaining + 6, remaining_row, f"{remaining:.1f} ms",
                  ha="left", va="center", color=TEXT_SECONDARY, fontsize=8.5)

        ttfa_row = remaining_row + 1
        axes.barh(ttfa_row, measured, left=0, height=bar_height, color=TTFA_COLOUR)
        axes.text(measured + 6, ttfa_row, f"{measured:.1f} ms", ha="left", va="center",
                  color=TEXT_PRIMARY, fontsize=9.5, fontweight="bold")
        axes.axvline(measured, color=TTFA_COLOUR, linewidth=0.8, linestyle="--", alpha=0.6)

        # Row names drawn by hand: a stage's name in bold with its event span under it in plain
        # text, and the parts of LLM start in small grey type.
        axes.set_yticks(range(len(rows)))
        axes.set_yticklabels([""] * len(rows))
        label_axis = axes.get_yaxis_transform()

        for index, (name, span) in enumerate(rows):
            is_part = 1 < index < path_rows[-1]

            if is_part:
                axes.text(-0.015, index, name, transform=label_axis, ha="right", va="center",
                          fontsize=8, color=TEXT_SECONDARY)
                continue

            axes.text(-0.015, index - 0.13, name, transform=label_axis, ha="right", va="center",
                      fontsize=9, fontweight="bold", color=TEXT_PRIMARY)
            axes.text(-0.015, index + 0.2, span, transform=label_axis, ha="right", va="center",
                      fontsize=7.5, color=TEXT_SECONDARY)
        axes.invert_yaxis()
        axes.set_xlim(0, x_max)
        axes.grid(axis="x", color=GRID, linewidth=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.tick_params(axis="y", length=0)
        axes.set_title(f"{panel_name} of each stage", loc="left", fontsize=10.5,
                       color=TEXT_PRIMARY, fontweight="bold")

        if BUDGET_MILLISECONDS <= x_max:
            axes.axvline(BUDGET_MILLISECONDS, color=TEXT_SECONDARY, linestyle="--", linewidth=1)

    all_axes[-1].set_xlabel("milliseconds after end of user speech", color=TEXT_SECONDARY)
    figure.suptitle(
        title
        or f"Latency waterfall: {numbers['turn_count']} passing turns "
        f"({numbers['rejected_count']} rejected)",
        x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY,
    )

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Read a trace, print the numbers and draw the figure."""
    arguments = parse_arguments(argv)
    numbers = waterfall_numbers(read_trace(arguments.trace))

    print(f"trace: {arguments.trace}")
    print(f"passing turns {numbers['turn_count']}, rejected {numbers['rejected_count']}")
    print(f"TTFA p50 {numbers['ttfa_p50']:.1f} ms   p95 {numbers['ttfa_p95']:.1f} ms")
    for stage, percentiles in numbers["stages"].items():
        print(f"  {stage:<14} p50 {percentiles['p50_milliseconds']:8.1f}   p95 {percentiles['p95_milliseconds']:8.1f}")

    parts = numbers.get("llm_start_parts")

    if parts:
        print(f"  llm_start split ({parts['turn_count']} turns with every time):")
        for part, *_ in LLM_START_PARTS:
            print(f"    {part:<18} p50 {parts[part]['p50_milliseconds']:8.1f}   p95 {parts[part]['p95_milliseconds']:8.1f}")

    draw(numbers, arguments.output, arguments.title)
    print(f"figure written to {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
