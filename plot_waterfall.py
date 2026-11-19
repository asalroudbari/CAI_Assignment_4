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

Needs matplotlib (not part of the agent image; any Python with matplotlib works).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from voice_kit import (  # noqa: E402
    read_trace,
    record_problems,
    summarize_stages,
    summarize_time_to_first_audio,
)

# Stages on the TTFA path, in time order, with their label and fill colour.
TTFA_PATH_STAGES = (
    ("asr", "ASR: end_of_user_speech → asr_final", "#2a78d6"),
    ("llm_start", "LLM start: asr_final → llm_first_token", "#eb6834"),
    ("speech_start", "Speech start: llm_first_token → tts_first_audio\n(rest of first sentence + its synthesis)", "#1baf7a"),
)
OFF_PATH_STAGE = ("llm_remaining", "Not on TTFA path: llm_first_token → llm_complete\n(overlaps speech start)")
OFF_PATH_COLOUR = "#898781"
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#55544e"
GRID = "#e4e3dd"
BUDGET_MILLISECONDS = 1500.0

# Segments narrower than this many milliseconds get their label outside the bar.
MIN_INLINE_LABEL_MILLISECONDS = 70.0


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
    }


def draw(numbers: dict, output: Path, title: str | None) -> None:
    """Draw the two stacked bars and write the figure."""
    rows = (("p50", "p50_milliseconds", numbers["ttfa_p50"]), ("p95", "p95_milliseconds", numbers["ttfa_p95"]))

    figure, axes = plt.subplots(figsize=(10, 3.6), dpi=200)
    bar_height = 0.42
    off_path_height = 0.12
    longest = 0.0

    for row_index, (row_name, key, measured_ttfa) in enumerate(rows):
        y = len(rows) - 1 - row_index
        left = 0.0

        for stage, label, colour in TTFA_PATH_STAGES:
            width = numbers["stages"][stage][key]
            axes.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=colour,
                edgecolor="white",
                linewidth=2,
                label=label if row_index == 0 else None,
            )

            text = f"{width:.0f}"
            if width >= MIN_INLINE_LABEL_MILLISECONDS:
                axes.text(left + width / 2, y, text, ha="center", va="center", color="white",
                          fontsize=9, fontweight="bold")
            else:
                axes.text(left + width / 2, y + bar_height / 2 + 0.04, text, ha="center",
                          va="bottom", color=TEXT_PRIMARY, fontsize=8)
            left += width

        llm_start_end = numbers["stages"]["asr"][key] + numbers["stages"]["llm_start"][key]
        remaining = numbers["stages"][OFF_PATH_STAGE[0]][key]
        off_y = y - bar_height / 2 - off_path_height / 2 - 0.04
        axes.barh(
            off_y,
            remaining,
            left=llm_start_end,
            height=off_path_height,
            color="white",
            edgecolor=OFF_PATH_COLOUR,
            hatch="////",
            linewidth=0.8,
            label=OFF_PATH_STAGE[1] if row_index == 0 else None,
        )
        axes.text(llm_start_end + remaining + 8, off_y, f"{remaining:.0f}", ha="left", va="center",
                  color=TEXT_SECONDARY, fontsize=7.5)

        axes.plot([measured_ttfa, measured_ttfa], [y - bar_height / 2 - 0.02, y + bar_height / 2 + 0.02],
                  color="black", linewidth=2)
        axes.text(max(left, measured_ttfa) + 12, y + 0.06,
                  f"measured TTFA {row_name} = {measured_ttfa:.0f} ms", ha="left", va="center",
                  color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
        axes.text(max(left, measured_ttfa) + 12, y - 0.12,
                  f"(sum of stage {row_name}s = {left:.0f} ms)", ha="left", va="center",
                  color=TEXT_SECONDARY, fontsize=7.5)

        longest = max(longest, left, measured_ttfa, llm_start_end + remaining)

    # The x range follows the data, so a fast run is not squeezed into the left edge. The 1500 ms
    # target is drawn only when it falls inside that range.
    x_max = max(longest * 1.75, 300.0)

    if BUDGET_MILLISECONDS <= x_max:
        axes.axvline(BUDGET_MILLISECONDS, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        axes.text(BUDGET_MILLISECONDS, len(rows) - 0.45, "1500 ms target (p50)", ha="center",
                  va="bottom", color=TEXT_SECONDARY, fontsize=8)

    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([name for name, _, _ in reversed(rows)], fontsize=10)
    axes.set_xlabel("milliseconds after end of user speech", color=TEXT_SECONDARY)
    axes.set_xlim(0, x_max)
    axes.set_ylim(-0.75, len(rows) - 0.2)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)

    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.tick_params(axis="y", length=0)

    axes.set_title(
        title
        or f"Latency waterfall, {numbers['turn_count']} passing turns "
        f"({numbers['rejected_count']} rejected)",
        loc="left",
        fontsize=11,
        color=TEXT_PRIMARY,
    )
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=8)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
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

    draw(numbers, arguments.output, arguments.title)
    print(f"figure written to {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
