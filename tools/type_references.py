"""Plays each recording and asks you to type what you said, into references.json.

    python3 tools/type_references.py --recordings recordings --speaker asal

The reference transcript has to be ground truth you produced yourself: wer/score_wer.py scores the
ASR transcript in the trace AGAINST this file, so a reference taken from ASR output would score the
recogniser against itself and measure nothing.

Work is saved after every answer, so you can stop with Ctrl-C and pick up where you left off. Files
you have already filled in are skipped unless you pass --redo.

At each prompt:

    <text>    the words you said, then Enter
    r         play it again
    s         skip this one for now
    q         save and quit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# The few things wer/score_wer.py does NOT normalise away, so they are worth getting right as you type.
TYPING_RULES = """\
Write it exactly as you SAID it. The scorer ignores case and punctuation and turns digits into
words on both sides, so "5" and "five" already match. What it will NOT forgive:

  * "it's" and "it is" are different words. Type the one you actually said.
  * Write "am" and "pm", not "a.m.".
  * Say a year the way you said it; "2026" normalises to "two thousand twenty six", not
    "twenty twenty six".
  * Include filler words you actually said ("um", "uh") only if you want them counted as words.
"""


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the settings of one typing session."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("references.json"),
        help="The file to write; it is grouped by speaker so both partners fit in one.",
    )
    parser.add_argument(
        "--speaker",
        required=True,
        help="Your handle, used as the group name (e.g. asal).",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Ask again about recordings that already have a reference.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not play the audio; just prompt (for when you already know what you said).",
    )

    return parser.parse_args(argv)


def load_references(references_path: Path) -> dict:
    """Return the references written so far, or an empty mapping."""
    if not references_path.exists():
        return {}

    try:
        existing = json.loads(references_path.read_text(encoding="utf-8"))

    except ValueError as error:
        raise SystemExit(f"{references_path} is not valid JSON: {error}") from error

    return existing if isinstance(existing, dict) else {}


def save_references(references_path: Path, references: dict) -> None:
    """Write the references file, sorted so a diff between two sessions reads cleanly."""
    references_path.write_text(
        json.dumps(references, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def play(wav_path: Path) -> None:
    """Play one recording through the speakers, or say why it could not be played."""
    try:
        subprocess.run(["afplay", str(wav_path)], check=True)

    except FileNotFoundError:
        print("  (afplay not found; use --no-play)", file=sys.stderr)

    except subprocess.CalledProcessError as error:
        print(f"  (afplay failed: {error})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Walk the recordings in replay order and collect a typed transcript for each."""
    arguments = parse_arguments(argv)

    recording_paths = sorted(arguments.recordings.glob("*.wav"))

    if not recording_paths:
        raise SystemExit(f"No .wav files in {arguments.recordings}.")

    references = load_references(arguments.references)
    speaker_references = references.setdefault(arguments.speaker, {})

    print(TYPING_RULES)
    print(f"{len(recording_paths)} recording(s) in {arguments.recordings}")
    print(f"writing to {arguments.references} under speaker {arguments.speaker!r}\n")

    for index, wav_path in enumerate(recording_paths, start=1):
        already = speaker_references.get(wav_path.name, "").strip()

        if already and not arguments.redo:
            continue

        print(f"[{index}/{len(recording_paths)}]  {wav_path.name}")

        if already:
            print(f"  current: {already}")

        while True:
            if not arguments.no_play:
                play(wav_path)

            try:
                answer = input("  you said: ").strip()

            except (EOFError, KeyboardInterrupt):
                print("\nstopped; everything typed so far is saved.")
                save_references(arguments.references, references)

                return 0

            if answer == "r":
                continue

            if answer == "s":
                print("  skipped\n")
                break

            if answer == "q":
                save_references(arguments.references, references)
                print("saved and stopped.")

                return 0

            if not answer:
                print("  empty — type the words, or s to skip, or q to quit.")

                continue

            speaker_references[wav_path.name] = answer
            # Saved every time, so a crash or a Ctrl-C never costs more than the current line.
            save_references(arguments.references, references)
            print("")

            break

    filled = sum(1 for value in speaker_references.values() if value.strip())
    print(f"{filled} of {len(recording_paths)} recording(s) have a reference.")

    missing = [path.name for path in recording_paths if not speaker_references.get(path.name, "").strip()]

    if missing:
        print("still missing: " + ", ".join(missing))
        print("run the same command again to finish them.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
