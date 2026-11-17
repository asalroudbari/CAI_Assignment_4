"""Shows what wer/score_wer.py will actually compare, and flags the ways a reference goes wrong.

    PYTHONPATH=. python3 tools/lint_references.py --references references.json
    PYTHONPATH=. python3 tools/lint_references.py --references references.json --trace run_output/trace.jsonl

A reference is "what you said", and WER punishes you for getting that wrong exactly as hard as it
punishes the ASR for mishearing it. The scorer normalizes both sides before comparing (case,
punctuation, digits to words), so most of what looks like a difference is not one -- but a few
things are NOT normalized away, and they are the ones people lose points to:

  * A contraction is not its expansion. "it's" and "it is" are two different word sequences, and so
    are "she's"/"she is", "I'd"/"I would", "don't"/"do not". The apostrophe survives normalization.
    Write whichever one came out of your mouth.
  * A year is not read as a year. "2026" becomes "two thousand twenty six", never "twenty twenty
    six". If you said "twenty twenty six", write those words, not the digits.
  * "am" and "pm" are the two letters. "p.m." is read as "pm" too, but "P.M." inside other
    punctuation is safer written plainly.
  * A colon makes a time: "3:30" -> "three thirty", "3:00" -> "three o'clock", "3:05" -> "three oh
    five". A dot makes a decimal: "3.5" -> "three point five".

With --trace it also shows a word-level diff against what your ASR heard, which is the fastest way
to find a reference you typed wrong: when the reference and the transcript differ by one small
function word, the reference is as likely to be the wrong one as the ASR.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wer"))

from score_wer import normalize_words  # noqa: E402

CONTRACTION = re.compile(r"\b\w+'\w+\b")
EXPANDABLE = re.compile(
    r"\b(?:it|he|she|that|there|what|who|they|we|you|i)\s+(?:is|are|am|will|would|have|had)\b",
    re.IGNORECASE,
)
FOUR_DIGIT = re.compile(r"\b\d{4}\b")
AMPM_DOTTED = re.compile(r"\b[ap]\s*\.\s*m\s*\.?", re.IGNORECASE)
COLON_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
DECIMAL = re.compile(r"\b\d+\.\d+\b")
CURLY = re.compile(r"[‘’“”]")
LEADING_JUNK = re.compile(r"^[^A-Za-z0-9]+")
DOUBLED_WORD = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def load_grouped(path: Path) -> dict[str, dict[str, str]]:
    """Return {speaker: {recording: text}} from a flat or grouped references file."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if all(isinstance(value, str) for value in data.values()):
        return {"all": data}

    return data


def hazards(text: str) -> list[str]:
    """Return the normalization hazards of one reference text."""
    found = []

    for match in CONTRACTION.finditer(text):
        found.append(
            f'contraction "{match.group(0)}" stays a contraction -- it will NOT match the expanded '
            f"form if that is what you actually said"
        )

    for match in EXPANDABLE.finditer(text):
        found.append(
            f'"{match.group(0)}" is written out -- it will NOT match a contraction if that is what '
            f"you actually said"
        )

    for match in FOUR_DIGIT.finditer(text):
        found.append(
            f'"{match.group(0)}" becomes "{" ".join(normalize_words(match.group(0)))}", not a year. '
            f"Write the words if you said it as a year."
        )

    for match in AMPM_DOTTED.finditer(text):
        found.append(f'"{match.group(0)}" -- write "am" or "pm" as two plain letters')

    for match in COLON_TIME.finditer(text):
        found.append(f'"{match.group(0)}" is read as a time: "{" ".join(normalize_words(match.group(0)))}"')

    for match in DECIMAL.finditer(text):
        found.append(f'"{match.group(0)}" is read as a decimal: "{" ".join(normalize_words(match.group(0)))}"')

    if CURLY.search(text):
        found.append("curly quotes present (harmless: the scorer folds them to straight ones)")

    if LEADING_JUNK.match(text):
        found.append(f'starts with punctuation ({LEADING_JUNK.match(text).group(0)!r}) -- a script artifact?')

    for match in DOUBLED_WORD.finditer(text):
        found.append(f'doubled word "{match.group(0)}" -- a typo, or did you really repeat it?')

    return found


def read_transcripts(trace_path: Path) -> dict[str, str]:
    """Return {recording: transcript} from a trace."""
    transcripts = {}

    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            extras = json.loads(line).get("extras", {})

            if extras.get("recording"):
                transcripts[extras["recording"]] = extras.get("transcript") or ""

    return transcripts


def diff_words(reference: list[str], hypothesis: list[str]) -> list[str]:
    """Return a readable word-level diff of reference against hypothesis."""
    lines = []

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=reference, b=hypothesis).get_opcodes():
        if tag == "equal":
            continue

        said = " ".join(reference[i1:i2]) or "-"
        heard = " ".join(hypothesis[j1:j2]) or "-"
        lines.append(f'      said "{said}"   heard "{heard}"')

    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--references", type=Path, default=Path("references.json"))
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--show-normalized", action="store_true", help="print the token stream of every reference")
    arguments = parser.parse_args(argv)

    grouped = load_grouped(arguments.references)
    transcripts = read_transcripts(arguments.trace) if arguments.trace else {}

    hazard_count = 0
    mismatch_count = 0

    for speaker, references in grouped.items():
        print(f"\n{'=' * 78}\n{speaker}: {len(references)} reference(s)\n{'=' * 78}")

        for recording, text in sorted(references.items()):
            notes = hazards(text)
            reference_words = normalize_words(text)
            heard_words = normalize_words(transcripts.get(recording, "")) if transcripts else []
            differs = bool(transcripts) and recording in transcripts and heard_words != reference_words

            if not notes and not differs and not arguments.show_normalized:
                continue

            print(f"\n  {recording}")
            print(f'    text       "{text}"')

            if arguments.show_normalized or notes:
                print(f'    compared   "{" ".join(reference_words)}"')

            for note in notes:
                hazard_count += 1
                print(f"    HAZARD     {note}")

            if differs:
                mismatch_count += 1
                print(f"    ASR heard  \"{' '.join(heard_words)}\"")
                print("    CHECK BY EAR - which side is wrong?")

                for line in diff_words(reference_words, heard_words):
                    print(line)

    print(f"\n{'=' * 78}")
    print(f"{hazard_count} normalization hazard(s)")

    if transcripts:
        print(f"{mismatch_count} reference(s) the ASR heard differently -- listen before blaming the ASR")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
