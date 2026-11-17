"""Decides, by measurement, whether the noise after your last word is speech or a breath.

    PYTHONPATH=. python3 tools/verify_speech_end.py

WHY. make_sidecars.py flags a quiet burst of sound that sits after the sentence, separated from it
by a silence, because that is what a breath or a chair creak looks like. But it is ALSO what the
release of a word-final stop consonant looks like: /t/ in "that", /k/ in "book", /d/ in "old" all
close the mouth (silence) and then burst (a short, quiet transient). Phonetically the two are very
hard to tell apart from energy alone.

They are easy to tell apart by measurement. speech_end_seconds decides how much audio the ASR ever
receives -- run_agent.py drains the queue the moment end of speech is announced, so audio after that
point is fed but never transcribed. So:

    transcribe the audio cut at the energy rule's value
    transcribe the audio cut at the earlier "sentence looks done here" value
    if the two transcripts differ, the burst carried a word, and cutting early would LOSE it

A cut that changes the transcript is wrong twice over: it costs WER, and it moves end_of_user_speech
earlier, which makes your own time to first audio look WORSE. A cut that leaves the transcript alone
costs nothing, and is the honest choice if what you hear there really is a breath.

This writes its findings into sidecar_review.json so they show up while you review. It NEVER sets
confirmed and never picks a value for you.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voice_kit.audio import SAMPLE_RATE_HZ, SAMPLE_WIDTH_BYTES, read_wav

BYTES_PER_SECOND = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES
REVIEW_FILE = Path("sidecar_review.json")


def pcm_until(pcm_bytes: bytes, seconds: float) -> bytes:
    """Return the audio the agent would have drained if speech ended at this time."""
    cut = int(seconds * BYTES_PER_SECOND) // SAMPLE_WIDTH_BYTES * SAMPLE_WIDTH_BYTES

    return pcm_bytes[:cut]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument(
        "--all", action="store_true",
        help="Check every recording, not only the ones with a detached burst.",
    )
    arguments = parser.parse_args(argv)

    if not REVIEW_FILE.exists():
        raise SystemExit(f"No {REVIEW_FILE}. Run: PYTHONPATH=. python3 make_sidecars.py propose")

    data = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    entries = data["recordings"]
    todo = [
        entry for entry in entries
        if arguments.all or entry.get("_detached_bursts_after_sentence")
    ]

    if not todo:
        print("No recording has a detached burst after the sentence. Nothing to verify.")
        return 0

    from asr import AsrSettings, FasterWhisperAsr

    print(f"loading ASR to check {len(todo)} recording(s) ...")
    recognizer = FasterWhisperAsr(AsrSettings())
    recognizer.warmup()
    print()

    keeps = 0
    cuts_are_safe = 0

    for entry in todo:
        wav_path = arguments.recordings / entry["wav"]
        pcm_bytes = read_wav(wav_path)
        rule_seconds = entry["_energy_rule_guess_seconds"]
        body_seconds = entry["_body_end_candidate_seconds"]

        at_rule = recognizer.transcribe(pcm_until(pcm_bytes, rule_seconds)).strip()
        at_body = recognizer.transcribe(pcm_until(pcm_bytes, body_seconds)).strip()
        differs = at_rule != at_body

        entry["_asr_cut_at_rule"] = at_rule
        entry["_asr_cut_at_body_end"] = at_body
        entry["_cutting_early_changes_transcript"] = differs

        print(f"{entry['wav']}")
        print(f'  cut at {rule_seconds:5.2f} s (energy rule)  "{at_rule}"')
        print(f'  cut at {body_seconds:5.2f} s (body end)     "{at_body}"')

        if differs:
            keeps += 1
            entry["_note"] = (
                f"VERIFIED SPEECH: cutting at {body_seconds:.2f} s changes the transcript, so that "
                f"burst carried part of a word -- almost certainly the release of the final "
                f"consonant. KEEP {rule_seconds:.2f} s. Cutting early would cost WER and would make "
                f"your own time to first audio look worse, not better."
            )
            print("  -> THE BURST IS SPEECH. Keep the energy rule's value (press ENTER).\n")

        else:
            cuts_are_safe += 1
            entry["_note"] = (
                f"CUT IS SAFE FOR WER: the transcript is identical either way, so nothing is lost by "
                f"ending speech at {body_seconds:.2f} s instead of {rule_seconds:.2f} s. Now listen: "
                f"if what you hear after the last word is a breath or a bump, press b -- leaving it "
                f"would credit you {(rule_seconds - body_seconds) * 1000:.0f} ms you did not earn. "
                f"If you hear a word, press ENTER."
            )
            print(f"  -> Transcript unchanged. Cutting costs no WER; listen and decide "
                  f"({(rule_seconds - body_seconds) * 1000:.0f} ms at stake).\n")

    REVIEW_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print("=" * 70)
    print(f"{keeps} recording(s) where the burst IS speech -- keep the energy rule's value")
    print(f"{cuts_are_safe} recording(s) where cutting costs no WER -- your ear decides")
    print(f"\nFindings written into {REVIEW_FILE}. Nothing was confirmed for you.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
