"""Proposes a speech_end_seconds for every recording, lets you confirm it by ear, then writes sidecars.

    PYTHONPATH=. python3 make_sidecars.py propose            # make the review file + marked audio
    PYTHONPATH=. python3 make_sidecars.py status             # how much is left to confirm
    PYTHONPATH=. python3 make_sidecars.py write              # write recordings/*.json

WHY THIS IS TWO STEPS. The shim's fallback is an energy rule: the end of the last 20 ms chunk whose
RMS reaches 200 (voice_kit/audio.py:detect_speech_end_seconds). A breath, a lip smack or a chair
creak after the last word clears 200 easily, and the rule then puts the end of speech hundreds of
milliseconds late. end_of_user_speech is the zero of every latency number in this assignment, so
that error is subtracted from your own TTFA and makes the agent look faster than it is. The energy
rule cannot tell a creak from a syllable. You can. So this tool never writes a sidecar on its own:

    propose  ->  you listen  ->  you edit + set "confirmed": true  ->  write

WHAT PROPOSE GIVES YOU, per recording:
  * the energy rule's guess, which is exactly what the shim uses when there is no sidecar
  * a "body end" candidate: the end of the last loud run BEFORE any silent gap near the end. When
    the two differ, something isolated is making noise after you stopped talking.
  * a 100 ms RMS ladder of the tail, so the gap is visible as text
  * review_audio/<name>.marked.wav -- your recording with a 1 kHz tick mixed in AT THE GUESS.
    Play it. The tick must land immediately after your last word, not after a breath.

Nothing in review_audio/ is ever fed to the agent; run_agent.py globs recordings/*.wav only.
"""

from __future__ import annotations

import argparse
import array
import os
import json
import math
import sys
from pathlib import Path

from voice_kit.audio import (
    DEFAULT_CHUNK_MILLISECONDS,
    DEFAULT_ENERGY_THRESHOLD,
    SAMPLE_RATE_HZ,
    SAMPLE_WIDTH_BYTES,
    chunk_byte_count,
    detect_speech_end_seconds,
    read_wav,
    root_mean_square,
    write_wav,
)

BYTES_PER_SECOND = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES

# A silence this long between the body of the sentence and a later loud chunk means the later chunk
# may be a separate noise rather than the tail of a word. Deliberately short: a breath follows the
# last word quickly, and in this recording set the real breaths sit only 100-160 ms after the
# sentence. A stop consonant can hold a 60-80 ms closure, so this does flag some genuine speech --
# that is the right trade, because a false flag only costs you one listen while a missed breath
# silently credits you 100-200 ms of latency you did not earn.
SUSPICIOUS_GAP_SECONDS = 0.06

# A detached run also has to be QUIET to count as noise. Measured on this recording set, a breath or
# a bump after the last word peaks at 0.05-0.12 of the loudest run of the sentence and lasts 60-80 ms,
# while a genuine trailing phrase peaks at 0.5-0.8 and lasts hundreds of milliseconds. Without this
# test the walk-back chains across ordinary mid-sentence pauses and proposes cutting real words off.
NOISE_PEAK_RATIO = 0.20

# How much of the end of the recording the ladder shows.
TAIL_LADDER_SECONDS = 2.5
LADDER_BUCKET_SECONDS = 0.1

TICK_HZ = 1000.0
TICK_SECONDS = 0.04
TICK_AMPLITUDE = 7000

# How much of the run-up to the proposed end of speech the tail clip keeps. Enough to hear the last
# word or two land, short enough that reviewing 36 files is a couple of minutes and not twenty.
TAIL_CLIP_LEAD_SECONDS = 1.5

REVIEW_FILE = Path("sidecar_review.json")
REVIEW_AUDIO_DIRECTORY = Path("review_audio")

HOW_TO_USE = (
    "EDIT THIS FILE BY HAND. For each recording: play review_audio/<wav>.marked.wav, check that the "
    "1 kHz tick lands right after your last word, correct speech_end_seconds if it does not, fill in "
    "scenario_node, then set confirmed to true. 'python3 make_sidecars.py write' refuses every entry "
    "whose confirmed is still false, so a sidecar can only exist because you listened to it."
)


def chunk_rms_series(pcm_bytes: bytes) -> list[float]:
    """Return the RMS of every 20 ms chunk, the unit the shim's energy rule thinks in."""
    step = chunk_byte_count(DEFAULT_CHUNK_MILLISECONDS)

    return [
        root_mean_square(pcm_bytes[start : start + step])
        for start in range(0, len(pcm_bytes), step)
    ]


def loud_runs(rms_series: list[float]) -> list[tuple[int, int, float]]:
    """Return (first chunk, last chunk, loudest RMS) of every unbroken run of audible chunks."""
    chunk_seconds = DEFAULT_CHUNK_MILLISECONDS / 1000
    runs: list[tuple[int, int, float]] = []
    current: list[int] = []

    for index, value in enumerate(rms_series):
        if value >= DEFAULT_ENERGY_THRESHOLD:
            current.append(index)

        elif current:
            runs.append((current[0], current[-1], max(rms_series[i] for i in current)))
            current = []

    if current:
        runs.append((current[0], current[-1], max(rms_series[i] for i in current)))

    return runs


def body_end_seconds(rms_series: list[float]) -> tuple[float, float, int]:
    """Return (end of the sentence body, the silence before the first detached noise, how many).

    The energy rule returns the end of the LAST audible chunk, wherever it is. A breath or a chair
    creak is not one chunk but a RUN of them, so comparing the last two chunks finds no gap -- the
    comparison has to be between RUNS. This walks back from the final run for as long as each run is
    separated from the one before it by a real silence, and returns the end of the run where that
    stops. A returned gap of 0.0 means the energy rule and this agree.
    """
    chunk_seconds = DEFAULT_CHUNK_MILLISECONDS / 1000
    runs = loud_runs(rms_series)

    if not runs:
        return 0.0, 0.0, 0

    body_peak = max(peak for _, _, peak in runs)
    index = len(runs) - 1
    first_gap = 0.0
    detached = 0

    while index > 0:
        gap_seconds = (runs[index][0] - runs[index - 1][1] - 1) * chunk_seconds
        is_quiet = runs[index][2] < NOISE_PEAK_RATIO * body_peak

        if gap_seconds < SUSPICIOUS_GAP_SECONDS or not is_quiet:
            break

        first_gap = gap_seconds
        detached += 1
        index -= 1

    return (runs[index][1] + 1) * chunk_seconds, first_gap, detached


def tail_ladder(rms_series: list[float], duration_seconds: float) -> list[str]:
    """Return a readable RMS-per-100 ms ladder of the end of the recording."""
    chunk_seconds = DEFAULT_CHUNK_MILLISECONDS / 1000
    per_bucket = max(1, round(LADDER_BUCKET_SECONDS / chunk_seconds))
    first_bucket = max(0, int((duration_seconds - TAIL_LADDER_SECONDS) / LADDER_BUCKET_SECONDS))
    lines = []

    for bucket in range(first_bucket, math.ceil(len(rms_series) / per_bucket)):
        window = rms_series[bucket * per_bucket : (bucket + 1) * per_bucket]

        if not window:
            continue

        peak = max(window)
        audible = "AUDIBLE" if peak >= DEFAULT_ENERGY_THRESHOLD else "silent "
        bar = "#" * min(40, int(peak / 150))
        lines.append(f"{bucket * LADDER_BUCKET_SECONDS:6.2f}s  {peak:8.1f}  {audible}  {bar}")

    return lines


def samples_of(pcm_bytes: bytes) -> array.array:
    """Return the 16-bit samples of PCM bytes, in machine order."""
    samples = array.array("h")
    samples.frombytes(pcm_bytes[: len(pcm_bytes) - len(pcm_bytes) % SAMPLE_WIDTH_BYTES])

    if sys.byteorder == "big":
        samples.byteswap()

    return samples


def write_marked_audio(wav_path: Path, pcm_bytes: bytes, mark_seconds: float, out_path: Path) -> None:
    """Write a copy of the recording with a tick MIXED IN at mark_seconds, for listening only.

    The tick is added to the audio rather than replacing it, so nothing underneath is hidden: if the
    mark is too early you still hear the word it lands on.
    """
    samples = samples_of(pcm_bytes)
    start = int(mark_seconds * SAMPLE_RATE_HZ)

    for offset in range(int(TICK_SECONDS * SAMPLE_RATE_HZ)):
        index = start + offset

        if not 0 <= index < len(samples):
            continue

        # A short fade in and out, so the tick is a click you can place and not a burst of noise.
        fade = math.sin(math.pi * offset / (TICK_SECONDS * SAMPLE_RATE_HZ))
        tick = TICK_AMPLITUDE * fade * math.sin(2 * math.pi * TICK_HZ * offset / SAMPLE_RATE_HZ)
        samples[index] = max(-32768, min(32767, int(samples[index] + tick)))

    if sys.byteorder == "big":
        samples.byteswap()

    write_wav(out_path, samples.tobytes(), SAMPLE_RATE_HZ)


def write_tail_clip(wav_path: Path, pcm_bytes: bytes, mark_seconds: float, out_path: Path) -> None:
    """Write just the end of the recording, with the tick, so a reviewer hears only what matters."""
    start_seconds = max(0.0, mark_seconds - TAIL_CLIP_LEAD_SECONDS)
    start_byte = int(start_seconds * BYTES_PER_SECOND) // SAMPLE_WIDTH_BYTES * SAMPLE_WIDTH_BYTES

    write_marked_audio(wav_path, pcm_bytes[start_byte:], mark_seconds - start_seconds, out_path)


def load_plan(plan_path: Path | None) -> dict[str, str]:
    """Return {wav name: scenario_node} from a plan file, or an empty mapping."""
    if plan_path is None or not plan_path.exists():
        return {}

    data = json.loads(plan_path.read_text(encoding="utf-8"))

    return {name: str(node) for name, node in data.items()}


def load_reference_texts(references_path: Path | None) -> dict[str, str]:
    """Return {wav name: reference text} from a flat or grouped references file, to show while reviewing."""
    if references_path is None or not references_path.exists():
        return {}

    data = json.loads(references_path.read_text(encoding="utf-8"))
    texts: dict[str, str] = {}

    for key, value in data.items():
        if isinstance(value, str):
            texts[key] = value

        elif isinstance(value, dict):
            for name, text in value.items():
                if isinstance(text, str):
                    texts[name] = text

    return texts


def propose(arguments: argparse.Namespace) -> int:
    """Measure every recording, write the review file and the marked audio, and print the summary."""
    wav_paths = sorted(arguments.recordings.glob("*.wav"))

    if not wav_paths:
        raise SystemExit(f"No .wav files in {arguments.recordings}.")

    plan = load_plan(arguments.plan)
    texts = load_reference_texts(arguments.references)

    existing: dict[str, dict] = {}

    if REVIEW_FILE.exists() and not arguments.overwrite:
        previous = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        existing = {entry["wav"]: entry for entry in previous.get("recordings", [])}

    entries = []
    suspicious = []

    for wav_path in wav_paths:
        pcm_bytes = read_wav(wav_path)
        duration_seconds = len(pcm_bytes) / BYTES_PER_SECOND
        rms_series = chunk_rms_series(pcm_bytes)
        energy_guess = detect_speech_end_seconds(pcm_bytes)
        body_end, gap_seconds, detached_count = body_end_seconds(rms_series)

        kept = existing.get(wav_path.name, {})
        was_confirmed = bool(kept.get("confirmed"))

        # A confirmed value is never silently replaced by a fresh guess.
        chosen = float(kept["speech_end_seconds"]) if was_confirmed else energy_guess

        marked_path = REVIEW_AUDIO_DIRECTORY / f"{wav_path.stem}.marked.wav"
        write_marked_audio(wav_path, pcm_bytes, chosen, marked_path)

        tail_path = REVIEW_AUDIO_DIRECTORY / f"{wav_path.stem}.tail.wav"
        write_tail_clip(wav_path, pcm_bytes, chosen, tail_path)

        note = ""

        if gap_seconds:
            note = (
                f"SUSPECT: {detached_count} burst(s) of sound sit AFTER the sentence, cut off from it "
                f"by {gap_seconds:.2f} s of silence. The sentence looks finished at {body_end:.2f} s, "
                f"but the energy rule takes the last burst and says {energy_guess:.2f} s -- "
                f"{(energy_guess - body_end) * 1000:.0f} ms later. If those bursts are a breath or a "
                f"bump and not a word, set speech_end_seconds to {body_end:.2f}; leaving it alone "
                f"credits you {(energy_guess - body_end) * 1000:.0f} ms of latency you did not earn."
            )
            suspicious.append(wav_path.name)

        entries.append(
            {
                "wav": wav_path.name,
                "confirmed": was_confirmed,
                "speech_end_seconds": round(chosen, 3),
                "scenario_node": kept.get("scenario_node") or plan.get(wav_path.name, ""),
                "_said": texts.get(wav_path.name, ""),
                "_duration_seconds": round(duration_seconds, 3),
                "_energy_rule_guess_seconds": round(energy_guess, 3),
                "_body_end_candidate_seconds": round(body_end, 3),
                "_silent_gap_before_last_noise_seconds": round(gap_seconds, 3),
                "_detached_bursts_after_sentence": detached_count,
                "_trailing_silence_seconds": round(duration_seconds - chosen, 3),
                "_note": note,
                "_play": f"afplay {marked_path}",
                "_play_tail": f"afplay {tail_path}",
                "_tail_runs": [
                    f"{first * 0.02:.2f}-{(last + 1) * 0.02:.2f}s  peak {peak:7.1f}  "
                    f"{(last - first + 1) * 20:4d} ms  "
                    + ("quiet, likely not speech" if peak < NOISE_PEAK_RATIO * max(
                        item[2] for item in loud_runs(rms_series)) else "")
                    for first, last, peak in loud_runs(rms_series)[-6:]
                ],
                "_tail_ladder": tail_ladder(rms_series, duration_seconds),
            }
        )

    REVIEW_FILE.write_text(
        json.dumps({"_how_to_use": HOW_TO_USE, "recordings": entries}, indent=2) + "\n",
        encoding="utf-8",
    )

    confirmed_count = sum(1 for entry in entries if entry["confirmed"])
    print(f"{len(entries)} recording(s) measured")
    print(f"review file         {REVIEW_FILE}")
    print(f"marked audio        {REVIEW_AUDIO_DIRECTORY}/ (listening only, never replayed)")
    print(f"already confirmed   {confirmed_count}")
    print(f"still to confirm    {len(entries) - confirmed_count}")

    if suspicious:
        print(f"\n{len(suspicious)} recording(s) have a quiet burst after the sentence. That is what a breath")
        print("looks like -- and ALSO what the release of a final /t/, /k/ or /d/ looks like. Resolve it")
        print("by measurement before you trust it:  PYTHONPATH=. python3 tools/verify_speech_end.py\n")

        for name in suspicious:
            entry = next(item for item in entries if item["wav"] == name)
            print(
                f"  {name}: rule says {entry['_energy_rule_guess_seconds']:.2f} s, sentence looks "
                f"done at {entry['_body_end_candidate_seconds']:.2f} s -- "
                f"{(entry['_energy_rule_guess_seconds'] - entry['_body_end_candidate_seconds']) * 1000:.0f} ms "
                f"of possible false credit"
            )

    else:
        print("\nNo recording shows a silent gap before its last loud chunk.")
        print("That is the breath-and-creak signature only; it does not prove the guesses are right.")
        print("Listen to the marked audio before confirming anything.")

    return 0


def read_review() -> list[dict]:
    """Return the entries of the review file, or exit if it is not there."""
    if not REVIEW_FILE.exists():
        raise SystemExit(f"No {REVIEW_FILE}. Run: PYTHONPATH=. python3 make_sidecars.py propose")

    return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))["recordings"]


def status(arguments: argparse.Namespace) -> int:
    """Print which recordings are confirmed and which still need an ear on them."""
    entries = read_review()
    unconfirmed = [entry for entry in entries if not entry.get("confirmed")]
    missing_node = [
        entry for entry in entries if entry.get("confirmed") and not entry.get("scenario_node")
    ]

    print(f"{len(entries) - len(unconfirmed)} of {len(entries)} confirmed")

    for entry in unconfirmed:
        print(
            f"  not confirmed  {entry['wav']:<20} rule says {entry['_energy_rule_guess_seconds']:.2f} s"
            + ("   <- SUSPECT" if entry["_silent_gap_before_last_noise_seconds"] else "")
        )

    for entry in missing_node:
        print(f"  no scenario_node  {entry['wav']}")

    return 0 if not unconfirmed and not missing_node else 1


def write_sidecars(arguments: argparse.Namespace) -> int:
    """Write one sidecar per confirmed recording; refuse everything not confirmed."""
    entries = read_review()
    refused = []
    written = []

    for entry in entries:
        wav_path = arguments.recordings / entry["wav"]

        if not entry.get("confirmed"):
            refused.append(f"{entry['wav']}: confirmed is false, nobody has listened to it")
            continue

        if not wav_path.exists():
            refused.append(f"{entry['wav']}: no such recording")
            continue

        node = entry.get("scenario_node")

        if not node:
            refused.append(f"{entry['wav']}: scenario_node is empty")
            continue

        speech_end = float(entry["speech_end_seconds"])
        duration_seconds = len(read_wav(wav_path)) / BYTES_PER_SECOND

        # The same bound voice_kit/source.py:load_recording enforces, checked here so a typo in the
        # review file is caught now and not halfway through a timed run.
        if not 0.0 < speech_end <= duration_seconds:
            refused.append(
                f"{entry['wav']}: speech_end_seconds {speech_end} is outside the recording "
                f"({duration_seconds:.3f} s)"
            )
            continue

        sidecar_path = wav_path.with_suffix(".json")
        sidecar_path.write_text(
            json.dumps({"speech_end_seconds": speech_end, "scenario_node": node}, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(f"{sidecar_path.name}  speech_end_seconds={speech_end:.3f}  {node}")

    for line in written:
        print(f"wrote    {line}")

    for line in refused:
        print(f"REFUSED  {line}")

    print(f"\n{len(written)} sidecar(s) written, {len(refused)} refused")

    return 1 if refused else 0


def review(arguments: argparse.Namespace) -> int:
    """Play the tail of each unconfirmed recording and record the answer, one file at a time."""
    data = json.loads(REVIEW_FILE.read_text(encoding="utf-8")) if REVIEW_FILE.exists() else None

    if data is None:
        raise SystemExit(f"No {REVIEW_FILE}. Run: PYTHONPATH=. python3 make_sidecars.py propose")

    entries = data["recordings"]
    todo = [entry for entry in entries if not entry.get("confirmed")]

    if getattr(arguments, "only", None):
        todo = [entry for entry in todo if arguments.only in entry["wav"]]

    if not todo:
        print("Every recording is already confirmed.")
        return 0

    if getattr(arguments, "no_audio", False):
        print("--no-audio: nothing will be played. Only use this for files you have ALREADY heard.")

    print(f"{len(todo)} to review. The tick is where the shim would call end of speech.")
    print("It must land right after your last word -- not after a breath or a bump.\n")
    print("  ENTER  the tick was right, keep this value and mark it confirmed")
    print("  b      the tick was late, use the body-end candidate instead")
    print("  <num>  use this many seconds instead")
    print("  r      play it again")
    print("  s      skip this one for now")
    print("  q      save and stop\n")

    for position, entry in enumerate(todo, start=1):
        tail_path = REVIEW_AUDIO_DIRECTORY / f"{Path(entry['wav']).stem}.tail.wav"

        while True:
            print(f"\n[{position}/{len(todo)}] {entry['wav']}")
            print(f'  said                "{entry["_said"]}"')
            print(f"  tick is at          {entry['speech_end_seconds']:.2f} s of {entry['_duration_seconds']:.2f} s")

            if entry["_detached_bursts_after_sentence"]:
                verified = entry.get("_cutting_early_changes_transcript")

                if verified is True:
                    print("  VERIFIED SPEECH     a burst sits after the sentence, but cutting it changes the")
                    print("                      transcript, so it carried a word. Press ENTER to keep "
                          f"{entry['_energy_rule_guess_seconds']:.2f} s.")

                elif verified is False:
                    print(f"  LISTEN              a burst sits after the sentence and cutting it costs no WER.")
                    print(f"                      Breath or bump -> press b for {entry['_body_end_candidate_seconds']:.2f} s "
                          f"({(entry['_energy_rule_guess_seconds'] - entry['_body_end_candidate_seconds']) * 1000:.0f} ms). A word -> ENTER.")

                else:
                    print(f"  UNVERIFIED          a burst sits after the sentence; it may be a breath OR the")
                    print(f"                      release of a final consonant. Run "
                          "tools/verify_speech_end.py first.")

            if not getattr(arguments, "no_audio", False):
                os.system(f"afplay {tail_path}" if tail_path.exists() else "true")
            answer = input("  > ").strip().lower()

            if answer == "r":
                continue

            if answer == "s":
                break

            if answer == "q":
                REVIEW_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                print(f"\nSaved. Run 'make_sidecars.py status' to see what is left.")
                return 0

            if answer == "b":
                entry["speech_end_seconds"] = entry["_body_end_candidate_seconds"]

            elif answer:
                try:
                    value = float(answer)

                except ValueError:
                    print("  Not a number. Try again.")
                    continue

                if not 0.0 < value <= entry["_duration_seconds"]:
                    print(f"  Must be above 0 and at most {entry['_duration_seconds']:.2f}. Try again.")
                    continue

                entry["speech_end_seconds"] = round(value, 3)

            entry["confirmed"] = True
            entry["_trailing_silence_seconds"] = round(
                entry["_duration_seconds"] - entry["speech_end_seconds"], 3)
            print(f"  confirmed at {entry['speech_end_seconds']:.2f} s")

            # Written after every answer, so stopping never loses work already done.
            REVIEW_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            break

    confirmed = sum(1 for entry in entries if entry.get("confirmed"))
    print(f"\n{confirmed} of {len(entries)} confirmed.")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("propose", "review", "status", "write"))
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument(
        "--plan", type=Path, default=Path("tasks/scenario_nodes.json"),
        help="Optional {wav name: scenario_node} used to prefill the review file.",
    )
    parser.add_argument(
        "--references", type=Path, default=Path("references.json"),
        help="Optional references file, so the review shows what you said.",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Do not play anything; for a second pass over files you have already listened to.",
    )
    parser.add_argument(
        "--only", default=None,
        help="Review only the recordings whose name contains this text.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Throw away the existing review file instead of keeping confirmed entries.",
    )
    arguments = parser.parse_args(argv)

    return {"propose": propose, "review": review, "status": status,
            "write": write_sidecars}[arguments.command](arguments)


if __name__ == "__main__":
    raise SystemExit(main())
