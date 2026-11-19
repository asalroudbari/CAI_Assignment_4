"""Plays a recorded PACE-ICE run back as a conversation, for the demo video.

    PYTHONPATH=. python3 tools/demo_playback.py
    PYTHONPATH=. python3 tools/demo_playback.py --dialogues jae_d21 asal_d06 jae_d31

PACE-ICE has no microphone and no speaker, so the agent is measured by replaying recordings through
the voice_kit shim and capturing what the TTS writes. This tool plays one such run back on a laptop,
turn by turn, with the timing that run measured:

    1. the user's recording, cut at its end of speech (the shim's speech_end_seconds)
    2. silence for exactly that turn's TTFA (tts_first_audio - end_of_user_speech, from the trace)
    3. the agent's captured TTS audio for that turn (run_output_pace/<run>/tts/<turn_id>.wav)

The transcript, the reply and the TTFA are printed as each turn plays. Each user clip is played at a
peak of -3 dBFS (--user-peak-dbfs, --no-normalize), because some recordings sit ~16 dB below the
TTS; only the played copy is changed, never the recording the agent was measured on. Nothing is generated here:
every sound and every number comes from the shipped run. Standard library only; it plays audio with
afplay (macOS), else aplay or paplay (Linux).
"""

from __future__ import annotations

import argparse
import array
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_kit.source import load_recording  # noqa: E402

DEFAULT_RUN = Path("run_output_pace/full_7164027")
DEFAULT_DIALOGUES = ("jae_d21", "asal_d06", "jae_d31")  # one per task family, both voices
# How loud each user clip is made for playback (peak level), and the most it may be amplified.
DEFAULT_USER_PEAK_DBFS = -3.0
MAX_GAIN = 20.0
FAMILY = {"timer": "Timers and reminders", "reminder": "Timers and reminders",
          "lookup": "Lookup", "booking": "Booking a room with two constraints"}


def player() -> list[str]:
    """Return the command that plays a WAV file on this machine."""
    for command in (["afplay"], ["aplay", "-q"], ["paplay"]):
        if shutil.which(command[0]):
            return command
    raise SystemExit("No audio player found (afplay, aplay or paplay).")


def play(command: list[str], path: Path) -> None:
    subprocess.run([*command, str(path)], check=True)


def cut_at_speech_end(wav_path: Path, seconds: float, out_path: Path, peak_dbfs: float | None) -> None:
    """Write the recording up to its end of speech, which is where the shim starts the clock.

    With ``peak_dbfs`` the clip is scaled so its loudest sample sits at that level. Only the copy
    played in the demo is louder; the recording the agent was measured on is untouched.
    """
    with wave.open(str(wav_path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(int(seconds * source.getframerate()))

    if peak_dbfs is not None:
        samples = array.array("h", frames)
        peak = max((abs(sample) for sample in samples), default=0)

        if peak:
            gain = min(32767 * 10 ** (peak_dbfs / 20) / peak, MAX_GAIN)
            samples = array.array("h", (max(-32768, min(32767, round(x * gain))) for x in samples))
            frames = samples.tobytes()

    with wave.open(str(out_path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="A shipped run directory.")
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--dialogues", nargs="+", default=list(DEFAULT_DIALOGUES),
                        help="Recording-name prefixes of the dialogues to play, in order.")
    parser.add_argument("--pause", type=float, default=1.5, help="Seconds between dialogues.")
    parser.add_argument("--user-peak-dbfs", type=float, default=DEFAULT_USER_PEAK_DBFS,
                        help="Play each user clip at this peak level (default -3 dBFS).")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Play the user recordings at their recorded level.")
    arguments = parser.parse_args()

    records = [json.loads(line) for line in (arguments.run / "trace.jsonl").read_text().splitlines() if line]
    command = player()

    print(f"Replaying PACE-ICE run {arguments.run} ({len(records)} turns in its trace).", flush=True)
    print("User audio is cut at end of speech; the agent's audio starts after that turn's measured TTFA.\n", flush=True)

    with tempfile.TemporaryDirectory() as scratch:
        for prefix in arguments.dialogues:
            turns = [r for r in records if r["extras"]["recording"].startswith(prefix + "_")]
            if not turns:
                print(f"(no turns for {prefix})", flush=True)
                continue

            family = FAMILY.get((turns[0].get("scenario_node") or "").split(".")[0], "")
            print("=" * 78, flush=True)
            print(f"Dialogue {prefix}  {family}", flush=True)
            print("=" * 78, flush=True)

            for record in turns:
                events, extras = record["events"], record["extras"]
                ttfa_ms = (events["tts_first_audio"] - events["end_of_user_speech"]) / 1e6
                recording = load_recording(arguments.recordings / extras["recording"])
                clip = Path(scratch) / extras["recording"]
                cut_at_speech_end(
                    arguments.recordings / extras["recording"],
                    recording.speech_end_seconds,
                    clip,
                    None if arguments.no_normalize else arguments.user_peak_dbfs,
                )

                print(f"\n[{record['turn_id']}  {record.get('scenario_node')}]", flush=True)
                print("USER  (speaking)", flush=True)
                play(command, clip)
                print(f"  heard: {extras['transcript']}", flush=True)
                time.sleep(ttfa_ms / 1000)
                print(f"AGENT (TTFA {ttfa_ms:.0f} ms): {extras['reply']}", flush=True)
                play(command, arguments.run / "tts" / f"{record['turn_id']}.wav")

            print(flush=True)
            time.sleep(arguments.pause)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
