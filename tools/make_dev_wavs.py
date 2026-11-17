"""Synthetic dev fixtures only: these are tone bursts, never speech, and must never be used for any
reported measurement, for the waterfall, or for WER.

    PYTHONPATH=. python3 tools/make_dev_wavs.py <directory> --count 4

They exist so the turn loop can be smoke-tested before any real utterance has been recorded. Each
file is 16 kHz mono 16-bit, shaped the way a real recording is — a little silence, then a burst of
tone standing in for speech, then trailing silence — so the shim's energy rule finds an end of speech
in a sensible place. A sidecar JSON carries a scenario_node label.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from voice_kit.audio import SAMPLE_RATE_HZ, write_wav

LEADING_SILENCE_SECONDS = 0.3
TRAILING_SILENCE_SECONDS = 0.8
SPEECH_SECONDS = 1.2
SPEECH_AMPLITUDE = 6000
SPEECH_HERTZ = 220.0

DEV_SCENARIO_NODES = (
    "dev.fixture.one",
    "dev.fixture.two",
    "dev.fixture.three",
    "dev.fixture.four",
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the settings of one generation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument("directory", type=Path, help="Where to write the fixtures.")
    parser.add_argument(
        "--count", type=int, default=4, help="How many fixtures to write (default: 4)."
    )

    return parser.parse_args(argv)


def build_fixture_pcm(speech_seconds: float) -> bytes:
    """Return silence, then a tone burst standing in for speech, then silence."""
    pcm_bytes = bytearray()

    leading_samples = int(LEADING_SILENCE_SECONDS * SAMPLE_RATE_HZ)
    speech_samples = int(speech_seconds * SAMPLE_RATE_HZ)
    trailing_samples = int(TRAILING_SILENCE_SECONDS * SAMPLE_RATE_HZ)

    pcm_bytes += bytes(leading_samples * 2)

    for index in range(speech_samples):
        angle = 2.0 * math.pi * SPEECH_HERTZ * index / SAMPLE_RATE_HZ
        value = int(SPEECH_AMPLITUDE * math.sin(angle))
        pcm_bytes += value.to_bytes(2, "little", signed=True)

    pcm_bytes += bytes(trailing_samples * 2)

    return bytes(pcm_bytes)


def main(argv: list[str] | None = None) -> int:
    """Write the fixtures and say plainly what they are not."""
    arguments = parse_arguments(argv)
    arguments.directory.mkdir(parents=True, exist_ok=True)

    for index in range(1, arguments.count + 1):
        # Vary the length a little, so the turns are not all identical.
        speech_seconds = SPEECH_SECONDS + 0.2 * (index % 3)

        wav_path = arguments.directory / f"dev_{index:02d}.wav"
        write_wav(wav_path, build_fixture_pcm(speech_seconds), SAMPLE_RATE_HZ)

        sidecar = {
            "scenario_node": DEV_SCENARIO_NODES[(index - 1) % len(DEV_SCENARIO_NODES)]
        }
        wav_path.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=1) + "\n", encoding="utf-8"
        )

        print(f"wrote {wav_path}")

    print(
        f"\n{arguments.count} SYNTHETIC fixture(s) in {arguments.directory}. "
        "They are tone bursts, not speech: never report a number measured on them."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
