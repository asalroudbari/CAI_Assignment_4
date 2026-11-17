"""Checks recorded utterances against exactly what the audio shim will see.

    PYTHONPATH=. python3 tools/check_recordings.py --recordings recordings

Every number here comes from voice_kit's own functions, so a file that passes this passes the shim.
The one that matters most is the detected end of speech: the shim takes it as the end of the last
20 ms chunk whose root mean square reaches 200, and every latency number in the assignment is
measured from there. Background noise above that floor drags it to the end of the file and quietly
ruins the run.

Exit code is 0 only when every file passes every hard check.
"""

from __future__ import annotations

import argparse
import array
import json
import sys
import wave
from pathlib import Path

# Both tools live in tools/ and are run as scripts from the repository root, so the script's own
# directory is on sys.path and this import is the directory next door rather than a package.
from normalize_recordings import WAVE_FORMAT_PCM, NotCanonicalWav, describe_format, read_chunks

from voice_kit.audio import (
    DEFAULT_CHUNK_MILLISECONDS,
    DEFAULT_ENERGY_THRESHOLD,
    SAMPLE_RATE_HZ,
    SAMPLE_WIDTH_BYTES,
    CHANNEL_COUNT,
    chunk_byte_count,
    detect_speech_end_seconds,
    read_wav,
    root_mean_square,
)

BYTES_PER_SECOND = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES

# Below this the speech is marginal: the energy rule still finds it, but a quieter syllable at the
# end of the sentence may fall under the threshold and move the detected end of speech earlier.
MARGINAL_SPEECH_RMS = 500.0

# The shim keeps feeding after speech ends, and that trailing silence is what lets the end-of-speech
# stamp land honestly instead of at the edge of the file.
MINIMUM_TRAILING_SILENCE_SECONDS = 0.3

# A 16-bit sample at or above this is at the ceiling, so the recording is probably clipped.
CLIPPING_SAMPLE = 32_700


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the settings of one check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument(
        "--recordings",
        type=Path,
        default=Path("recordings"),
        help="Directory of recorded utterances to check.",
    )

    return parser.parse_args(argv)


def wav_format(wav_path: Path) -> tuple[int, int, int]:
    """Return the sample rate, channel count and sample width of a WAV file."""
    with wave.open(str(wav_path), "rb") as wav_file:
        return (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
        )


def peak_sample(pcm_bytes: bytes) -> int:
    """Return the largest absolute sample value."""
    samples = array.array("h")
    samples.frombytes(pcm_bytes[: len(pcm_bytes) - len(pcm_bytes) % SAMPLE_WIDTH_BYTES])

    if sys.byteorder == "big":
        samples.byteswap()

    return max((abs(sample) for sample in samples), default=0)


def loudest_chunk_rms(pcm_bytes: bytes) -> float:
    """Return the root mean square of the loudest 20 ms chunk, the unit the shim thinks in."""
    step = chunk_byte_count(DEFAULT_CHUNK_MILLISECONDS)

    return max(
        (
            root_mean_square(pcm_bytes[start : start + step])
            for start in range(0, len(pcm_bytes), step)
        ),
        default=0.0,
    )


def check_one(wav_path: Path) -> list[str]:
    """Print a report for one recording; return its hard failures."""
    failures: list[str] = []
    warnings: list[str] = []

    print(f"\n{wav_path.name}")

    # Checked before anything else, and NOT through the wave module, because the wave module is
    # exactly what hides this: a recent Python opens a WAVE_FORMAT_EXTENSIBLE file without
    # complaint, so a recording set checked on a laptop passes here and then dies on PACE-ICE with
    # "wave.Error: unknown format: 65534". tools/normalize_recordings.py rewrites the header.
    try:
        format_tag = describe_format(read_chunks(wav_path)[0])[0]

    except NotCanonicalWav as error:
        print(f"  header              unreadable: {error}")

        return [f"{wav_path.name}: {error}"]

    if format_tag == WAVE_FORMAT_PCM:
        print("  header              WAVE_FORMAT_PCM, readable by every Python")

    else:
        print(
            f"  header              format tag {format_tag} (WAVE_FORMAT_EXTENSIBLE)"
            "   <- Python 3.9 and 3.10 REFUSE this"
        )
        failures.append(
            f"{wav_path.name}: extensible WAV header; it works here but fails on PACE-ICE and in "
            "the Ubuntu 22.04 image. Fix with: "
            "PYTHONPATH=. python3 tools/normalize_recordings.py --recordings <dir> --in-place"
        )

    try:
        sample_rate_hz, channel_count, sample_width_bytes = wav_format(wav_path)

    except Exception as error:  # noqa: BLE001
        print(f"  cannot be opened as a WAV file: {error}")

        return [f"{wav_path.name}: not a readable WAV file"]

    expected = (SAMPLE_RATE_HZ, CHANNEL_COUNT, SAMPLE_WIDTH_BYTES)
    actual = (sample_rate_hz, channel_count, sample_width_bytes)
    print(
        f"  format              {sample_rate_hz} Hz, {channel_count} channel(s), "
        f"{sample_width_bytes * 8}-bit"
        + ("" if actual == expected else "   <- the shim needs 16000 Hz, 1 channel, 16-bit")
    )

    if actual != expected:
        failures.append(f"{wav_path.name}: wrong format, convert it with afconvert")

        return failures

    pcm_bytes = read_wav(wav_path)
    duration_seconds = len(pcm_bytes) / BYTES_PER_SECOND
    print(f"  duration            {duration_seconds:.3f} s")

    peak = peak_sample(pcm_bytes)
    print(f"  peak sample         {peak} of 32767")

    if peak >= CLIPPING_SAMPLE:
        failures.append(f"{wav_path.name}: clipped, record it again further from the mic")

    loudest = loudest_chunk_rms(pcm_bytes)
    print(f"  loudest 20 ms rms   {loudest:.1f}  (shim calls 200 audible)")

    if loudest < DEFAULT_ENERGY_THRESHOLD:
        failures.append(f"{wav_path.name}: no chunk reaches the shim's threshold, it has no speech")

        return failures

    if loudest < MARGINAL_SPEECH_RMS:
        warnings.append("speech is quiet, the end of the sentence may fall under the threshold")

    try:
        speech_end_seconds = detect_speech_end_seconds(pcm_bytes)

    except ValueError as error:
        print(f"  speech end          not found: {error}")

        return failures + [f"{wav_path.name}: no speech found"]

    trailing_silence_seconds = duration_seconds - speech_end_seconds
    print(f"  speech ends at      {speech_end_seconds:.3f} s")
    print(f"  trailing silence    {trailing_silence_seconds:.3f} s")

    chunk_seconds = DEFAULT_CHUNK_MILLISECONDS / 1000

    if trailing_silence_seconds < chunk_seconds:
        failures.append(
            f"{wav_path.name}: speech end is within one chunk of the end of the file — background "
            "noise beat the energy rule, or you stopped recording too soon"
        )

    elif trailing_silence_seconds < MINIMUM_TRAILING_SILENCE_SECONDS:
        failures.append(
            f"{wav_path.name}: only {trailing_silence_seconds:.3f} s of trailing silence, "
            f"record at least {MINIMUM_TRAILING_SILENCE_SECONDS} s"
        )

    sidecar_path = wav_path.with_suffix(".json")

    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        print(f"  sidecar             {sidecar_path.name}: {sidecar}")

        if "speech_end_seconds" in sidecar:
            print(
                f"                      the sidecar wins over the energy rule: the shim will use "
                f"{float(sidecar['speech_end_seconds']):.3f} s"
            )

    else:
        print("  sidecar             none (the energy rule above is what the shim will use)")

    for warning in warnings:
        print(f"  WARNING             {warning}")

    for failure in failures:
        print(f"  FAIL                {failure.split(': ', 1)[1]}")

    if not failures and not warnings:
        print("  ok")

    return failures


def main(argv: list[str] | None = None) -> int:
    """Check every recording in a directory."""
    arguments = parse_arguments(argv)
    wav_paths = sorted(arguments.recordings.glob("*.wav"))

    if not wav_paths:
        raise SystemExit(f"No .wav files in {arguments.recordings}.")

    print(f"checking {len(wav_paths)} recording(s) in {arguments.recordings}")

    failures: list[str] = []

    for wav_path in wav_paths:
        failures += check_one(wav_path)

    print("\n" + "=" * 70)

    if failures:
        print(f"{len(failures)} problem(s):")

        for failure in failures:
            print(f"  {failure}")

        return 1

    print(f"all {len(wav_paths)} recording(s) are ready for the shim")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
