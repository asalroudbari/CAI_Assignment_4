"""Rewrites recordings as canonical PCM WAV, the only kind every Python's ``wave`` module reads.

    PYTHONPATH=. python3 tools/normalize_recordings.py --recordings recordings            # report
    PYTHONPATH=. python3 tools/normalize_recordings.py --recordings recordings --in-place # convert

WHY THIS EXISTS. macOS ``afconvert`` writes WAVE_FORMAT_EXTENSIBLE (format tag 65534, a 40-byte
``fmt`` chunk) rather than plain WAVE_FORMAT_PCM (tag 1, 16-byte ``fmt``). The audio is identical;
only the header differs. Python's ``wave`` module gained support for the extensible header late, so
such a file opens on a recent Python and raises

    wave.Error: unknown format: 65534

on Python 3.9 and 3.10 — which is PACE-ICE's system Python and the Python of the Ubuntu 22.04
image this agent is graded in. A recording set that works on the laptop then fails on the cluster
with an error that says nothing about where it came from.

This tool parses the RIFF chunks by hand instead of through ``wave``, so it reads the extensible
header on any Python, and writes the file back through ``voice_kit.audio.write_wav``, which emits
the canonical 16-byte PCM header. THE SAMPLES ARE COPIED BYTE FOR BYTE: nothing is resampled,
requantised or re-encoded, so a converted recording is the same audio and the same WER.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from voice_kit.audio import (
    CHANNEL_COUNT,
    SAMPLE_RATE_HZ,
    SAMPLE_WIDTH_BYTES,
    write_wav,
)

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

# Where the SubFormat GUID starts in an extensible fmt chunk: 18 bytes of common fields, then
# wValidBitsPerSample (2) and dwChannelMask (4). Its first two bytes hold the real format code.
SUBFORMAT_OFFSET = 24


class NotCanonicalWav(ValueError):
    """Raised for a file this tool cannot safely rewrite."""


def read_chunks(wav_path: Path) -> tuple[bytes, bytes]:
    """Return the ``fmt`` and ``data`` chunk bodies of a RIFF/WAVE file, without using ``wave``."""
    raw = wav_path.read_bytes()

    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise NotCanonicalWav("not a RIFF/WAVE file")

    format_chunk = None
    data_chunk = None
    position = 12

    while position + 8 <= len(raw):
        chunk_id = raw[position : position + 4]
        size = int.from_bytes(raw[position + 4 : position + 8], "little")
        body = raw[position + 8 : position + 8 + size]

        if chunk_id == b"fmt ":
            format_chunk = body

        elif chunk_id == b"data":
            data_chunk = body

        # RIFF chunks are word aligned: an odd size is followed by one padding byte.
        position += 8 + size + (size % 2)

    if format_chunk is None or data_chunk is None:
        raise NotCanonicalWav("no fmt or no data chunk")

    return format_chunk, data_chunk


def describe_format(format_chunk: bytes) -> tuple[int, int, int, int]:
    """Return (format tag, channels, sample rate, bits per sample), resolving the extensible tag."""
    tag = int.from_bytes(format_chunk[0:2], "little")
    channels = int.from_bytes(format_chunk[2:4], "little")
    sample_rate = int.from_bytes(format_chunk[4:8], "little")
    bits = int.from_bytes(format_chunk[14:16], "little")

    if tag == WAVE_FORMAT_EXTENSIBLE:
        if len(format_chunk) < SUBFORMAT_OFFSET + 2:
            raise NotCanonicalWav("extensible header with no SubFormat")

        # The GUID's first two bytes carry the format this really is: 1 for PCM, 3 for float.
        actual = int.from_bytes(
            format_chunk[SUBFORMAT_OFFSET : SUBFORMAT_OFFSET + 2], "little"
        )

        if actual != WAVE_FORMAT_PCM:
            raise NotCanonicalWav(
                f"extensible SubFormat is {actual}, not PCM; this tool only rewrites headers"
            )

    elif tag != WAVE_FORMAT_PCM:
        raise NotCanonicalWav(f"format tag {tag} is neither PCM nor extensible PCM")

    return tag, channels, sample_rate, bits


def convert(wav_path: Path, in_place: bool, backup_directory: Path | None) -> str:
    """Report, and optionally rewrite, one recording. Returns a one-line verdict."""
    try:
        format_chunk, data_chunk = read_chunks(wav_path)
        tag, channels, sample_rate, bits = describe_format(format_chunk)

    except NotCanonicalWav as error:
        return f"CANNOT  {wav_path.name}: {error}"

    expected = (CHANNEL_COUNT, SAMPLE_RATE_HZ, SAMPLE_WIDTH_BYTES * 8)
    actual = (channels, sample_rate, bits)

    if actual != expected:
        return (
            f"CANNOT  {wav_path.name}: {sample_rate} Hz, {channels} channel(s), {bits}-bit; "
            "the shim needs 16000 Hz, 1 channel, 16-bit, so record or convert it again"
        )

    if tag == WAVE_FORMAT_PCM:
        return f"ok      {wav_path.name}: already canonical PCM"

    if not in_place:
        return (
            f"WOULD   {wav_path.name}: extensible header ({len(format_chunk)}-byte fmt) "
            "-> canonical PCM"
        )

    if backup_directory is not None:
        backup_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav_path, backup_directory / wav_path.name)

    write_wav(wav_path, data_chunk, SAMPLE_RATE_HZ)

    return f"FIXED   {wav_path.name}: extensible -> canonical PCM, {len(data_chunk)} bytes kept"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the settings of one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Actually rewrite the files. Without it this only reports what it would do.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Copy each file here before rewriting it.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Report or rewrite every recording in a directory."""
    arguments = parse_arguments(argv)
    wav_paths = sorted(arguments.recordings.glob("*.wav"))

    if not wav_paths:
        raise SystemExit(f"No .wav files in {arguments.recordings}.")

    verdicts = [convert(path, arguments.in_place, arguments.backup_dir) for path in wav_paths]

    for verdict in verdicts:
        print(verdict)

    cannot = [v for v in verdicts if v.startswith("CANNOT")]
    would = [v for v in verdicts if v.startswith("WOULD")]

    print("")
    print(f"{len(wav_paths)} file(s): {len(verdicts) - len(cannot) - len(would)} fine or fixed, "
          f"{len(would)} to fix, {len(cannot)} cannot be fixed here")

    if would:
        print("Re-run with --in-place to rewrite them.")

    return 1 if cannot else 0


if __name__ == "__main__":
    raise SystemExit(main())
