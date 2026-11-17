"""Stand-ins for the three stages, so the pipeline can be proved before any real stage exists.

Owned by Partner A. These exist for one reason: to show that the turn loop stamps the five events in
the right order, fills the extras the checker wants, and keeps the shim's feed lag down — none of
which needs a model to be true. Nothing here produces a number worth reporting. The turn loop writes
``extras.stage_implementations`` on every record, so a trace made with these can never be mistaken
for a measurement.

Each class satisfies the matching Protocol in ``interfaces.py``.
"""

from __future__ import annotations

import math
import time
from typing import Iterator

from interfaces import Asr, Brain, Tts

FAKE_TRANSCRIPT = "this is a fake transcript from fakes dot fake asr"

FAKE_FIRST_SENTENCE = "This is a fake reply from FakeBrain."
FAKE_SECOND_SENTENCE = "No model produced it, so this turn is not a real measurement."

# A 20 ms chunk, the same length the shim feeds in, at the rate Piper's lessac medium voice uses.
FAKE_TTS_SAMPLE_RATE_HZ = 22_050
FAKE_TONE_HERTZ = 440.0
FAKE_TONE_MILLISECONDS = 20

# The shim calls a chunk audible at a root mean square of 200. A sine of this amplitude has a root
# mean square of 8000 / sqrt(2), about 5657, so the first chunk is unambiguously the first audio.
FAKE_TONE_AMPLITUDE = 8000

# One chunk of tone stands for this many characters of the sentence.
CHARACTERS_PER_TONE_CHUNK = 50


class FakeAsr:
    """Returns a canned transcript and ignores the audio entirely."""

    def __init__(self, transcript: str = FAKE_TRANSCRIPT) -> None:
        self.transcript = transcript

    def warmup(self) -> None:
        """Transcribe one short buffer of silence, the same path a timed turn takes."""
        self.transcribe(bytes(640))

    def transcribe(self, pcm_bytes: bytes) -> str:
        """Return the canned transcript; no delay, so the ASR stage of a fake run is near zero."""
        return self.transcript


class FakeBrain:
    """Yields two canned sentences, the first after a short delay that stands in for prefill."""

    def __init__(
        self,
        first_sentence_delay_seconds: float = 0.15,
        inter_sentence_delay_seconds: float = 0.05,
    ) -> None:
        self.first_sentence_delay_seconds = first_sentence_delay_seconds
        self.inter_sentence_delay_seconds = inter_sentence_delay_seconds
        self.reset_count = 0

    def reset(self) -> None:
        """Forget the dialogue so far. The fake holds no state, so this only counts the calls,
        which is how the turn loop's dialogue boundaries are checked without a real brain."""
        self.reset_count += 1

    def warmup(self) -> None:
        """Run one whole reply with the delays set to zero.

        The delays stand in for model time, not for load time, so waiting them out here would only
        make start-up slower without exercising anything a timed turn has not already exercised.
        """
        list(self._sentences(0.0, 0.0))

    def reply(self, text: str) -> Iterator[str]:
        """Yield the two canned sentences, paced by the configured delays."""
        return self._sentences(
            self.first_sentence_delay_seconds, self.inter_sentence_delay_seconds
        )

    def _sentences(
        self, first_delay_seconds: float, inter_delay_seconds: float
    ) -> Iterator[str]:
        # time.sleep releases the interpreter lock, so this delay does not hold up the shim's feed
        # thread the way a busy loop would.
        if first_delay_seconds > 0:
            time.sleep(first_delay_seconds)

        yield FAKE_FIRST_SENTENCE

        if inter_delay_seconds > 0:
            time.sleep(inter_delay_seconds)

        yield FAKE_SECOND_SENTENCE


class FakeTts:
    """Yields 20 ms of a 440 Hz tone for every 50 characters of the sentence."""

    sample_rate_hz = FAKE_TTS_SAMPLE_RATE_HZ

    def __init__(self) -> None:
        # The tone is built once, here, and the same bytes object is yielded again and again. Building
        # it per chunk would burn main-thread CPU while the shim's feed thread is still pacing the
        # recording, and because that thread needs the interpreter lock to wake on time, the cost would
        # land in extras.max_feed_lag_milliseconds instead of anywhere honest.
        self.tone_chunk = self._build_tone_chunk()

    def warmup(self) -> None:
        """Consume one stream, the same path a timed turn takes."""
        list(self.stream("warmup"))

    def stream(self, sentence: str) -> Iterator[bytes]:
        """Yield one 20 ms tone chunk per 50 characters, at least one, the first already audible."""
        chunk_count = max(
            1, math.ceil(len(sentence) / CHARACTERS_PER_TONE_CHUNK)
        )

        for _ in range(chunk_count):
            yield self.tone_chunk

    @classmethod
    def _build_tone_chunk(cls) -> bytes:
        """Return one 20 ms chunk of 440 Hz tone as little-endian 16-bit mono PCM."""
        sample_count = cls.sample_rate_hz * FAKE_TONE_MILLISECONDS // 1000
        samples = bytearray()

        for index in range(sample_count):
            angle = 2.0 * math.pi * FAKE_TONE_HERTZ * index / cls.sample_rate_hz
            value = int(FAKE_TONE_AMPLITUDE * math.sin(angle))
            samples += value.to_bytes(2, "little", signed=True)

        return bytes(samples)


def build_asr() -> Asr:
    """Return the stand-in speech recognition stage."""
    return FakeAsr()


def build_brain() -> Brain:
    """Return the stand-in dialogue manager."""
    return FakeBrain()


def build_tts() -> Tts:
    """Return the stand-in speech synthesis stage."""
    return FakeTts()
