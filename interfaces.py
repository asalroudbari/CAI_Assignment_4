"""The module boundary between the turn loop and the three stages of the pipeline.

This file is the contract. ``run_agent.py`` knows nothing about faster-whisper, about a chat
model or about Piper; it only knows these three shapes. Every stage, real or fake, satisfies one
of them, so a stage can be swapped without touching the turn loop.

Partner A owns :class:`Asr` (``asr.py``). Partner B owns :class:`Brain` (``brain.py``) and
:class:`Tts` (``tts.py``). The fakes that stand in for all three until the real stages exist are
in ``fakes.py``, which Partner A owns.

Every timestamp in the trace comes from one monotonic clock, and three of the five are stamped by
the turn loop at the moments these methods hand something back. That is why the contract below is
about *when* a method returns as much as about *what* it returns.
"""

from __future__ import annotations

from typing import Iterator, Protocol


class Asr(Protocol):
    """Turns the audio the shim fed us into the text the user said."""

    def warmup(self) -> None:
        """Do the expensive one-time work: load the model and run one inference.

        The README is explicit that start-up cost may never land inside a turn. The turn loop calls
        this once, before the first timed turn, so that the first real transcription costs what the
        hundredth costs.
        """

    def transcribe(self, pcm_bytes: bytes) -> str:
        """Return what was said in ``pcm_bytes``, as a non-empty string.

        ``pcm_bytes`` is 16 kHz mono 16-bit little-endian PCM: exactly the bytes the shim fed to
        ``on_chunk`` up to the end of user speech, concatenated in order and nothing else.

        The turn loop stamps ``asr_final`` the moment this returns, so anything done here is counted
        as ASR time, honestly. Returning an empty or whitespace-only string makes the checker reject
        the turn (``extras.transcript is missing or empty``), so a stage that cannot hear anything
        should still return a short honest string rather than ``""``.
        """


class Brain(Protocol):
    """Decides what to say back, one whole sentence at a time."""

    def warmup(self) -> None:
        """Do the expensive one-time work: connect to the server, load the model, generate once."""

    def reset(self) -> None:
        """Forget the conversation so far; the next reply starts a new dialogue.

        The turn loop calls this at the first turn of every dialogue, including the first one. A
        recording set is several short conversations, not one long one: the timers, reminders and
        bookings of one dialogue must not be visible in the next, or the agent answers "what
        reminders do I have" with another dialogue's reminders and the turn is marked a failure by
        whoever is peer-testing it.

        Within a dialogue this is NOT called, and the state kept between turns is the whole point:
        "Actually make it fifteen" only means something if the previous turn is still in hand.

        Which recordings belong to one dialogue comes from the file name (``name_d01_t01.wav``), so
        this method only has to clear state, never to work out where the boundary is.
        """

    def reply(self, text: str) -> Iterator[str]:
        """Yield the answer to ``text`` as WHOLE SENTENCES, not tokens.

        The turn loop stamps ``llm_first_token`` when the *first* sentence is yielded and
        ``llm_complete`` when the iterator is exhausted. Two consequences:

        * Yield the first sentence as early as it is complete. Buffering the whole answer and then
          yielding it moves ``llm_first_token`` to the end of generation and pushes the time to first
          audio out by the whole generation time. Stream from the model and cut at sentence ends.
        * Yield at least one non-empty sentence. A reply that yields nothing makes the turn loop raise,
          and an empty ``extras.reply`` makes the checker reject the turn.

        Each yielded sentence goes straight to the TTS, so yield text that is ready to be spoken: no
        markdown, no bullet characters, no stray partial clause.
        """


class Tts(Protocol):
    """Turns one sentence into the PCM the shim captures as the agent's voice."""

    sample_rate_hz: int
    """The sample rate of the chunks :meth:`stream` yields.

    The turn loop reads this and hands it to ``ReplaySession``, so the captured WAV is written at the
    rate the audio actually has. Piper's lessac medium voice is 22050.
    """

    def warmup(self) -> None:
        """Do the expensive one-time work: load the voice and synthesise once."""

    def stream(self, sentence: str) -> Iterator[bytes]:
        """Yield ``sentence`` as 16-bit mono PCM chunks at :attr:`sample_rate_hz`.

        THE FIRST CHUNK MUST BE AUDIBLE. The shim stamps ``tts_first_audio`` on the first chunk whose
        root mean square reaches 200, and that stamp is the end of the headline latency number of this
        assignment. A leading chunk of padding silence does not delay the sound a listener hears, but
        it does delay the stamp, so it inflates the reported time to first audio by exactly its own
        length. Write only the chunks the synthesiser actually produced, with nothing in front.

        Yield as soon as each chunk exists rather than collecting them, for the same reason: the first
        yield is what stops the clock.
        """
