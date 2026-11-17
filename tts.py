"""The real speech synthesis stage, on Piper. Owned by Partner B.

Partner B: fill this file in. The contract is ``interfaces.Tts``, and the one thing that silently
costs latency is in its docstring: the first chunk you yield must already be audible. The shim stamps
``tts_first_audio`` on the first chunk whose root mean square reaches 200, so a leading chunk of
padding silence inflates the headline number of this assignment by its own length.

You should not need to change ``run_agent.py``, ``interfaces.py`` or ``fakes.py`` to do this.

Until this is implemented, the agent runs with ``--fake-tts`` (or ``--fake-all``).
"""

from __future__ import annotations

from typing import Iterator

from interfaces import Tts

# Piper's en_US lessac medium voice, the one in the shared cache. If you choose a different voice,
# change this to its rate: the turn loop hands it to the shim so the captured WAV is written right.
PIPER_SAMPLE_RATE_HZ = 22_050


class PiperTts:
    """Synthesises one sentence at a time with a local Piper voice."""

    sample_rate_hz = PIPER_SAMPLE_RATE_HZ

    def warmup(self) -> None:
        """Load the voice and synthesise one short sentence, so no load cost lands inside a turn."""
        raise NotImplementedError(
            "PiperTts.warmup is not implemented yet — see interfaces.Tts for the contract. "
            "Run with --fake-tts to use the stand-in in fakes.py."
        )

    def stream(self, sentence: str) -> Iterator[bytes]:
        """Yield 16-bit mono PCM chunks of ``sentence``; the first chunk must already be audible."""
        raise NotImplementedError(
            "PiperTts.stream is not implemented yet — see interfaces.Tts for the contract. "
            "Run with --fake-tts to use the stand-in in fakes.py."
        )


def build() -> Tts:
    """Return the real speech synthesis stage."""
    return PiperTts()
