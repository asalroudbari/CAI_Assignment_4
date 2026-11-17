"""The real dialogue manager, on a locally served chat model. Owned by Partner B.

Partner B: fill this file in. Everything you need to satisfy is in ``interfaces.Brain``, and the one
thing that is easy to get wrong is in its docstring: yield WHOLE SENTENCES as soon as each one is
complete, because the first yield is what the turn loop stamps as ``llm_first_token``.

You should not need to change ``run_agent.py``, ``interfaces.py`` or ``fakes.py`` to do this. If you
think you do, say so rather than editing them, because the turn loop is what the latency numbers are
measured through.

Until this is implemented, the agent runs with ``--fake-brain`` (or ``--fake-all``).
"""

from __future__ import annotations

from typing import Iterator

from interfaces import Brain


class LlmBrain:
    """Holds the dialogue state and asks a locally served chat model what to say next.

    The model name and the server URL are settings of the agent, never hard-coded. The toy knowledge
    base is small enough that how much of it goes into each prompt is a deliberate choice, and that
    choice shows up directly in the ``llm_start`` stage of the waterfall.
    """

    def warmup(self) -> None:
        """Reach the server, load the model and generate once, so no load cost lands inside a turn."""
        raise NotImplementedError(
            "LlmBrain.warmup is not implemented yet — see interfaces.Brain for the contract. "
            "Run with --fake-brain to use the stand-in in fakes.py."
        )

    def reset(self) -> None:
        """Drop the conversation state, because a new dialogue is starting."""
        raise NotImplementedError(
            "LlmBrain.reset is not implemented yet — see interfaces.Brain for the contract. "
            "Run with --fake-brain to use the stand-in in fakes.py."
        )

    def reply(self, text: str) -> Iterator[str]:
        """Yield the answer to ``text``, one whole sentence at a time, first sentence as early as possible."""
        raise NotImplementedError(
            "LlmBrain.reply is not implemented yet — see interfaces.Brain for the contract. "
            "Run with --fake-brain to use the stand-in in fakes.py."
        )


def build() -> Brain:
    """Return the real dialogue manager."""
    return LlmBrain()
