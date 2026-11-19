"""Sanity check: llm_complete is when the answer arrived, not when the last sentence was synthesised.

    python3 test/test_stream_timing.py

A fake server sends a two-sentence answer in 0.1 s; the consumer takes 0.3 s to "synthesise" each
sentence, as the turn loop does. The brain reads the stream on its own thread, so the answer must
be marked complete about 0.1 s after the request, long before the consumer is done (about 0.6 s).
No model and no server needed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain  # noqa: E402
import llm_client  # noqa: E402

SERVER_SECONDS_PER_DELTA = 0.025
SYNTHESIS_SECONDS = 0.3


class FakeClient:
    """Stands in for llm_client.ChatClient: streams four deltas, 25 ms apart."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def stream_chat(self, messages, **options):
        for text in ("Ten minutes ", "for the tea. ", "It is ", "running now."):
            time.sleep(SERVER_SECONDS_PER_DELTA)
            yield llm_client.Delta(text)


def check(label: str, got, expected) -> None:
    status = "PASS" if got == expected else "FAIL"
    print(f"{status}  {label}: got {got!r}, expected {expected!r}")
    assert got == expected


def main() -> int:
    brain.llm_client.ChatClient = FakeClient
    agent = brain.LlmBrain(brain.BrainSettings(model="fake", base_url="http://127.0.0.1:1/v1", nlu="rules"))

    started = brain.now_nanoseconds()
    sentences = []

    for sentence in agent.reply("Start a ten minute timer for the tea."):
        sentences.append(sentence)
        time.sleep(SYNTHESIS_SECONDS)  # what the turn loop spends synthesising before it asks again

    finished = brain.now_nanoseconds()
    complete = (agent.answer_complete_nanoseconds - started) / 1e9
    consumer = (finished - started) / 1e9

    check("two sentences", sentences, ["Ten minutes for the tea.", "It is running now."])
    print(f"      answer complete after {complete:.3f} s, consumer done after {consumer:.3f} s")
    check("complete when the stream ended, not after synthesis", complete < 2 * SYNTHESIS_SECONDS, True)
    check("complete after the first answer token", agent.answer_complete_nanoseconds >= agent.first_answer_token_nanoseconds, True)

    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
