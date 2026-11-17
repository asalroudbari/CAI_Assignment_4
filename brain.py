"""The real dialogue manager, on a locally served chat model. Owned by Partner B.

WHAT THIS STAGE IS MEASURED ON
------------------------------
The turn loop stamps ``llm_first_token`` when this iterator yields its FIRST SENTENCE, not when the
first token arrives from the server, and that stamp is almost the whole time to first audio once the
ASR is 39 ms. Three decisions in this file are therefore latency decisions, not style:

* **Stream, and cut at the first sentence end.** The reply is assembled sentence by sentence as the
  deltas arrive (:class:`SentenceSplitter`), so the first short sentence is spoken while the rest is
  still being generated.
* **Ask for a short first sentence, and cap the whole reply.** The model is told to answer in one or
  two short sentences and ``max_tokens`` enforces it. A first sentence of 25 words costs its own
  generation time before any audio exists.
* **Send a small prompt.** The knowledge base is 6 kB, around 1500 tokens, and every token of it is
  re-read on every prefill. By default only the facts that match the question are sent
  (``--llm-kb-mode relevant``); ``full`` and ``none`` exist so the choice can be measured rather than
  asserted. The dialogue history is carried as plain turns, without the facts those turns were
  answered from, so the prompt does not grow by a knowledge base per turn.

WHAT THIS STAGE IS NOT ASKED TO DO
----------------------------------
It does not own the timers, the reminders or the bookings, and it is never asked to do arithmetic or
to search the knowledge base. ``dialogue_state`` does both in Python and hands the answer down as
FACTS. See the head of that file for why.

SETTINGS, NEVER CONSTANTS
-------------------------
The model name and the server URL arrive from ``LLM_MODEL`` and ``LLM_BASE_URL`` or from the command
line (``run_agent.py --llm-model --llm-base-url``), because the README requires it and because the
development machine and the graded H100 node are not the same computer.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

import knowledge_base as knowledge_base_module
import llm_client
from dialogue_state import DialogueState
from interfaces import Brain
from voice_kit import now_nanoseconds

# The smallest of the four models the assignment allows. 3B is the default because it answers these
# task families as well as 8B does and its first token comes sooner, which is the graded number.
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

MODEL_ENVIRONMENT_VARIABLE = llm_client.MODEL_ENVIRONMENT_VARIABLE
KB_MODE_ENVIRONMENT_VARIABLE = "LLM_KB_MODE"

KB_MODES = ("relevant", "full", "none")

# What the model is told it is. Short on purpose: every line here is prefill on every turn.
SYSTEM_PROMPT = """You are the voice assistant of {venue}. You are being listened to, not read.

Answer in at most two short sentences. Make the FIRST sentence the short one: it is spoken while the
rest is still being written.

Rules:
- Everything you may state as fact is in the FACTS block. Never invent an hour, a price, a room, a
  rule or an availability.
- If the FACTS do not answer the question, say you do not have that, and name one or two things you
  do know about.
- WHAT JUST HAPPENED has already been done. Say it back as done, in your own words. Never offer to
  do it again and never contradict it.
- Repeat back the values you acted on: the length, the time, the room, the day, the name.
- If MISSING appears, ask exactly one short question for that value, and say nothing else.
- Speak plainly. No markdown, no bullet points, no emoji, no asterisks, no stage directions."""

# How a turn is handed to the model: what is true, what is known, and what was said.
TURN_TEMPLATE = """FACTS (true right now, say these back):
{facts}

KNOWLEDGE BASE:
{knowledge}

The user just said: "{transcript}\""""

NOTHING_RETRIEVED = "Nothing in the knowledge base is about this."

# Said when the server returns an empty answer. It is spoken rather than raised because a turn that
# raises is a turn with no reply at all, and a peer testing the agent hears silence; it is logged to
# stderr so a run that needed it cannot look like a clean run.
EMPTY_ANSWER_REPLY = "Sorry, I did not catch that. Could you say it again?"

# What warmup says, so the first timed turn is not the first time this path runs.
WARMUP_TRANSCRIPT = "What time do you close on Sunday?"

SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*")

# Full stops that do not end a sentence. Splitting on them sends half a clause to the TTS.
ABBREVIATIONS = frozenset(
    "mr mrs ms dr st jr sr prof vs etc eg ie approx dept no a.m p.m am pm u.s".split()
)

# Markdown the model may emit even when told not to. It is removed rather than spoken, because a
# TTS reads an asterisk aloud or chokes on it.
MARKDOWN = re.compile(r"[*_`#]+")


@dataclass(frozen=True)
class BrainSettings:
    """Everything about this stage that a run is allowed to change."""

    model: str = field(
        default_factory=lambda: llm_client.default_model() or DEFAULT_MODEL
    )
    base_url: str = field(default_factory=llm_client.default_base_url)
    api_key: str | None = None

    # Greedy. A spoken answer that changes between two runs of the same recording cannot be
    # compared, and the report has to be reproducible from the shipped trace.
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 20_241_118

    # Two short sentences is about 40 tokens. The cap is what stops a model that has decided to
    # list all seven rooms from generating for two seconds after the first sentence was spoken —
    # which is off the time-to-first-audio path but not off the user's clock.
    max_tokens: int = 120

    # How many earlier turns of this dialogue go into the prompt. Six covers the longest dialogue
    # in the task families; the facts of those turns are not carried, only what was said.
    history_turns: int = 6

    kb_mode: str = field(
        default_factory=lambda: os.environ.get(KB_MODE_ENVIRONMENT_VARIABLE, "relevant")
    )
    kb_facts: int = 4
    kb_path: Path = knowledge_base_module.DEFAULT_PATH

    timeout_seconds: float = 60.0

    # gpt-oss-20b reasons before it answers and those tokens are never spoken, so they are pure
    # latency. "low" is the setting to report if that model is used; it is ignored by the others.
    reasoning_effort: str | None = None

    # A sentence longer than this is cut at its last comma, so one runaway sentence cannot hold the
    # first audio back for the whole generation.
    max_sentence_characters: int = 160

    def resolved(self) -> "BrainSettings":
        """Return these settings with anything invalid rejected early rather than at turn one."""
        if self.kb_mode not in KB_MODES:
            raise ValueError(f"--llm-kb-mode must be one of {KB_MODES}, not {self.kb_mode!r}")

        return replace(self, kb_path=Path(self.kb_path))

    def describe(self) -> str:
        """Return a one-line summary for the banner."""
        return (
            f"{self.model} at {self.base_url} temperature={self.temperature} "
            f"max_tokens={self.max_tokens} kb_mode={self.kb_mode} kb_facts={self.kb_facts} "
            f"history_turns={self.history_turns}"
            + (f" reasoning_effort={self.reasoning_effort}" if self.reasoning_effort else "")
        )


class SentenceSplitter:
    """Turns a stream of token deltas into whole sentences, as early as each one is whole.

    The first sentence is what stops the latency clock, so this emits as soon as a sentence end is
    followed by anything at all — it never waits for the next sentence to start.
    """

    def __init__(self, max_characters: int = 160) -> None:
        self.max_characters = max_characters
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add ``text`` to the buffer and return every sentence that is now complete."""
        self.buffer += text
        sentences: list[str] = []

        while True:
            sentence = self._take_one()

            if sentence is None:
                break

            sentences.append(sentence)

        return sentences

    def flush(self) -> str | None:
        """Return whatever is left when the stream ends, which is usually the last sentence."""
        remaining = clean_for_speech(self.buffer)
        self.buffer = ""

        return remaining or None

    def _take_one(self) -> str | None:
        """Return the next complete sentence in the buffer, or ``None``."""
        for match in SENTENCE_END.finditer(self.buffer):
            end = match.end()

            # A sentence is only known to have ended once a character after it has arrived: "3." in
            # "3.5" and "Dr." in "Dr. Lee" both look like ends until the next character is seen.
            if end >= len(self.buffer):
                return None

            # A number keeps its own full stop: in "3.5" the character after the stop is a
            # digit, not a space, so a decimal never reaches the test below.
            if self.buffer[end] not in " \n\t":
                continue

            head = self.buffer[:end]

            if self._is_abbreviation(head):
                continue

            sentence = clean_for_speech(head)
            self.buffer = self.buffer[end:].lstrip()

            if sentence:
                return sentence

            return None

        # No end in sight and the sentence has run long: cut it at a clause boundary so the first
        # audio is not held back by a model that is writing a paragraph.
        if len(self.buffer) > self.max_characters:
            cut = max(self.buffer.rfind(", ", 0, self.max_characters),
                      self.buffer.rfind("; ", 0, self.max_characters))

            if cut > 40:
                sentence = clean_for_speech(self.buffer[:cut])
                self.buffer = self.buffer[cut + 1:].lstrip()

                if sentence:
                    return sentence

        return None

    @staticmethod
    def _is_abbreviation(head: str) -> bool:
        """Return whether the full stop at the end of ``head`` belongs to an abbreviation."""
        last_word = re.split(r"[\s(]", head)[-1].rstrip(".!?\"')]").lower()

        return last_word in ABBREVIATIONS


def clean_for_speech(text: str) -> str:
    """Return ``text`` with anything a synthesiser should not read aloud removed."""
    without_markdown = MARKDOWN.sub("", text)
    collapsed = re.sub(r"\s+", " ", without_markdown)

    return collapsed.strip(" -\n\t")


class LlmBrain:
    """Holds the dialogue state and asks a locally served chat model what to say next."""

    def __init__(self, settings: BrainSettings | None = None) -> None:
        self.settings = (settings or BrainSettings()).resolved()
        self.knowledge_base = knowledge_base_module.load(self.settings.kb_path)
        self.state = DialogueState(self.knowledge_base)
        self.client = llm_client.ChatClient(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            timeout_seconds=self.settings.timeout_seconds,
        )
        self.system_prompt = SYSTEM_PROMPT.format(venue=self.knowledge_base.venue_name)
        self.history: list[tuple[str, str]] = []

        # Read by the turn loop into extras.first_reasoning_token_nanoseconds, for a model that
        # thinks before it answers. It stays None for a model that does not.
        self.first_reasoning_token_nanoseconds: int | None = None

        # Set by tools/try_brain.py to see the prompt of each turn. It is called from inside
        # reply(), AFTER the state has been updated and before the request goes out, because the
        # only other way to see the prompt is to run the state update a second time — which starts
        # a second timer, books a second room, and shows a prompt that the turn never sent.
        self.on_prompt = None

    def describe(self) -> str:
        """Return what this stage is running, for the banner."""
        return self.settings.describe()

    # -- the contract ---------------------------------------------------------------------------

    def warmup(self) -> None:
        """Reach the server, check it serves the model, and run one whole reply through this path."""
        served = self.client.models()

        if not served:
            raise llm_client.LlmError(
                f"the server at {self.client.describe()} is serving no model at all."
            )

        if self.settings.model not in served:
            raise llm_client.LlmError(
                f"the server at {self.client.describe()} serves {served}, not "
                f"{self.settings.model!r}. Pass --llm-model or set LLM_MODEL to one it serves."
            )

        # The same path a timed turn takes, so the first timed turn pays for nothing one-off: the
        # connection, the first prefill, and whatever vLLM compiles on its first request.
        for _ in self.reply(WARMUP_TRANSCRIPT):
            pass

        self.reset()

    def reset(self) -> None:
        """Drop the conversation state, because a new dialogue is starting."""
        self.state.reset()
        self.history.clear()
        self.first_reasoning_token_nanoseconds = None

    def reply(self, text: str) -> Iterator[str]:
        """Yield the answer to ``text``, one whole sentence at a time, first sentence as early as possible."""
        transcript = (text or "").strip()
        self.first_reasoning_token_nanoseconds = None

        # Python reads the utterance and applies what it clearly asks for BEFORE the model is
        # called, so the prompt can state what is now true rather than ask the model to work it out.
        self.state.observe(transcript)

        messages = self._messages(transcript)

        if self.on_prompt is not None:
            self.on_prompt(messages)

        splitter = SentenceSplitter(self.settings.max_sentence_characters)
        spoken: list[str] = []

        for delta in self.client.stream_chat(messages, **self._options()):
            if delta.is_reasoning:
                if self.first_reasoning_token_nanoseconds is None:
                    self.first_reasoning_token_nanoseconds = now_nanoseconds()

                continue

            for sentence in splitter.feed(delta.text):
                spoken.append(sentence)

                yield sentence

        last = splitter.flush()

        if last:
            spoken.append(last)

            yield last

        if not spoken:
            print(
                f"brain: the model returned nothing for {transcript!r}; speaking the fallback.",
                file=sys.stderr,
                flush=True,
            )
            spoken.append(EMPTY_ANSWER_REPLY)

            yield EMPTY_ANSWER_REPLY

        self.history.append((transcript, " ".join(spoken)))

    # -- the prompt -----------------------------------------------------------------------------

    def _options(self) -> dict:
        """Return the generation settings sent with every request."""
        options = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
            "seed": self.settings.seed,
        }

        if self.settings.reasoning_effort:
            options["reasoning_effort"] = self.settings.reasoning_effort

        return options

    def _messages(self, transcript: str) -> list[dict]:
        """Return the whole prompt: the rules, what was said before, and this turn."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # The history carries what was SAID, not the facts those turns were answered from. Carrying
        # the facts would put a knowledge base in the prompt once per turn of the dialogue.
        for said, answered in self.history[-self.settings.history_turns:]:
            messages.append({"role": "user", "content": said})
            messages.append({"role": "assistant", "content": answered})

        messages.append(
            {
                "role": "user",
                "content": TURN_TEMPLATE.format(
                    facts="\n".join(self.state.fact_lines()),
                    knowledge=self._knowledge_lines(transcript),
                    transcript=transcript,
                ),
            }
        )

        return messages

    def _knowledge_lines(self, transcript: str) -> str:
        """Return the knowledge base lines this turn gets, under the configured policy."""
        if self.settings.kb_mode == "none":
            return NOTHING_RETRIEVED

        if self.settings.kb_mode == "full":
            return "\n".join(self.knowledge_base.all_lines())

        facts = self.knowledge_base.relevant(transcript, self.settings.kb_facts)

        if not facts:
            return NOTHING_RETRIEVED

        return "\n".join(fact.line for fact in facts)


def build(settings: BrainSettings | None = None) -> Brain:
    """Return the real dialogue manager."""
    return LlmBrain(settings)
