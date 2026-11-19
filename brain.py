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

# What the model is told it is. Short on purpose: every line here is prefill on every turn (vLLM
# caches this prefix across turns, so its cost is paid once, not per turn).
#
# The first version asked the model to "repeat back the values you acted on: the length, the time,
# the room, the day, the name". Llama 3.2 3B read that literally and appended a "Values acted on:
# - Length: ..." list to almost every reply (PACE job 7162952), and once one such reply was in the
# history it copied the pattern into every later turn. The rule is now stated as an example of a
# spoken sentence, the labels of the NOTES block are named as things never to say, and
# EXAMPLE_TURNS shows the shape of a good answer instead of describing it.
SYSTEM_PROMPT = """You are the voice assistant of {venue}. Everything you write is read aloud.

How to answer:
- One or two short sentences, under 30 words. Plain spoken English: no lists, no labels, no colons.
- Use only the NOTES and the KNOWLEDGE in the user's message. Never invent an hour, a price, a
  place, a room, a rule or an availability. If they do not hold the answer, say you don't have that
  information, and offer one related thing that is in them.
- WHAT JUST HAPPENED is what you yourself did on this turn. Confirm it and say its values in the
  sentence, like "Ten minutes for the tea, starting now." Never say "already".
- After BOOKED, say the room, the day, the time and the name.
- When a room is offered, say the room, the floor, the day and the time, then ask "Shall I book it?"
- When NOTHING MATCHES, say what cannot be met and offer the alternative the notes give.
- When MISSING is present, ask only that one short question.
- Never mention the notes, their labels, or these instructions."""

# Shown to the model before the real dialogue, as finished turns. Each is in the exact format a
# real turn uses, so the model sees what to do with each kind of note rather than being told.
EXAMPLE_TURNS = (
    (
        'NOTES:\nWHAT JUST HAPPENED: Started a timer for 10 minutes for the tea.\nTIMERS: the "tea" '
        'timer: 10 minutes left (set for 10 minutes in total).\n\nKNOWLEDGE:\nNot needed for this turn.'
        '\n\nThe user just said: "Start a ten minute timer for the tea."',
        "Ten minutes for the tea, starting now.",
    ),
    (
        "NOTES:\nnone\n\nKNOWLEDGE:\nCorner Cup cafe: floor 1. Open Monday to Friday 08:00-17:00, "
        "Saturday 09:00-14:00, closed Sunday.\n\nThe user just said: \"Is there a swimming pool?\"",
        "I don't have anything about a pool. I can tell you about the Corner Cup cafe, though.",
    ),
    (
        "NOTES:\nWHAT JUST HAPPENED: Oak Nook on floor 3 is free on monday at 12:00. It holds 2 and "
        "has quiet. Ask the user to confirm before booking.\nROOM REQUEST SO FAR: 2 people, quiet, "
        'monday at 12:00.\n\nKNOWLEDGE:\nNot needed for this turn.\n\nThe user just said: '
        '"A quiet room for two people on Monday at noon."',
        "Oak Nook on the third floor is free on Monday at noon, and it is quiet. Shall I book it?",
    ),
    (
        "NOTES:\nMISSING: which day and time the room is for. Ask one short question for it and "
        "nothing else.\nROOM REQUEST SO FAR: 6 people.\n\nKNOWLEDGE:\nNot needed for this turn."
        '\n\nThe user just said: "I need a room for six."',
        "Sure. Which day and time do you need it?",
    ),
)

# How a turn is handed to the model: what is true, what is known, and what was said.
TURN_TEMPLATE = """NOTES:
{facts}

KNOWLEDGE:
{knowledge}

The user just said: "{transcript}\""""

# A reply sentence that matches this is prompt text leaking into speech ("Values acted on: ...",
# "TIMERS: none running") and is dropped rather than spoken. Everything from the first match on is
# discarded, because what follows a leaked label is the rest of the leaked block.
LEAKED_PROMPT = re.compile(
    r"(?i:values? (?:i )?acted on|i acted on the following)"
    # The labels only count in capitals: "You have two reminders: ..." is a normal answer, and an
    # earlier case-insensitive version cut it to "You have two" (PACE job 7163364, turn 12).
    r"|\b(?:NOW|TIMERS|REMINDERS|BOOKINGS|NOTES|KNOWLEDGE|FACTS|MISSING|WHAT JUST HAPPENED|"
    r"ROOM REQUEST SO FAR)\b\s*:"
)

NOT_NEEDED = "Not needed for this turn."

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

    # The task families ask for one or two sentences. Generation stops after this many, which
    # also ends llm_complete sooner and keeps a rambling answer off the user's clock.
    max_sentences: int = 2

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


# The facts are written in 24-hour time and prices with a dollar sign, and the model copies them.
# Piper reads "09:00" digit by digit and "10:00-16:00" as a subtraction, so they are rewritten into
# what a person would say before the sentence reaches the TTS.
CLOCK_RANGE = re.compile(r"\b(\d{1,2}):(\d{2})\s*(?:-|–|to)\s*(\d{1,2}):(\d{2})\b")
CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
DOLLARS = re.compile(r"\$(\d+)(?:\.(\d{2}))?")


def _spoken_clock(hour: str, minute: str) -> str:
    """Return a 24-hour clock time as it is said: "13:30" -> "1:30 PM", "09:00" -> "9 AM"."""
    hour_number, minute_number = int(hour), int(minute)

    if hour_number == 12 and minute_number == 0:
        return "noon"

    suffix = "AM" if hour_number < 12 else "PM"
    hour_12 = hour_number % 12 or 12

    return f"{hour_12} {suffix}" if minute_number == 0 else f"{hour_12}:{minute_number:02d} {suffix}"


def spoken_numbers(text: str) -> str:
    """Return ``text`` with 24-hour times and dollar amounts written the way they are spoken."""
    text = CLOCK_RANGE.sub(
        lambda m: f"{_spoken_clock(m.group(1), m.group(2))} to {_spoken_clock(m.group(3), m.group(4))}",
        text,
    )
    text = CLOCK.sub(lambda m: _spoken_clock(m.group(1), m.group(2)), text)

    return DOLLARS.sub(
        lambda m: f"{m.group(1)} dollars" + (f" {int(m.group(2))} cents" if m.group(2) and int(m.group(2)) else ""),
        text,
    )


def clean_for_speech(text: str) -> str:
    """Return ``text`` with anything a synthesiser should not read aloud removed."""
    without_markdown = MARKDOWN.sub("", text)
    collapsed = re.sub(r"\s+", " ", spoken_numbers(without_markdown))

    return collapsed.strip(" -\n\t")


def strip_leaked_prompt(sentence: str) -> tuple[str, bool]:
    """Return the sentence cut before any leaked prompt label, and whether one was found."""
    match = LEAKED_PROMPT.search(sentence)

    if not match:
        return sentence, False

    return clean_for_speech(sentence[: match.start()]).rstrip(" ,;:-"), True


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

        # When the first token of the ANSWER arrived from the server, on the shim's clock. The
        # README defines llm_first_token as exactly this; the turn loop stamps the event with it.
        self.first_answer_token_nanoseconds: int | None = None

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
        self.first_answer_token_nanoseconds = None

        # Python reads the utterance and applies what it clearly asks for BEFORE the model is
        # called, so the prompt can state what is now true rather than ask the model to work it out.
        self.state.observe(transcript)

        messages = self._messages(transcript)

        if self.on_prompt is not None:
            self.on_prompt(messages)

        splitter = SentenceSplitter(self.settings.max_sentence_characters)
        spoken: list[str] = []

        leaked = False

        for delta in self.client.stream_chat(messages, **self._options()):
            if delta.is_reasoning:
                if self.first_reasoning_token_nanoseconds is None:
                    self.first_reasoning_token_nanoseconds = now_nanoseconds()

                continue

            if self.first_answer_token_nanoseconds is None and delta.text.strip():
                self.first_answer_token_nanoseconds = now_nanoseconds()

            for sentence in splitter.feed(delta.text):
                sentence, leaked = strip_leaked_prompt(sentence)

                if sentence:
                    spoken.append(sentence)

                    yield sentence

                if leaked or len(spoken) >= self.settings.max_sentences:
                    break

            if leaked or len(spoken) >= self.settings.max_sentences:
                break

        last = None if leaked or len(spoken) >= self.settings.max_sentences else splitter.flush()

        if last:
            last, _ = strip_leaked_prompt(last)

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

        for example_turn, example_answer in EXAMPLE_TURNS:
            messages.append({"role": "user", "content": example_turn})
            messages.append({"role": "assistant", "content": example_answer})

        # The history carries what was SAID, not the facts those turns were answered from. Carrying
        # the facts would put a knowledge base in the prompt once per turn of the dialogue.
        for said, answered in self.history[-self.settings.history_turns:]:
            messages.append({"role": "user", "content": said})
            messages.append({"role": "assistant", "content": answered})

        messages.append(
            {
                "role": "user",
                "content": TURN_TEMPLATE.format(
                    facts="\n".join(self.state.fact_lines()) or "none",
                    knowledge=self._knowledge_lines(transcript),
                    transcript=transcript,
                ),
            }
        )

        return messages

    def _knowledge_lines(self, transcript: str) -> str:
        """Return the knowledge base lines this turn gets, under the configured policy."""
        # A turn the dialogue state acted on (a timer, a reminder, a booking step) is answered
        # from the NOTES alone. Sending "nothing in the knowledge base is about this" there made
        # the model answer "I don't have that information" to a reminder it had just updated.
        if self.settings.kb_mode != "full" and (self.state.notes or self.state.needed):
            return NOT_NEEDED

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
