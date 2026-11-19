"""Reads what the user asked for with the LLM, as JSON actions the dialogue state then carries out.

The rules in ``dialogue_state`` only act on phrasings they were written for: "Could you time my eggs
for seven minutes?" or "just me and one other person" were left unread. Here the model does the
reading, and only the reading. It returns a list of actions in a fixed JSON schema (the server
constrains decoding to it), and ``DialogueState.apply_actions`` carries them out: the timers, the
arithmetic, the room search and the confirmation stay in Python, as before.

This is a second LLM call on every turn, made before the reply is generated, so its time lands in
the LLM start stage of the waterfall (asr_final -> llm_first_token).
"""

from __future__ import annotations

import datetime
import re
import json
import time

import llm_client
import spoken_text
from dialogue_state import NEEDED_BOOKING_NAME, NEEDED_IN_WORDS, DialogueState

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
PARTS_OF_DAY = ("morning", "afternoon", "evening")
LIST_KINDS = ("timers", "reminders", "bookings", "all")

INSTRUCTIONS = """Turn what the user said to a voice assistant into JSON actions. Never answer the user.

Actions: start_timer, change_timer (new total length), add_to_timer, cancel_timer, time_left, set_reminder, change_reminder, cancel_reminder, list, book_room (any detail of a room booking, including the name to book under), confirm_booking (yes to the proposed room), cancel_booking, question (anything else, such as a question about the venue).

Rules:
- Include every value the user said in this utterance (minutes, label, subject, time, people, features, day, name). Never add a value they did not say, and never copy one from the examples.
- minutes: "half an hour" is 30, "an hour and a half" is 90. "Add five more minutes" is add_to_timer with minutes 5, the time added, not the new total.
- people counts everyone: "me and one other person" is 2.
- time is 24-hour "HH:MM", and only when the user said a time of day. "noon" is 12:00. An hour from 1 to 6 with no "am" is in the afternoon: "at two" is 14:00.
- features are what the room must have or be: projector, whiteboard, tv screen, video call kit, microphone, stage, quiet.
- day (bookings) is a weekday name. "tomorrow" is the day after TODAY.
- date (reminders) only when the user said a date, written as they said it.
- room is set only when the user names one of the rooms, or says "it" for the room ASSISTANT LAST SAID. name is the person the booking is under: a real name the user said, never a room.
- A question about which meeting rooms have something, or whether a room can be booked for a day, is book_room with what they asked for ("Which rooms have a projector?" is book_room with features ["projector"]).
- change_reminder and cancel_reminder: subject is copied from REMINDERS, the one the user means.
- "morning", "afternoon" or "evening" without an hour is part_of_day, not a time.
- A question about whether something is allowed or how something works ("Can I hold my meeting in the lounge?") is a question, not a booking.
- confirm_booking needs a yes ("yes", "sure", "go ahead", "sounds good"). A time or a name alone is book_room.
- A short answer ("Five.", "Six in the evening.", "To pay the rent.", "Lee.") answers what the assistant last asked for, so it is the same kind of action as the request it completes.
- "I need a reminder" with nothing else is set_reminder with no fields."""

# Worked examples, sent before every turn. They never change, so the server's prefix cache keeps
# them and they cost no prefill time after the first turn; what does cost time is the length of the
# answer, which is why each action carries only the fields that were said.
EXAMPLES = (
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\n'
     'USER SAID: "Could you time the bread for twelve minutes?"',
     '{"actions":[{"type":"start_timer","minutes":12,"label":"bread"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: the "bread" timer: 11 minutes left\n'
     'ROOM PROPOSED: none\n\nUSER SAID: "And another one, four minutes for the kettle."',
     '{"actions":[{"type":"start_timer","minutes":4,"label":"kettle"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: the "bread" timer: 11 minutes left\n'
     'ROOM PROPOSED: none\n\nUSER SAID: "Make that twenty-five."',
     '{"actions":[{"type":"change_timer","minutes":25,"label":"bread"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\n'
     'USER SAID: "Remind me at eight tonight to feed the cat."',
     '{"actions":[{"type":"set_reminder","subject":"feed the cat","time":"20:00"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\n'
     'USER SAID: "Can you put a timer on for me?"',
     '{"actions":[{"type":"start_timer"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\n'
     'ROOM PROPOSED: Elm Room on friday at 10:00\n\nUSER SAID: "Three in the afternoon works."',
     '{"actions":[{"type":"book_room","time":"15:00"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: Birch Room and Elm Room have a TV screen.\nASSISTANT LAST ASKED FOR: nothing\n'
     'RUNNING TIMERS: none\nROOM PROPOSED: none\n\nUSER SAID: "Which of those has a whiteboard too?"',
     '{"actions":[{"type":"book_room","features":["tv screen","whiteboard"]}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: Elm Room has a whiteboard, but it is only free on Tuesday.\n'
     'ASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\nUSER SAID: "Can I get it on Wednesday instead?"',
     '{"actions":[{"type":"book_room","room":"Elm Room","day":"wednesday"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: the "bread" timer: 11 minutes left\n'
     'ROOM PROPOSED: none\n\nUSER SAID: "Give the bread three more minutes."',
     '{"actions":[{"type":"add_to_timer","minutes":3,"label":"bread"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: What would you like me to remind you about?\n'
     'ASSISTANT LAST ASKED FOR: what to remind the user about\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\n'
     'USER SAID: "To pay the rent."',
     '{"actions":[{"type":"set_reminder","subject":"pay the rent"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\n'
     'USER SAID: "Book the Birch Room for Wednesday at 9, there are three of us."',
     '{"actions":[{"type":"book_room","room":"Birch Room","day":"wednesday","time":"09:00","people":3}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\n'
     'USER SAID: "Can I eat my lunch in the cafe on Monday?"',
     '{"actions":[{"type":"question"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: how many people the room is for\nRUNNING TIMERS: none\n'
     'ROOM PROPOSED: none\n\nUSER SAID: "Nine."',
     '{"actions":[{"type":"book_room","people":9}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: what time the reminder is for\nRUNNING TIMERS: none\n'
     'ROOM PROPOSED: none\n\nUSER SAID: "Seven in the morning."',
     '{"actions":[{"type":"set_reminder","time":"07:00"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\n'
     'USER SAID: "A quiet place for me and a colleague tomorrow at noon."',
     '{"actions":[{"type":"book_room","people":2,"features":["quiet"],"day":"tuesday","time":"12:00"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\n'
     'ROOM PROPOSED: Elm Room on friday at 10:00\n\nUSER SAID: "Make it seven of us instead."',
     '{"actions":[{"type":"book_room","people":7}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\n'
     'ROOM PROPOSED: Elm Room on friday at 10:00\n\nUSER SAID: "Perfect, put it under Dana."',
     '{"actions":[{"type":"confirm_booking","name":"Dana"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: the "rice" timer: 9 minutes left | '
     'the "tea" timer: 3 minutes left\nROOM PROPOSED: none\n\nUSER SAID: "Stop the rice one."',
     '{"actions":[{"type":"cancel_timer","label":"rice"}]}'),
    ('TODAY: monday\nASSISTANT LAST SAID: nothing yet\nASSISTANT LAST ASKED FOR: nothing\nRUNNING TIMERS: none\nROOM PROPOSED: none\n\n'
     'USER SAID: "Do you have a gym, and when do you open on Sunday?"',
     '{"actions":[{"type":"question"}]}'),
)

# The examples go inside the system prompt, not as earlier chat turns: gpt-oss-20b answered an
# empty message when it saw assistant turns without its reasoning channel (job 7168442).
SYSTEM_PROMPT = INSTRUCTIONS + "\n\nExamples:\n\n" + "\n\n".join(
    f"{example}\n-> {answer}" for example, answer in EXAMPLES
)

CONTEXT_TEMPLATE = """TODAY: {today}
ASSISTANT LAST SAID: {last_reply}
ASSISTANT LAST ASKED FOR: {pending}
RUNNING TIMERS: {timers}
REMINDERS: {reminders}
ROOM REQUEST SO FAR: {request}
ROOM PROPOSED: {proposed}

USER SAID: "{transcript}\""""


def schema(room_names: list[str], features: list[str]) -> dict:
    """Return the JSON schema the server constrains the answer to: per action, only its own fields.

    A single object type with every field optional let Llama 3.2 3B write a room, a booking name and
    a day into a turn that named none of them (job 7167867); here a lookup question can only be
    {"type": "question"} and a timer can never carry a room.
    """
    clock = {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"}
    minutes = {"type": "number", "exclusiveMinimum": 0}
    text = {"type": "string", "maxLength": 40}
    fields = {
        "start_timer": {"minutes": minutes, "label": text},
        "change_timer": {"minutes": minutes, "label": text},
        "add_to_timer": {"minutes": minutes, "label": text},
        "cancel_timer": {"label": text},
        "time_left": {"label": text},
        "set_reminder": {"subject": text, "time": clock, "date": text},
        "change_reminder": {"subject": text, "time": clock, "date": text},
        "cancel_reminder": {"subject": text},
        "list": {"list_what": {"type": "string", "enum": list(LIST_KINDS)}},
        "book_room": {
            "people": {"type": "integer", "minimum": 1, "maximum": 50},
            "features": {"type": "array", "items": {"type": "string", "enum": features}, "maxItems": 3},
            "day": {"type": "string", "enum": list(WEEKDAYS)},
            "time": clock,
            "part_of_day": {"type": "string", "enum": list(PARTS_OF_DAY)},
            "room": {"type": "string", "enum": room_names},
            "name": text,
        },
        "confirm_booking": {"name": text},
        "cancel_booking": {},
        "question": {},
    }
    variants = [
        {
            "type": "object",
            "properties": {"type": {"const": kind}, **properties},
            "required": ["type"],
            "additionalProperties": False,
        }
        for kind, properties in fields.items()
    ]

    return {
        "type": "object",
        "properties": {"actions": {"type": "array", "items": {"anyOf": variants}, "minItems": 1, "maxItems": 2}},
        "required": ["actions"],
        "additionalProperties": False,
    }


def context(state: DialogueState, transcript: str, today: str, last_reply: str | None = None) -> str:
    """Return what the model needs to know about the dialogue so far, as plain lines."""
    now = state.clock()
    timers = [timer for timer in state.timers if not timer.cancelled and timer.seconds > 0]

    return CONTEXT_TEMPLATE.format(
        today=today,
        last_reply=last_reply or "nothing yet",
        pending=NEEDED_IN_WORDS.get(state.needed, "nothing") if state.needed else "nothing",
        timers=" | ".join(timer.describe(now) for timer in timers) or "none",
        reminders=" | ".join(reminder.describe() for reminder in state.reminders) or "none",
        request=state.request.describe(),
        proposed=(
            f"{state.proposed_room['name']} on {state.request.day} at {state.proposed_room['hour']}"
            if state.proposed_room
            else "none"
        ),
        transcript=transcript,
    )


TIME_CUE = re.compile(
    r"\b(?:at|around|by|about|after|before|past|to)\s+(?:\d|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|half|quarter|a quarter)|o'?\s?clock|\b[ap]\.?\s?m\b|\d:\d\d|\bnoon\b|\bmidnight\b"
)


def _merge_rule_values(action: dict, kind: str, state: DialogueState, transcript: str,
                       alone: bool = True) -> None:
    """Overwrite or fill the values of ``action`` with what the rule parsers read, in place.

    ``alone`` is False when the utterance carries more than one action of this kind ("five minutes
    for the pasta and ten for the sauce"): the rules read one value for the whole sentence, so they
    must not overwrite each action's own.
    """
    text = spoken_text.normalise(transcript)
    clock = spoken_text.parse_clock_time(transcript)

    if not alone:
        if action.get("time") and not TIME_CUE.search(text):
            action.pop("time")

        return

    if kind in ("start_timer", "change_timer", "add_to_timer"):
        seconds = spoken_text.parse_duration(transcript)

        if seconds:
            action["minutes"] = seconds / 60

    if kind in ("set_reminder", "change_reminder", "book_room"):
        # The rules' time wins. A time only the model read is kept when the words say a time of
        # day ("at", "past", "o'clock", "pm", "4:30"), so "eight people" -> 08:00 and "morning" ->
        # 07:00 are dropped while a phrasing the parser does not know still gets through.
        model_time = action.get("time")

        if clock and model_time and model_time != clock.as_24_hour() and (
            int(model_time[:2]) % 12 == clock.hour % 12 and model_time[3:] == f"{clock.minute:02d}"
        ):
            # Same hour, other half of the day ("dinner at 8"): morning or evening needs the
            # sense of the sentence, which the model has and the rules guess.
            pass
        elif clock:
            action["time"] = clock.as_24_hour()
        elif action.get("time") and not TIME_CUE.search(text):
            action.pop("time")

    if kind == "book_room":
        people = spoken_text.parse_people(transcript)

        if people:
            action["people"] = people

        # The day the rules read is used only when the model gave none: "not Monday, Tuesday" and
        # "any day except Friday" are read by the rules as the day they negate.
        day = spoken_text.parse_day(transcript)

        if day and not action.get("day"):
            action["day"] = day

        features = list(action.get("features") or [])
        action["features"] = features + [f for f in spoken_text.parse_features(transcript) if f not in features]

        if not action["features"]:
            action.pop("features")

        room = state._room_named(text)

        if room is not None:
            action["room"] = room["name"]

        for part in ("morning", "afternoon", "evening"):
            if re.search(rf"\b{part}\b", text) and not action.get("part_of_day"):
                action["part_of_day"] = part

    if kind in ("book_room", "confirm_booking"):
        name = spoken_text.parse_person_name(transcript)

        if name:
            action["name"] = name
        elif action.get("name") and not all(
            re.search(rf"\b{re.escape(word)}\b", transcript) and word[:1].isupper()
            for word in action["name"].strip(" ,.").split()
        ):
            # A name is a capitalised word the user said: not "eight people", not "as well".
            action.pop("name")


def _reminder_meant(state: DialogueState, transcript: str, subject: str | None) -> str | None:
    """Return the subject of the reminder whose words the utterance shares most, else ``subject``."""
    said = {word for word in spoken_text.normalise(transcript).split() if len(word) > 3}
    best, overlap = None, 0

    for reminder in state.reminders:
        shared = len(said & {word for word in reminder.subject.lower().split() if len(word) > 3})

        if shared > overlap:
            best, overlap = reminder.subject, shared

    return best or subject


def plausible(actions: list[dict], state: DialogueState, transcript: str) -> list[dict]:
    """Return the actions with the ones that cannot apply, or that name what was not said, removed.

    Llama 3.2 3B filled every field it was shown: a room, a booking name and a day the user never
    said, and a "how much is left" on a turn about printing, with no timer running. An action on a
    timer, reminder or proposal that does not exist is dropped, and a room or a name is kept only
    when its words are in the transcript.
    """
    said = transcript.lower()
    family = {"start_timer": "timer", "change_timer": "timer", "add_to_timer": "timer",
              "set_reminder": "reminder", "change_reminder": "reminder", "book_room": "booking",
              "confirm_booking": "booking"}
    counts: dict[str, int] = {}

    for action in actions:
        name = family.get(action.get("type"), action.get("type"))
        counts[name] = counts.get(name, 0) + 1
    room_names = {room["name"].lower(): room["name"] for room in state.knowledge_base.rooms}
    active = [timer for timer in state.timers if not timer.cancelled and timer.seconds > 0]
    kept = []

    for action in actions:
        kind = action.get("type")

        if kind in ("time_left", "cancel_timer", "change_timer", "add_to_timer") and not active:
            continue

        if kind in ("change_reminder", "cancel_reminder") and not state.reminders:
            continue

        if kind in ("confirm_booking", "cancel_booking") and state.proposed_room is None:
            continue

        # A booking is confirmed only on a yes the user said. "5 p.m. under the name Asa" was read as
        # a confirmation and booked the room unasked (job 7168496); without a yes it is a detail.
        if kind == "confirm_booking" and not spoken_text.says_yes(spoken_text.normalise(transcript)):
            action = {"type": "book_room", **({"name": action["name"]} if action.get("name") else {})}
            kind = "book_room"

        action = dict(action)

        # A booking name alone, with no booking under way and no name asked for, is not a booking:
        # "Can I have my coffee at the cafe?" came back as book_room named "cafe". Judged on what
        # the model said, before the rules add the day the question happened to mention.
        if kind == "book_room" and set(action) <= {"type", "name"} and state.request.is_empty() and (
            state.proposed_room is None and state.needed != NEEDED_BOOKING_NAME
        ):
            continue

        # The model decides what kind of action this is; a value the rules can read from the words
        # comes from the rules. The model dropped values that were said ("for eight", "at 1",
        # "10 o'clock") and invented ones that were not ("8 people" -> 08:00, "course 230" ->
        # 02:30, "morning" -> 07:00) in jobs 7169187 and 7169250.
        _merge_rule_values(action, kind, state, transcript,
                           alone=counts.get(family.get(kind, kind), 0) <= 1)

        # The reminder to change is the one the utterance talks about: the model named the
        # "qualification exam" for "the deadline for the homework" (job 7169187).
        if kind in ("change_reminder", "cancel_reminder") and state.reminders:
            action["subject"] = _reminder_meant(state, transcript, action.get("subject"))

        # gpt-oss-20b filled "time": "00:00" and "name": "? (not specified)" on turns that said
        # neither (job 7168960): midnight only counts when it was said, a name only when it is a name.
        if action.get("time") == "00:00" and not any(word in said for word in ("midnight", "12", "twelve")):
            action.pop("time")

        if action.get("name") and not re.fullmatch(r"[A-Za-z][A-Za-z' .-]{0,30}", action["name"].strip(" ,.")):
            action.pop("name")

        # gpt-oss-20b wrote "Book the Oak Nook" as the booking NAME "Oak Nook," (job 7168461).
        name = (action.get("name") or "").strip(" ,.").lower()

        if name in room_names and kind == "book_room":
            action.pop("name")
            action.setdefault("room", room_names[name])

        for key in ("room", "name"):
            value = action.get(key)

            if value and not all(word in said for word in value.lower().split()[:1]):
                action.pop(key)

        kept.append(action)

    return kept


class IntentReader:
    """Asks the served model for the actions in one utterance."""

    def __init__(self, client: llm_client.ChatClient, options: dict, room_names: list[str],
                 features: list[str], max_tokens: int = 160) -> None:
        self.client = client
        # A reasoning model (gpt-oss-20b) spends tokens thinking before the JSON, so it gets room for
        # them; its reasoning effort is the one the replies use.
        reasoning = bool(options.get("reasoning_effort"))
        self.options = {**options, "max_tokens": max_tokens * 4 if reasoning else max_tokens,
                        "temperature": 0.0}
        self.schema = schema(room_names, features)

        # How long the last read took, for the trace (extras.intent_milliseconds).
        self.last_milliseconds: float | None = None

    def read(self, state: DialogueState, transcript: str, today: str | None = None,
             last_reply: str | None = None) -> list[dict]:
        """Return the actions in ``transcript``. Raises llm_client.LlmError when the server fails."""
        today = today or datetime.date.today().strftime("%A").lower()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context(state, transcript, today, last_reply)},
        ]
        started = time.perf_counter()

        try:
            answer = self.client.complete_json(messages, self.schema, **self.options)

        finally:
            self.last_milliseconds = (time.perf_counter() - started) * 1000

        actions = answer.get("actions") if isinstance(answer, dict) else None

        if not isinstance(actions, list):
            raise llm_client.LlmError(f"intent answer has no action list: {json.dumps(answer)[:200]}")

        return [action for action in actions if isinstance(action, dict) and action.get("type")]
