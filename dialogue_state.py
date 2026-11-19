"""What the agent is holding in mind during one dialogue. Owned by Partner B.

The model phrases the reply. It does not own the state, and it is never asked to do arithmetic.
Every timer, reminder and booking lives here, in Python, and the brain renders them into the prompt
as FACTS the model is told to speak and forbidden to contradict. Three things in the task families
forced that split:

* "How much is left" is a subtraction from the moment the timer was set. A model reading its own
  earlier sentences has no idea how long the dialogue has taken.
* A booking must never break a constraint. Which rooms satisfy "ten people and a projector on
  Thursday" is a search over the knowledge base, and ``knowledge_base.rooms_matching`` does it.
* A value the user did not say must be asked for, not invented, so a slot that could not be parsed
  stays empty and is reported as NEEDED rather than filled with a plausible guess.

The intent rules are deliberately narrow. A state change happens only when the utterance carries
both an explicit verb and a value it can act on; anything else leaves the state alone and becomes
an ordinary question for the model. A false negative here costs one clarifying question, which the
task families explicitly allow. A false positive silently books the wrong room.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import spoken_text
from knowledge_base import KnowledgeBase

# What the agent is waiting for the user to supply, when a request arrived without it. The brain
# puts it in the prompt as NEEDED, and the next turn tries to read a bare value as that slot.
NEEDED_TIMER_DURATION = "timer_duration"
NEEDED_REMINDER_TIME = "reminder_time"
NEEDED_REMINDER_SUBJECT = "reminder_subject"
NEEDED_BOOKING_PEOPLE = "booking_people"
NEEDED_BOOKING_DAY = "booking_day"
NEEDED_BOOKING_NAME = "booking_name"

NOTHING_TO_CANCEL = "There was nothing to cancel."

NEEDED_IN_WORDS = {
    NEEDED_TIMER_DURATION: "how long the timer should run",
    NEEDED_REMINDER_TIME: "what time the reminder is for",
    NEEDED_REMINDER_SUBJECT: "what to remind the user about",
    NEEDED_BOOKING_PEOPLE: "how many people the room is for",
    NEEDED_BOOKING_DAY: "which day and time the room is for",
    NEEDED_BOOKING_NAME: "the name to book under",
}

TIMER_WORDS = re.compile(r"\b(timer|countdown|alarm)\b")
TIMER_VERBS = re.compile(r"\b(set|start|started|starting|put|make|begin|give)\b")
REMINDER_WORDS = re.compile(r"\b(remind|reminder|reminders)\b")
CANCEL_WORDS = re.compile(r"\b(cancel|stop|delete|remove|forget|scratch|never mind|nevermind)\b")
CHANGE_WORDS = re.compile(r"\b(make it|change|extend|add|update|instead|actually)\b")
LIST_WORDS = re.compile(
    r"\b(what|which|list|tell me|read)\b.{0,30}\b(timers?|reminders?|bookings?)\b"
)
READ_BACK_WORDS = re.compile(r"\bread\b.{0,25}\bback\b")
REMAINING_WORDS = re.compile(
    r"\b(how (much|long|many)|time left|left on|remaining|still (got|have))\b"
)
# "meeting" and "space" are deliberately absent. "Can I have my group meeting in the quiet
# lounge?" is a question about a policy, not a booking, and including them started a room
# negotiation that the user never asked for.
BOOKING_WORDS = re.compile(r"\b(room|rooms|book|booking|reserve|reservation)\b")

# A question that merely MENTIONS booking a room ("how far ahead can I book a room") is a lookup
# against the room-booking policy. Only an utterance that also carries a constraint — a day, a head
# count, a feature — is a booking in progress.
QUESTION_OPENER = re.compile(r"^(how|what|which|when|where|why|is there|are there|do you|does)\b")
PART_OF_DAY_HOURS = {
    "morning": (7, 12),
    "afternoon": (12, 18),
    "evening": (17, 22),
    "night": (18, 24),
}

# The shapes a time or a date is said in. They are cut out of a reminder before what is left is
# read as its subject; ``spoken_text`` reads the same phrases as values.
WHEN_PHRASES = (
    re.compile(r"\b(?:on|at|by)?\s*(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\s*(?:of\s+)?"
               r"(?:january|february|march|april|may|june|july|august|september|october|november|"
               r"december)\b"),
    re.compile(r"\b(?:on|at|by)\s+(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|"
               r"eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
               r"seventeenth|eighteenth|nineteenth|twentieth|thirtieth|\d{1,2}(?:st|nd|rd|th))\b"),
    re.compile(r"\b(?:at\s+)?\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?"),
    re.compile(r"\b(?:at|around|by)\s+\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?|o'?\s?clock)?\b"),
    re.compile(r"\b(?:at|around)?\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
               r"twelve)\s+(?:o'?\s?clock|in the (?:morning|afternoon|evening))\b"),
    re.compile(r"\b(?:today|tomorrow|tonight|this (?:morning|afternoon|evening))\b"),
    re.compile(r"\bon (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
)

# "for the pasta", "for my laundry": what a timer is for. Durations are excluded by the digit test,
# so "for 5 minutes" never becomes a label.
LABEL_PATTERN = re.compile(r"\bfor (?:the |my |our |a |an )?([a-z]+(?: [a-z]+)?)")
LABEL_STOP_WORDS = frozenset(
    "me us you minutes minute hours hour seconds second now then today tomorrow people person "
    "that this it one two three four five six seven eight nine ten".split()
)


def _seconds_in_words(seconds: float) -> str:
    """Return a length as the words the agent would say, rounded the way a person rounds."""
    seconds = max(0, int(round(seconds)))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minutes" if minutes else (
            f"{hours} hour{'s' if hours != 1 else ''}"
        )

    if minutes:
        return f"{minutes} minutes {remainder} seconds" if remainder else f"{minutes} minutes"

    return f"{remainder} seconds"


@dataclass
class Timer:
    """One running timer, timed from the monotonic clock rather than from the conversation."""

    label: str | None
    seconds: int
    started_at: float
    cancelled: bool = False

    def remaining_seconds(self, now: float) -> float:
        """Return how long is left, never below zero."""
        return max(0.0, self.seconds - (now - self.started_at))

    def describe(self, now: float) -> str:
        """Return one line for the prompt: what it is for, how long it was set for, what is left."""
        # The amount left comes first, because "how much is left" is what is asked of it. A timer
        # with no label is "the timer": "unnamed" was read aloud as if it were the label.
        name = f'the "{self.label}" timer' if self.label else "the timer"
        left = self.remaining_seconds(now)
        state = "finished" if left <= 0 else f"{_seconds_in_words(left)} left"

        return f"{name}: {state} (set for {_seconds_in_words(self.seconds)} in total)"


@dataclass
class Reminder:
    """One reminder: the thing to be reminded of, and when."""

    subject: str
    when: str | None = None
    # A date said without a time of day ("today", "on the 19th"), kept until the time is given.
    day: str | None = None

    def describe(self) -> str:
        """Return one line for the prompt."""
        return f'"{self.subject}" at {self.when}' if self.when else f'"{self.subject}" (no time set)'


@dataclass
class BookingRequest:
    """The constraints of the booking being negotiated, as far as the user has given them."""

    people: int | None = None
    features: list[str] = field(default_factory=list)
    day: str | None = None
    hour: str | None = None
    part_of_day: str | None = None

    def is_empty(self) -> bool:
        """Return whether the user has given nothing to search on yet."""
        return not (self.people or self.features or self.day or self.hour)

    def describe(self) -> str:
        """Return the constraints in the order a person would say them."""
        parts = []

        if self.people:
            parts.append(f"{self.people} people")

        if self.features:
            parts.append(" and ".join(self.features))

        if self.day:
            parts.append(self.day + (f" at {self.hour}" if self.hour else ""))
        elif self.hour:
            parts.append(f"at {self.hour}")

        if self.part_of_day and not self.hour:
            parts.append(self.part_of_day)

        return ", ".join(parts) if parts else "nothing yet"


@dataclass
class Booking:
    """A booking the user said yes to."""

    room_name: str
    floor: int
    day: str
    hour: str
    name: str

    def describe(self) -> str:
        """Return one line for the prompt."""
        return f"{self.room_name} (floor {self.floor}) on {self.day} at {self.hour} for {self.name}"


class DialogueState:
    """Everything the agent remembers inside one dialogue, and the reading of the latest turn."""

    def __init__(self, knowledge_base: KnowledgeBase, clock=time.monotonic) -> None:
        self.knowledge_base = knowledge_base
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        """Forget the whole dialogue. Called by the brain at the first turn of every dialogue."""
        self.timers: list[Timer] = []
        self.reminders: list[Reminder] = []
        self.bookings: list[Booking] = []
        self.request = BookingRequest()
        self.proposed_room: dict | None = None
        self.chosen_room_id: str | None = None
        self.booking_name: str | None = None
        self.needed: str | None = None
        self.notes: list[str] = []

    # -------------------------------------------------------------------------------------------
    # Reading one utterance
    # -------------------------------------------------------------------------------------------

    def observe(self, transcript: str) -> None:
        """Apply whatever the utterance clearly asks for, and record what was done.

        ``notes`` is what changed, in the words the agent should say back; ``needed`` is the one
        value the agent must ask for. Both are read by the brain when it builds the prompt.
        """
        self.notes = []
        text = spoken_text.normalise(transcript)
        pending = self.needed
        self.needed = None

        if not text:
            return

        if self._handle_cancel(text):
            return

        # "List my timers" / "What reminders do I have?" / "Read my reminders back to me". The
        # items are spelled out in the note, because with only the TIMERS line in the prompt the
        # model answered "I don't have a list of your timers".
        if LIST_WORDS.search(text) or READ_BACK_WORDS.search(text):
            self._list_items(text)

            return

        # The agent asked what time the reminder is for, and this turn answers it. The answer does
        # not have to repeat the word "remind" — "Six in the evening" is a whole turn — so this is
        # checked before the keyword rules below.
        if pending == NEEDED_REMINDER_TIME and self.reminders:
            when = self._when_from(transcript)

            if when:
                self.reminders[-1].when = when
                self.notes.append(
                    f'Set the time of the reminder "{self.reminders[-1].subject}" to {when}.'
                )

                return

        # "Actually the deadline for the homework is on the 18th as well, can you update that?"
        # is a change to a reminder that never says the word "reminder". It is only acted on when
        # all three hold: something to change, a word that means change, and a new time.
        if self.reminders and CHANGE_WORDS.search(text) and not REMINDER_WORDS.search(text):
            when = self._when_from(transcript)

            if when:
                target = self._matching_reminder(text) or self.reminders[-1]
                target.when = when
                self.notes.append(f'Updated the reminder "{target.subject}" to {when}.')

                return

        if REMINDER_WORDS.search(text):
            self._handle_reminder(transcript, text, pending)
            return

        if self._looks_like_timer(text, pending):
            self._handle_timer(transcript, text, pending)
            return

        if self._looks_like_booking(text, pending):
            self._handle_booking(transcript, text, pending)
            return

        # Nothing actionable: a lookup question, or a turn the rules are deliberately not sure
        # about. The state is untouched and the model answers from the facts it is given.
        if pending:
            self.needed = pending

    # -- timers ---------------------------------------------------------------------------------

    def _looks_like_timer(self, text: str, pending: str | None) -> bool:
        """Return whether this utterance is about a timer."""
        if TIMER_WORDS.search(text):
            return True

        if pending == NEEDED_TIMER_DURATION:
            return True

        if self.timers and (CHANGE_WORDS.search(text) or REMAINING_WORDS.search(text)):
            return True

        return bool(TIMER_VERBS.search(text) and spoken_text.parse_duration(text))

    def _handle_timer(self, transcript: str, text: str, pending: str | None) -> None:
        """Set, extend or report a timer."""
        duration = spoken_text.parse_duration(transcript)
        active = [timer for timer in self.timers if not timer.cancelled]

        # "Actually make it fifteen": a bare number only means something against a running timer,
        # and the unit comes from that timer rather than from a guess.
        if duration is None and active and CHANGE_WORDS.search(text):
            bare = spoken_text.parse_bare_number(transcript)

            if bare is not None:
                duration = bare * 60

        if duration is None and pending == NEEDED_TIMER_DURATION and active:
            # The length arrives on its own turn, answering the question the agent just asked.
            bare = spoken_text.parse_bare_number(transcript)
            duration = bare * 60 if bare is not None else None

        if REMAINING_WORDS.search(text) and duration is None:
            # A question about a timer, not a change to one. The TIMERS block already answers it;
            # the note says so, because a misheard label ("pesto" for "pasta") otherwise made the
            # model answer that it had no such timer.
            if not active:
                self.notes.append("There is no timer running.")
            else:
                self.notes.append(
                    "The user asked how much time is left. Time left: "
                    + " | ".join(timer.describe(self.clock()) for timer in active)
                    + "."
                )

            return

        if duration is None:
            if TIMER_WORDS.search(text) and not active:
                # "Put a timer for me" with no length: ask, do not invent one.
                self.needed = NEEDED_TIMER_DURATION
                self.timers.append(Timer(label=self._label_from(text), seconds=0, started_at=self.clock()))
                self.timers[-1].cancelled = True  # A placeholder must never count as running.

            return

        now = self.clock()
        adding = bool(re.search(r"\badd\b|\bmore\b|\bextra\b|\banother\b", text)) and active
        changing = bool(CHANGE_WORDS.search(text)) and active and not adding

        if adding:
            timer = active[-1]
            timer.seconds += duration
            self.notes.append(
                f"Added {_seconds_in_words(duration)} to the timer; it now has "
                f"{_seconds_in_words(timer.remaining_seconds(now))} left."
            )

            return

        if changing and not TIMER_WORDS.search(text):
            timer = active[-1]
            timer.seconds = duration
            timer.started_at = now
            self.notes.append(
                f"Changed the timer to {_seconds_in_words(duration)}"
                + (f" for the {timer.label}." if timer.label else ".")
            )

            return

        if pending == NEEDED_TIMER_DURATION and self.timers and self.timers[-1].cancelled and (
            self.timers[-1].seconds == 0
        ):
            # Fill the placeholder the previous turn created, so the label survives the question.
            timer = self.timers[-1]
            timer.seconds = duration
            timer.started_at = now
            timer.cancelled = False
        else:
            timer = Timer(label=self._label_from(text), seconds=duration, started_at=now)
            self.timers.append(timer)

        self.notes.append(
            f"Started a timer for {_seconds_in_words(duration)}"
            + (f" for the {timer.label}." if timer.label else ".")
        )

    def _label_from(self, text: str) -> str | None:
        """Return what the timer is for, when the user said so."""
        for candidate in LABEL_PATTERN.findall(text):
            words = candidate.split()

            if any(word.isdigit() or word in LABEL_STOP_WORDS for word in words):
                continue

            return candidate.strip()

        return None

    def _list_items(self, text: str) -> None:
        """Spell out the timers, reminders or bookings the user asked to hear."""
        now = self.clock()
        wanted = [kind for kind in ("timer", "reminder", "booking") if kind in text]

        if not wanted:
            wanted = ["timer", "reminder", "booking"]

        parts = []

        if "timer" in wanted:
            active = [timer for timer in self.timers if not timer.cancelled and timer.seconds > 0]
            parts.append(
                f"{len(active)} timer(s): " + "; ".join(timer.describe(now) for timer in active)
                if active
                else "no timers running"
            )

        if "reminder" in wanted:
            parts.append(
                f"{len(self.reminders)} reminder(s): "
                + "; ".join(reminder.describe() for reminder in self.reminders)
                if self.reminders
                else "no reminders"
            )

        if "booking" in wanted:
            parts.append(
                f"{len(self.bookings)} booking(s): "
                + "; ".join(booking.describe() for booking in self.bookings)
                if self.bookings
                else "no bookings"
            )

        self.notes.append("The user asked for a list. Read it out: " + ". ".join(parts) + ".")

    # -- reminders ------------------------------------------------------------------------------

    def _handle_reminder(self, transcript: str, text: str, pending: str | None) -> None:
        """Set or report a reminder."""
        if LIST_WORDS.search(text) or READ_BACK_WORDS.search(text):
            if not self.reminders:
                self.notes.append("There are no reminders.")

            return

        when = self._when_from(transcript)
        subject = self._reminder_subject(text)

        if pending == NEEDED_REMINDER_TIME and self.reminders and when:
            # The time arrives on its own turn: "At what time?" / "Six in the evening."
            self.reminders[-1].when = when
            self.notes.append(
                f'Set the time of the reminder "{self.reminders[-1].subject}" to {when}.'
            )

            return

        if not subject:
            self.needed = NEEDED_REMINDER_SUBJECT

            return

        # "Actually the deadline is on the 18th as well, can you update that?" changes a reminder
        # the dialogue already has rather than adding a second one for the same thing.
        existing = self._matching_reminder(subject)

        if existing is not None and CHANGE_WORDS.search(text):
            existing.when = when or existing.when
            self.notes.append(f'Updated the reminder "{existing.subject}" to {existing.when}.')

            return

        reminder = Reminder(subject=subject, when=when)
        self.reminders.append(reminder)

        if when:
            self.notes.append(f'Set a reminder: "{subject}" at {when}.')
        else:
            self.notes.append(f'Noted a reminder about "{subject}", with no time yet.')
            self.needed = NEEDED_REMINDER_TIME

    def _reminder_subject(self, text: str) -> str | None:
        """Return what the user wants to be reminded about, with the when-phrase cut out.

        The when-phrase does not reliably come last. "Remind me on the 18th of November that I have
        a qualification exam" puts the date in the middle, and an earlier version of this method —
        which only looked for "remind me to|that" and then split on the first time word — returned
        nothing at all for it and made the agent ask what to remind the user about, which they had
        just said. So the date and time phrases are removed wherever they are, and what is left is
        the subject.
        """
        match = re.search(r"\bremind(?:er)?\b\s*(?:me\b)?\s*(.*)$", text)

        if not match:
            return None

        rest = match.group(1)

        for pattern in WHEN_PHRASES:
            rest = pattern.sub(" ", rest)

        # Whatever connector is left at the front ("that", "to", "on", "about") joined the subject
        # to the part that has just been removed. Stripped first, because removing a phrase from
        # the middle leaves the space it sat in and ``^`` would then match nothing.
        rest = rest.strip()

        for _ in range(3):
            stripped = re.sub(r"^(?:and|also|another|an|a|that|to|about|for|on|at|in|me)\b\s*", "", rest).strip()

            if stripped == rest.strip():
                break

            rest = stripped

        rest = re.sub(r"\s+", " ", rest).strip(" ,.")

        return rest or None

    def _when_from(self, transcript: str) -> str | None:
        """Return the date and time the utterance names, as one phrase the agent can say back."""
        clock_time = spoken_text.parse_clock_time(transcript)
        date = spoken_text.parse_date(transcript)

        if clock_time and date:
            return f"{date} at {clock_time.spoken()}"

        if clock_time:
            return clock_time.spoken()

        return date

    def _matching_reminder(self, subject: str) -> Reminder | None:
        """Return a reminder the new one is plainly about, matched on shared content words."""
        wanted = {word for word in subject.split() if len(word) > 3}

        for reminder in self.reminders:
            existing = {word for word in reminder.subject.split() if len(word) > 3}

            if wanted and existing and len(wanted & existing) >= max(1, len(wanted) // 2):
                return reminder

        return None

    # -- cancelling -----------------------------------------------------------------------------

    def _handle_cancel(self, text: str) -> bool:
        """Cancel the most recent thing when the user asks, and say which one it was."""
        if not CANCEL_WORDS.search(text):
            return False

        if re.search(r"\breminder", text) or (self.reminders and not self.timers):
            if self.reminders:
                dropped = self.reminders.pop()
                self.notes.append(f'Cancelled the reminder "{dropped.subject}".')

                return True

        active = [timer for timer in self.timers if not timer.cancelled]

        if active and (TIMER_WORDS.search(text) or re.search(r"\b(that|it|this)\b", text)):
            active[-1].cancelled = True
            label = f' for the {active[-1].label}' if active[-1].label else ""
            self.notes.append(f"Cancelled the timer{label}.")

            return True

        if self.proposed_room is not None:
            self.proposed_room = None
            self.request = BookingRequest()
            self.notes.append("Dropped the room that was proposed.")

            return True

        if self.reminders:
            dropped = self.reminders.pop()
            self.notes.append(f'Cancelled the reminder "{dropped.subject}".')

            return True

        self.notes.append(NOTHING_TO_CANCEL)

        return True

    # -- bookings -------------------------------------------------------------------------------

    def _looks_like_booking(self, text: str, pending: str | None) -> bool:
        """Return whether this utterance is part of booking a room."""
        has_constraint = bool(
            spoken_text.parse_people(text)
            or spoken_text.parse_features(text)
            or spoken_text.parse_day(text)
            or spoken_text.parse_clock_time(text)
        )

        if BOOKING_WORDS.search(text):
            # "How far ahead can I book a room, and for how long?" is the room-booking POLICY, and
            # answering it with "which day would you like?" is a failed turn.
            if QUESTION_OPENER.search(text) and not has_constraint and self.request.is_empty():
                return False

            return True

        if pending in (NEEDED_BOOKING_PEOPLE, NEEDED_BOOKING_DAY, NEEDED_BOOKING_NAME):
            return True

        # Mid-negotiation, a bare constraint ("Five.", "A projector.", "Yes, under the name Sam",
        # "Five pm, under the name Asal") belongs to the booking in hand. The time and the name are
        # read from the transcript, because parse_person_name needs its original capitalisation.
        if not self.request.is_empty() or self.proposed_room is not None:
            return bool(
                spoken_text.parse_people(text)
                or spoken_text.parse_features(text)
                or spoken_text.parse_day(text)
                or spoken_text.parse_clock_time(text)
                or spoken_text.parse_person_name(text)
                or spoken_text.says_yes(text)
                or spoken_text.parse_bare_number(text) is not None
            )

        return False

    def _handle_booking(self, transcript: str, text: str, pending: str | None) -> None:
        """Collect constraints, search the knowledge base, propose a room or confirm one."""
        self._collect_constraints(transcript, text, pending)

        name = spoken_text.parse_person_name(transcript)

        # "Lee." on its own, right after the agent asked for the name: the whole turn is the name.
        if name is None and pending == NEEDED_BOOKING_NAME:
            bare = transcript.strip().strip(".!,")

            if re.fullmatch(r"[A-Z][a-zA-Z'-]+(?: [A-Z][a-zA-Z'-]+)?", bare):
                name = bare

        if name:
            if name != self.booking_name:
                self.notes.append(f"The booking will be under the name {name}.")

            self.booking_name = name

        # The name was the only thing missing after a yes, so giving it completes that yes.
        yes = spoken_text.says_yes(text) or bool(
            pending == NEEDED_BOOKING_NAME and name and self.proposed_room is not None
        )
        self._advance_booking(
            yes=yes,
            yes_alone=yes and not (spoken_text.parse_people(text) or spoken_text.parse_day(text)),
        )

    def _advance_booking(self, yes: bool, yes_alone: bool) -> None:
        """Take the booking one step on from its constraints: ask, refuse, propose or confirm.

        ``yes`` is whether the user said yes this turn; ``yes_alone`` is a yes that brought no new
        constraint with it, which confirms the room that was proposed last turn.
        """
        # Confirming the room that was proposed last turn.
        if self.proposed_room is not None and yes_alone:
            if not self.booking_name:
                self.needed = NEEDED_BOOKING_NAME
                self.notes.append("The room is ready to book; only the name is missing.")

                return

            self._confirm_booking()

            return

        if self.request.is_empty():
            self.needed = NEEDED_BOOKING_DAY

            return

        # Refuse an impossible pair BEFORE asking for the day. "A room for fifteen people with a
        # whiteboard" has no answer on any day, and asking which day the user wants only to refuse
        # them on the next turn is the failure the task families call out by name.
        if self.chosen_room_id is None and (self.request.people or self.request.features):
            possible = self.knowledge_base.rooms_matching(
                capacity=self.request.people, features=tuple(self.request.features)
            )

            if not possible:
                self.proposed_room = None
                self.notes.append(
                    "NOTHING MATCHES. "
                    + self.knowledge_base.describe_conflict(
                        self.request.people, tuple(self.request.features), None, None
                    )
                )

                return

        if self.request.day is None:
            self.needed = NEEDED_BOOKING_DAY

            # "Which rooms do you suggest?" before a day is given. Without the candidates in the
            # notes the model answered that no room had a whiteboard at all.
            if self.request.people or self.request.features:
                fitting = self.knowledge_base.rooms_matching(
                    capacity=self.request.people, features=tuple(self.request.features)
                )

                if fitting:
                    self.notes.append(
                        "Rooms that fit so far (availability depends on the day): "
                        + ", ".join(f"{room['name']} (floor {room['floor']})" for room in fitting)
                        + "."
                    )

            return

        if self.request.people is None and not self.request.features:
            self.needed = NEEDED_BOOKING_PEOPLE

            return

        self._search_rooms()

        # A name given together with a confirmation closes the booking in one turn.
        if self.proposed_room is not None and self.booking_name and yes:
            self._confirm_booking()

    def _room_named(self, text: str) -> dict | None:
        """Return the room the user named, when they named one.

        "Maple Hall is fine" is the user accepting an alternative that breaks a constraint they
        gave earlier, and it has to beat the constraint: they have just overruled it themselves.
        """
        for room in self.knowledge_base.rooms:
            name = room["name"].lower()

            if name in text or name.split()[0] in text.split():
                return room

        return None

    def _collect_constraints(self, transcript: str, text: str, pending: str | None) -> None:
        """Fold whatever constraints this utterance carries into the request being negotiated."""
        named = self._room_named(text)

        if named is not None and named["id"] != self.chosen_room_id:
            self.chosen_room_id = named["id"]
            self.proposed_room = None
            self.notes.append(f"The user asked for {named['name']} by name.")

        people = spoken_text.parse_people(transcript)

        if people is None and pending == NEEDED_BOOKING_PEOPLE:
            people = spoken_text.parse_bare_number(transcript)

        if people is not None:
            if self.request.people is not None and people != self.request.people:
                # A changed constraint invalidates the room that was proposed under the old one.
                self.proposed_room = None

            self.request.people = people
            self._release_named_room_if_too_small()

        for feature in spoken_text.parse_features(transcript):
            if feature not in self.request.features:
                self.request.features.append(feature)
                self.proposed_room = None

        day = spoken_text.parse_day(transcript)

        if day and day != self.request.day:
            self.request.day = day
            self.proposed_room = None

        clock_time = spoken_text.parse_clock_time(transcript)

        if clock_time:
            hour = f"{clock_time.hour:02d}:00"

            if hour != self.request.hour:
                self.request.hour = hour
                self.proposed_room = None

        for part in PART_OF_DAY_HOURS:
            if re.search(rf"\b{part}\b", text):
                self.request.part_of_day = part

    def _release_named_room_if_too_small(self) -> None:
        """Stop holding on to a room the user named once it cannot hold the people now coming.

        "Book the Oak Nook ... it's just me" then "Actually there will be four of us": Oak Nook holds
        2, so searching only Oak Nook could never succeed. The name was a choice of room for one
        person; four people need the normal search, which may find another room at the same time.
        """
        if self.chosen_room_id is None or not self.request.people:
            return

        room = next(room for room in self.knowledge_base.rooms if room["id"] == self.chosen_room_id)

        if room["capacity"] < self.request.people:
            self.chosen_room_id = None
            self.notes.append(f"{room['name']} holds only {room['capacity']}, so other rooms were searched.")

    def _hours_allowed(self) -> tuple[int, int]:
        """Return the hour window the user asked for: the part of the day, or the whole day."""
        if self.request.part_of_day:
            return PART_OF_DAY_HOURS[self.request.part_of_day]

        return (0, 24)

    def _search_rooms(self) -> None:
        """Find the rooms that satisfy every constraint, and propose the smallest that does."""
        request = self.request

        if self.chosen_room_id is not None:
            # The user named the room, so the search is over which hour of their day it is free,
            # not over which room fits: they have already overruled the constraints it breaks.
            matches = [
                room
                for room in self.knowledge_base.rooms
                if room["id"] == self.chosen_room_id
                and (request.hour is None or request.hour in room["free_hour_starts"].get(request.day, []))
                and room["free_hour_starts"].get(request.day)
            ]
        else:
            matches = self.knowledge_base.rooms_matching(
                capacity=request.people,
                features=tuple(request.features),
                day=request.day,
                hour=request.hour,
            )

        earliest, latest = self._hours_allowed()
        usable: list[tuple[dict, str]] = []

        for room in matches:
            free_hours = room["free_hour_starts"].get(request.day, [])

            if request.hour:
                if request.hour in free_hours:
                    usable.append((room, request.hour))

                continue

            for hour in sorted(free_hours):
                if earliest <= int(hour[:2]) < latest:
                    usable.append((room, hour))
                    break

        if not usable:
            self.proposed_room = None

            if self.chosen_room_id is not None:
                room = next(
                    room for room in self.knowledge_base.rooms if room["id"] == self.chosen_room_id
                )
                free = room["free_hour_starts"].get(request.day)
                self.notes.append(
                    f"NOTHING MATCHES. {room['name']} is free on {request.day} at "
                    + ", ".join(free)
                    + "."
                    if free
                    else f"NOTHING MATCHES. {room['name']} is not free on {request.day} at all; "
                    f"it is free on {', '.join(sorted(room['free_hour_starts']))}."
                )

                return

            self.notes.append(
                "NOTHING MATCHES. "
                + self.knowledge_base.describe_conflict(
                    request.people, tuple(request.features), request.day, request.hour
                )
            )

            return

        room, hour = usable[0]
        self.proposed_room = {**room, "hour": hour}
        self.notes.append(
            f"{room['name']} on floor {room['floor']} is free on {request.day} at {hour}. "
            f"It holds {room['capacity']} and has {', '.join(room['features'])}. "
            "Ask the user to confirm before booking."
        )

    def _confirm_booking(self) -> None:
        """Turn the proposed room into a booking and say the confirmation line back."""
        room = self.proposed_room
        booking = Booking(
            room_name=room["name"],
            floor=room["floor"],
            day=self.request.day,
            hour=room["hour"],
            name=self.booking_name,
        )
        self.bookings.append(booking)
        self.notes.append(f"BOOKED: {booking.describe()}.")
        self.proposed_room = None
        self.chosen_room_id = None
        self.request = BookingRequest()

    # -------------------------------------------------------------------------------------------
    # Carrying out actions the LLM read from the utterance (intent_reader, --nlu llm)
    # -------------------------------------------------------------------------------------------

    def apply_actions(self, actions: list[dict]) -> None:
        """Carry out the actions ``intent_reader`` read from one utterance, instead of ``observe``.

        The reading came from the model; everything done with it is the same Python as the rules
        path: the timers and their arithmetic, the reminders, the room search and the confirmation.
        Only values the action carries are used, and a missing one is asked for, never invented.
        """
        self.notes = []
        pending = self.needed
        self.needed = None
        booking_actions = []

        for action in actions:
            kind = action.get("type")

            if kind in ("book_room", "confirm_booking"):
                booking_actions.append(action)
            elif kind == "cancel_booking":
                self.proposed_room = None
                self.request = BookingRequest()
                self.notes.append("Dropped the room that was proposed.")
            elif kind in ("start_timer", "change_timer", "add_to_timer", "cancel_timer", "time_left"):
                self._timer_action(kind, action, pending)
            elif kind in ("set_reminder", "change_reminder", "cancel_reminder"):
                self._reminder_action(kind, action, pending)
            elif kind == "list":
                what = action.get("list_what") or "all"
                self._list_items("timer reminder booking" if what == "all" else what)

        if booking_actions:
            yes = False

            for action in booking_actions:
                self._booking_values(action)
                yes = yes or action.get("type") == "confirm_booking" or bool(
                    pending == NEEDED_BOOKING_NAME and action.get("name") and self.proposed_room is not None
                )

            self._advance_booking(yes=yes, yes_alone=yes)

        # Nothing the state acts on (a lookup question): keep waiting for what was asked for.
        if not self.notes and self.needed is None and pending:
            self.needed = pending

    def _timer_named(self, label: str | None) -> Timer | None:
        """Return the running timer the label points at, else the most recent running one."""
        active = [timer for timer in self.timers if not timer.cancelled and timer.seconds > 0]

        if label:
            wanted = set(label.lower().split())

            for timer in reversed(active):
                if timer.label and wanted & set(timer.label.lower().split()):
                    return timer

        return active[-1] if active else None

    def _timer_action(self, kind: str, action: dict, pending: str | None) -> None:
        """Start, change, extend, cancel or report a timer from one action."""
        minutes = action.get("minutes")
        seconds = int(round(float(minutes) * 60)) if isinstance(minutes, (int, float)) and minutes > 0 else None
        label = (action.get("label") or "").strip().lower() or None
        now = self.clock()
        target = self._timer_named(label)

        if kind == "time_left":
            if target is None:
                self.notes.append("There is no timer running.")
            else:
                self.notes.append(f"The user asked how much time is left. Time left: {target.describe(now)}.")

            return

        if kind == "cancel_timer":
            if target is None:
                self.notes.append("There was no timer to cancel.")
            else:
                target.cancelled = True
                self.notes.append("Cancelled " + (f"the timer for the {target.label}." if target.label else "the timer."))

            return

        if seconds is None and kind != "start_timer":
            # A change or an extension with no length read: nothing to apply, and no new timer.
            return

        if seconds is None:
            # "Put a timer for me": ask for the length, keep the label for when it comes.
            self.needed = NEEDED_TIMER_DURATION
            placeholder = Timer(label=label, seconds=0, started_at=now, cancelled=True)
            self.timers.append(placeholder)

            return

        if kind == "add_to_timer" and target is not None:
            target.seconds += seconds
            self.notes.append(
                f"Added {_seconds_in_words(seconds)} to the timer"
                + (f" for the {target.label}" if target.label else "")
                + f". It now has {_seconds_in_words(target.remaining_seconds(now))} left."
            )

            return

        if kind == "change_timer" and target is not None:
            target.seconds = seconds
            target.started_at = now
            self.notes.append(
                f"Changed the timer to {_seconds_in_words(seconds)}"
                + (f" for the {target.label}." if target.label else ".")
            )

            return

        # A new timer, or the length for the one the agent just asked about.
        placeholder = self.timers[-1] if self.timers else None

        if pending == NEEDED_TIMER_DURATION and placeholder and placeholder.cancelled and placeholder.seconds == 0:
            placeholder.seconds, placeholder.started_at, placeholder.cancelled = seconds, now, False
            placeholder.label = placeholder.label or label
            timer = placeholder
        else:
            timer = Timer(label=label, seconds=seconds, started_at=now)
            self.timers.append(timer)

        self.notes.append(
            f"Started a timer for {_seconds_in_words(seconds)}"
            + (f" for the {timer.label}." if timer.label else ".")
        )

    @staticmethod
    def _when_of(action: dict) -> str | None:
        """Return the date and time of an action as one phrase the agent can say back."""
        clock = action.get("time")
        date = (action.get("date") or "").strip() or None
        spoken = None

        if isinstance(clock, str) and re.fullmatch(r"\d{2}:\d{2}", clock):
            spoken = spoken_text.ClockTime(int(clock[:2]), int(clock[3:]), clock).spoken()

        if spoken and date:
            return f"{date} at {spoken}"

        return spoken or date

    def _reminder_action(self, kind: str, action: dict, pending: str | None) -> None:
        """Set, change or cancel a reminder from one action."""
        subject = (action.get("subject") or "").strip().rstrip(".") or None
        when = self._when_of(action)
        target = (self._matching_reminder(subject.lower()) if subject else None) or (
            self.reminders[-1] if self.reminders else None
        )

        if kind == "cancel_reminder":
            if target is None:
                self.notes.append("There was no reminder to cancel.")
            else:
                self.reminders.remove(target)
                self.notes.append(f'Cancelled the reminder "{target.subject}".')

            return

        has_time = bool(action.get("time"))

        # The time for the reminder the agent just asked about, or a change to an existing one.
        # A new reminder about something else ("and another reminder that ...") is not the time
        # of the one waiting for its time, even while that one is pending (job 7169250).
        about_target = not subject or (target is not None and self._matching_reminder(subject.lower()) is target)

        if target is not None and when and about_target and (
            kind == "change_reminder" or (pending == NEEDED_REMINDER_TIME and target.when is None)
        ):
            if target.day and has_time and not action.get("date"):
                when = f"{target.day} at {when}"

            target.when = when
            self.notes.append(f'Set the time of the reminder "{target.subject}" to {when}.')

            return

        if not subject:
            self.needed = NEEDED_REMINDER_SUBJECT

            return

        # A reminder is for a time of day, so a date alone ("today") still needs one, as the rules
        # path asks too.
        reminder = Reminder(subject=subject, when=when if has_time else None,
                            day=None if has_time else (action.get("date") or None))
        self.reminders.append(reminder)

        if reminder.when:
            self.notes.append(f'Set a reminder: "{subject}" at {reminder.when}.')
        else:
            self.notes.append(
                f'Noted a reminder about "{subject}"' + (f" for {reminder.day}" if reminder.day else "")
                + ", with no time yet."
            )
            self.needed = NEEDED_REMINDER_TIME

    def _booking_values(self, action: dict) -> None:
        """Fold the booking values of one action into the request being negotiated."""
        room_name = action.get("room")

        if room_name:
            named = next((room for room in self.knowledge_base.rooms if room["name"] == room_name), None)

            if named is not None and named["id"] != self.chosen_room_id:
                self.chosen_room_id = named["id"]
                self.proposed_room = None
                self.notes.append(f"The user asked for {named['name']} by name.")

        people = action.get("people")

        if isinstance(people, int) and people > 0 and people != self.request.people:
            self.request.people = people
            self.proposed_room = None
            self._release_named_room_if_too_small()

        for feature in action.get("features") or []:
            if feature not in self.request.features:
                self.request.features.append(feature)
                self.proposed_room = None

        day = action.get("day")

        if day and day != self.request.day:
            self.request.day = day
            self.proposed_room = None

        clock = action.get("time")

        if isinstance(clock, str) and re.fullmatch(r"\d{2}:\d{2}", clock):
            hour = f"{clock[:2]}:00"

            if hour != self.request.hour:
                self.request.hour = hour
                self.proposed_room = None

        if action.get("part_of_day") in PART_OF_DAY_HOURS:
            self.request.part_of_day = action["part_of_day"]

        name = (action.get("name") or "").strip()

        if name:
            name = name[0].upper() + name[1:]

            if name != self.booking_name:
                self.notes.append(f"The booking will be under the name {name}.")

            self.booking_name = name

    # -------------------------------------------------------------------------------------------
    # What the model is told
    # -------------------------------------------------------------------------------------------

    def fact_lines(self) -> list[str]:
        """Return the state of this dialogue as the lines that go into the prompt."""
        now = self.clock()

        # What this turn did and what it still needs come FIRST. With the state lists first,
        # Llama 3.2 3B read the first reminder in the list back as the one it had just set (PACE
        # job 7163276, turn 10). The wall-clock "NOW" line was dropped: no family needs it, and it
        # made two runs of the same recording send different prompts.
        lines = []

        if self.notes:
            lines.append("WHAT JUST HAPPENED: " + " ".join(self.notes))

        if self.needed:
            lines.append(
                f"MISSING: {NEEDED_IN_WORDS[self.needed]}. Ask one short question for it."
            )

        active = [timer for timer in self.timers if not timer.cancelled and timer.seconds > 0]

        # Empty lists are left out: "TIMERS: none running." on every lookup turn was noise the
        # model sometimes answered instead of the question.
        if active:
            lines.append("TIMERS: " + " | ".join(timer.describe(now) for timer in active))

        if self.reminders:
            lines.append(
                "REMINDERS: " + " | ".join(reminder.describe() for reminder in self.reminders)
            )

        if self.bookings:
            lines.append("BOOKINGS: " + " | ".join(booking.describe() for booking in self.bookings))

        if not self.request.is_empty() or self.proposed_room is not None:
            lines.append(f"ROOM REQUEST SO FAR: {self.request.describe()}.")

        return lines
