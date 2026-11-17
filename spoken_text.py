"""Reading values out of what the user said. Owned by Partner B.

The brain hands the transcript to a locally served chat model, but it does NOT ask that model to do
arithmetic or to remember state. Durations, clock times, days, head counts and room features are
pulled out here, in Python, and the model is told what was found. The reason is in the task families:
"how much is left" has a right answer that is a subtraction from the moment the timer was set, and a
booking must not propose a room that breaks a constraint. A 1B-3B instruct model asked to do either
from a prompt gets it wrong often enough to cost task-success points, and it costs them silently,
because a wrong number is spoken in exactly the same confident sentence as a right one.

Everything here is deliberately conservative. A value is only reported when the utterance really
contains it; anything unrecognised comes back as ``None`` and the brain's prompt then says the value
is missing, which is what makes the agent ask for it instead of inventing it.

This module holds no state and knows nothing about the knowledge base. It is pure text to values, so
it can be tested on a transcript alone:

    PYTHONPATH=. python3 -c "import spoken_text; print(spoken_text.parse_duration('half an hour'))"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Number words up to the sizes a spoken request actually uses: minutes, hours, head counts and
# clock times. Anything larger is written as digits by the ASR anyway.
UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60}

# Ordinals turn up in dates ("the eighteenth of november") and are read as their number.
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}

DAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)

MONTHS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)

# Phrases that mean a length but contain no number to match. They are REWRITTEN rather than
# returned, so that "an hour and ten minutes" still sums both parts instead of stopping at the hour.
FIXED_DURATION_REWRITES = (
    (re.compile(r"\bhour and a half\b"), "90 minutes"),
    (re.compile(r"\bhalf an hour\b"), "30 minutes"),
    (re.compile(r"\bhalf hour\b"), "30 minutes"),
    (re.compile(r"\bquarter of an hour\b"), "15 minutes"),
    (re.compile(r"\ban hour\b"), "60 minutes"),
    (re.compile(r"\ba minute\b"), "60 seconds"),
)

DURATION_PATTERN = re.compile(
    r"\b(\d{1,4})\s*(?:more|extra|additional|further|further\s+)?\s*"
    r"(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
)
SECONDS_PER_UNIT = {"h": 3600, "m": 60, "s": 1}

CLOCK_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?")
HOUR_MERIDIEM_PATTERN = re.compile(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)")
OCLOCK_PATTERN = re.compile(r"\b(\d{1,2})\s*o'?\s?clock\b")
# "two thirty" survives digitising as "2 30", which is only a clock time when the sentence is
# talking about a time at all; without this guard "for 5 30 minute timers" would become 05:30.
SPLIT_CLOCK_PATTERN = re.compile(r"\b(\d{1,2})\s+(\d{2})\b")
TIME_CUES = re.compile(r"\b(at|by|around|today|tomorrow|tonight|o'?clock|am|pm|a\.m\.|p\.m\.)\b")

PART_OF_DAY = (
    (re.compile(r"\b(in the evening|at night|tonight|this evening)\b"), "pm"),
    (re.compile(r"\b(in the afternoon|this afternoon)\b"), "pm"),
    (re.compile(r"\b(in the morning|this morning)\b"), "am"),
)

PEOPLE_PATTERN = re.compile(
    r"\b(?:for|holds?|seats?|fits?|group of|party of|room for|table for)\s+(\d{1,3})\b"
)
PEOPLE_SUFFIX_PATTERN = re.compile(r"\b(\d{1,3})\s+(?:people|persons?|of us|guests?)\b")

# The canonical feature strings are the ones in toy_kb/knowledge_base.json. Everything a user might
# say for one of them maps to the canonical string, because the room search compares exact strings.
FEATURE_SYNONYMS = {
    "projector": "projector",
    "projection": "projector",
    "screen to project": "projector",
    "whiteboard": "whiteboard",
    "white board": "whiteboard",
    "board": "whiteboard",
    "tv": "tv screen",
    "tv screen": "tv screen",
    "television": "tv screen",
    "monitor": "tv screen",
    "quiet": "quiet",
    "silent": "quiet",
    "video call": "video call kit",
    "video conference": "video call kit",
    "conference call": "video call kit",
    "zoom": "video call kit",
    "webcam": "video call kit",
    "camera": "video call kit",
    "microphone": "microphone",
    "mic": "microphone",
    "stage": "stage",
}


@dataclass(frozen=True)
class ClockTime:
    """A time of day that was actually said, kept with the words it was said in."""

    hour: int
    minute: int
    said_as: str

    def as_24_hour(self) -> str:
        """Return ``HH:MM``, the form the knowledge base uses for free hours and opening times."""
        return f"{self.hour:02d}:{self.minute:02d}"

    def spoken(self) -> str:
        """Return a form that reads aloud, so the agent can repeat the time back to the user."""
        hour_12 = self.hour % 12 or 12
        meridiem = "in the morning" if self.hour < 12 else (
            "in the afternoon" if self.hour < 18 else "in the evening"
        )

        if self.minute == 0:
            return f"{hour_12} o'clock {meridiem}"

        return f"{hour_12}:{self.minute:02d} {meridiem}"


def normalise(text: str) -> str:
    """Return ``text`` lowercased, with the punctuation that carries no meaning here removed.

    Colons and apostrophes survive: "2:30" and "o'clock" are both times, and dropping either would
    turn a clock time into two unrelated numbers.
    """
    lowered = text.lower()
    kept = re.sub(r"[^a-z0-9:'\s]", " ", lowered)

    return re.sub(r"\s+", " ", kept).strip()


def digitise(text: str) -> str:
    """Return ``text`` with spoken numbers rewritten as digits.

    "twenty five minutes" becomes "25 minutes", "the eighteenth" becomes "the 18". Compounds are
    joined ("twenty" + "five"), which is why this is a scan and not a dictionary lookup.
    """
    words = normalise(text).split()
    output: list[str] = []
    index = 0

    while index < len(words):
        word = words[index].strip("'")

        if word in TENS:
            value = TENS[word]

            # "twenty five" is one number; "twenty minutes" is not.
            if index + 1 < len(words) and words[index + 1] in UNITS and UNITS[words[index + 1]] < 10:
                value += UNITS[words[index + 1]]
                index += 1

            output.append(str(value))

        elif word in UNITS:
            output.append(str(UNITS[word]))

        elif word in ORDINALS:
            output.append(str(ORDINALS[word]))

        elif word in ("a", "an") and index + 1 < len(words) and words[index + 1] in (
            "minute", "minutes", "hour", "hours", "second", "seconds"
        ):
            # "a 20 minute one" is already a number; "a minute" is one minute.
            output.append("1")

        else:
            output.append(word)

        index += 1

    return " ".join(output)


def parse_duration(text: str) -> int | None:
    """Return how many seconds the utterance asks for, or ``None`` when it names no length.

    Several lengths in one sentence are summed, so "an hour and ten minutes" is 4200 seconds.
    """
    # The fixed phrases are rewritten BEFORE digitising, because digitising is what destroys them:
    # "half an hour" contains "an hour", which becomes "1 hour", and the phrase would then be read
    # as sixty minutes rather than thirty.
    plain = normalise(text)

    for pattern, replacement in FIXED_DURATION_REWRITES:
        plain = pattern.sub(replacement, plain)

    digits = digitise(plain)

    total = 0
    found = False

    for count, unit in DURATION_PATTERN.findall(digits):
        total += int(count) * SECONDS_PER_UNIT[unit[0]]
        found = True

    return total if found and total > 0 else None


def parse_bare_number(text: str) -> int | None:
    """Return the only number in the utterance, or ``None`` when there is not exactly one.

    "Actually make it fifteen" names a length with no unit at all, and it is only interpretable
    against something already running — which is why the unit is supplied by the caller
    (``dialogue_state``) and never guessed here.
    """
    numbers = re.findall(r"\b(\d{1,4})\b", digitise(text))

    if len(numbers) != 1:
        return None

    return int(numbers[0])


def _meridiem_from_context(text: str) -> str | None:
    """Return "am" or "pm" when the sentence says which part of the day it means."""
    for pattern, meridiem in PART_OF_DAY:
        if pattern.search(text):
            return meridiem

    return None


def _to_24_hour(hour: int, meridiem: str | None) -> int:
    """Return the 24-hour hour, guessing only where a listener would guess the same way.

    With no "am" or "pm" and no part of the day, 1 to 6 is read as the afternoon and 7 to 12 as
    said. The venue opens at 07:00 and closes at 21:00, so "at four" is four in the afternoon for
    every room in the knowledge base, while "at eight" is genuinely ambiguous and is left as 08:00
    rather than invented. The agent repeats the time back, which is where a wrong guess gets caught.
    """
    if meridiem == "pm":
        return hour if hour == 12 else hour + 12

    if meridiem == "am":
        return 0 if hour == 12 else hour

    if 1 <= hour <= 6:
        return hour + 12

    return hour


def parse_clock_time(text: str) -> ClockTime | None:
    """Return the time of day the utterance names, or ``None``."""
    digits = digitise(text)
    meridiem = _meridiem_from_context(digits)

    if re.search(r"\b(noon|midday)\b", digits):
        return ClockTime(12, 0, "noon")

    if re.search(r"\bmidnight\b", digits):
        return ClockTime(0, 0, "midnight")

    match = CLOCK_PATTERN.search(digits)

    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        said = match.group(3)
        marker = said.replace(".", "") if said else meridiem

        return ClockTime(_to_24_hour(hour, marker) % 24, minute, match.group(0).strip())

    match = HOUR_MERIDIEM_PATTERN.search(digits)

    if match:
        hour = int(match.group(1))

        return ClockTime(_to_24_hour(hour, match.group(2).replace(".", "")) % 24, 0, match.group(0))

    match = OCLOCK_PATTERN.search(digits)

    if match:
        hour = int(match.group(1))

        return ClockTime(_to_24_hour(hour, meridiem) % 24, 0, match.group(0))

    if TIME_CUES.search(digits):
        match = SPLIT_CLOCK_PATTERN.search(digits)

        if match and 1 <= int(match.group(1)) <= 12 and int(match.group(2)) < 60:
            hour, minute = int(match.group(1)), int(match.group(2))

            return ClockTime(_to_24_hour(hour, meridiem) % 24, minute, match.group(0))

        # "at six", with nothing after it, is still a time.
        bare = re.search(r"\b(?:at|around|by)\s+(\d{1,2})\b", digits)

        if bare and 1 <= int(bare.group(1)) <= 12:
            return ClockTime(_to_24_hour(int(bare.group(1)), meridiem) % 24, 0, bare.group(0))

    # "six in the evening" names no cue word at all: the part of the day IS the cue, and the only
    # number in the sentence is the hour.
    if meridiem is not None:
        bare = re.search(r"\b(\d{1,2})\b", digits)

        if bare and 1 <= int(bare.group(1)) <= 12:
            return ClockTime(_to_24_hour(int(bare.group(1)), meridiem) % 24, 0, bare.group(0))

    return None


def parse_day(text: str) -> str | None:
    """Return the weekday the utterance names, lowercased, or ``None``.

    Only a named weekday counts. "today" and "tomorrow" are deliberately not resolved here: the
    recordings are replayed on a different day from the one they were spoken on, so a date derived
    from the clock of the replay machine would be a fact the user never said. The brain passes
    "today" through to the model as the user's own words instead.
    """
    words = normalise(text).split()

    for day in DAYS:
        if day in words:
            return day

    return None


def parse_date(text: str) -> str | None:
    """Return a calendar date the utterance names, as it was said, or ``None``.

    Reminders in the recordings carry dates ("on the eighteenth of November") as often as times, and
    a date is kept as text because nothing downstream computes with it: the agent only says it back.
    """
    digits = digitise(text)

    for month in MONTHS:
        match = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?{month}\b", digits)

        if match:
            return f"{month.capitalize()} {int(match.group(1))}"

        match = re.search(rf"\b{month}\s+(\d{{1,2}})\b", digits)

        if match:
            return f"{month.capitalize()} {int(match.group(1))}"

    # "on the nineteenth", with the month left implied by the turn before it. The ordinal must have
    # been SAID as an ordinal: digitising turns "nineteenth" into "19", and without this check
    # "the 19" of "the 19 minutes" would be read as a date.
    plain = normalise(text)
    said_as_ordinal = any(word in ORDINALS for word in plain.split()) or re.search(
        r"\b\d{1,2}(?:st|nd|rd|th)\b", plain
    )

    if said_as_ordinal:
        match = re.search(r"\bthe (\d{1,2})\b", digits)

        if match and 1 <= int(match.group(1)) <= 31:
            day = int(match.group(1))
            suffix = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}.get(day, "th")

            return f"the {day}{suffix}"

    return None


def parse_people(text: str) -> int | None:
    """Return how many people the utterance mentions, or ``None``."""
    digits = digitise(text)

    for pattern in (PEOPLE_SUFFIX_PATTERN, PEOPLE_PATTERN):
        match = pattern.search(digits)

        if match:
            count = int(match.group(1))

            # "for 5 minutes" is a duration, not a head count, and both start with "for".
            after = digits[match.end():].lstrip()

            if after.startswith(("minute", "hour", "second", "min", "hr")):
                continue

            if 1 <= count <= 200:
                return count

    return None


def parse_features(text: str) -> list[str]:
    """Return the room features the utterance asks for, as the knowledge base spells them."""
    digits = normalise(text)
    found: list[str] = []

    for phrase, canonical in FEATURE_SYNONYMS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", digits) and canonical not in found:
            found.append(canonical)

    return found


def parse_person_name(text: str) -> str | None:
    """Return the name a booking is to be made under, or ``None``.

    Only an explicit "under the name X" / "for X" after a booking verb counts. A name is the one
    value in this file that cannot be checked against anything, so a loose rule here would book
    rooms under words like "sure".
    """
    match = re.search(
        r"\b(?:under the name|in the name of|name is|under|it's for|book it for)\s+([a-z]+)\b",
        text.lower(),
    )

    if not match:
        return None

    word = match.group(1)

    if word in ("me", "us", "the", "a", "an", "my", "our", "it", "that", "tomorrow", *DAYS):
        return None

    return word.capitalize()


def says_yes(text: str) -> bool:
    """Return whether the utterance is an agreement, for the "shall I book it?" turn."""
    digits = normalise(text)

    return bool(
        re.search(
            r"\b(yes|yeah|yep|sure|please do|go ahead|sounds good|that works|book it|do it|ok|okay)\b",
            digits,
        )
    )


def says_no(text: str) -> bool:
    """Return whether the utterance is a refusal."""
    digits = normalise(text)

    return bool(re.search(r"\b(no|nope|don't|do not|never mind|forget it|cancel that)\b", digits))
