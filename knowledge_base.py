"""The toy knowledge base, as facts a prompt can carry. Owned by Partner B.

``toy_kb/knowledge_base.json`` is about 6 kB, roughly 1500 tokens. Sending all of it on every turn
is the easy choice and it is not free: the whole file is re-read by the model on every prefill, and
that time lands squarely in ``llm_first_token``, the one stage the headline latency number is made
of. Sending none of it is worse, because then the agent invents opening hours.

So the file is cut into FACTS — one short line each — and only the facts a turn actually needs are
put in the prompt. Which policy runs is a setting of the agent (``--llm-kb-mode``), not a constant,
because the assignment asks for the choice to be deliberate and then measured:

    relevant  (default)  the facts that match the question, at most --llm-kb-facts of them
    full                 every fact, every turn: the baseline the default is measured against
    none                 no facts at all: what the model knows on its own, which is nothing

Matching is keyword overlap with a small synonym table, not an embedding model. A second model in
the turn would cost more latency than the prefill it saves, and on a knowledge base with seven rooms
and four policies the keywords are the content.

Room search is here too, and it is NOT a prompt trick: the booking family is graded on never
proposing a room that breaks a constraint, so the rooms that satisfy the constraints are worked out
in Python and the model is handed the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("toy_kb/knowledge_base.json")

# Words that carry no topic. Dropping them keeps "where is the quiet lounge" from matching every
# fact that happens to contain "the".
STOP_WORDS = frozenset(
    """a an the and are as at be bring by can could do does doing for from get give go going have has
    how i if in is it its me my need of on or our out please she he they them there this that
    to up us want was we what when where which who why will with would you your do don't i'm
    it's there's much many any some tell know like just""".split()
)

# What a user says mapped to what the knowledge base calls it. One entry per way of asking that
# the task families actually show.
SYNONYMS = {
    "close": "hours", "closes": "hours", "closing": "hours", "open": "hours",
    "opens": "hours", "opening": "hours", "hour": "hours", "time": "hours",
    "late": "hours", "early": "hours", "today": "hours", "weekend": "hours",
    "cost": "price", "costs": "price", "charge": "price", "fee": "price",
    "dollars": "price", "cents": "price", "expensive": "price", "cheap": "price",
    "print": "printing", "prints": "printing", "printer": "printing", "copy": "printing",
    "locker": "lockers", "padlock": "lockers", "lock": "lockers",
    "laminate": "laminating", "lamination": "laminating",
    "coffee": "cafe", "snack": "cafe", "snacks": "cafe", "eat": "cafe", "drink": "cafe",
    "quiet": "quiet lounge", "silence": "quiet lounge", "study": "quiet lounge",
    "reading": "quiet lounge", "lounge": "quiet lounge",
    "tools": "makerspace", "workshop": "makerspace", "maker": "makerspace",
    "kid": "children", "kids": "children", "child": "children",
    "guest": "guests", "visitor": "guests", "friend": "guests",
    "laptop": "laptop loans", "laptops": "laptop loans", "borrow": "laptop loans",
    "book": "room booking", "booking": "room booking", "reserve": "room booking",
    "ahead": "room booking", "advance": "room booking", "long": "room booking",
    "rooms": "room", "meeting": "room",
    "desk": "front desk", "reception": "front desk", "staff": "front desk",
}

# A fact the turn is about is worth more than a fact that merely mentions the word. Title hits are
# what tell "where is the quiet lounge" (the lounge) from "can I print in the lounge" (printing).
TITLE_WEIGHT = 3
TEXT_WEIGHT = 1

# What a fact has to score to be worth prompt space. A title hit is 3, so any fact the question is
# actually ABOUT passes; one incidental word in a long line scores 1 and does not. Without this,
# "set a timer for 5 minutes for the pasta" retrieved four facts about the building, because the
# room-booking policy happens to contain the word "minutes" — prefill spent on nothing, on a turn
# the knowledge base has no part in.
MINIMUM_SCORE = 2


@dataclass(frozen=True)
class Fact:
    """One line of the knowledge base, with the words that should retrieve it."""

    id: str
    kind: str
    title: str
    line: str
    title_words: frozenset[str]
    text_words: frozenset[str]


def content_words(text: str) -> frozenset[str]:
    """Return the meaningful words of ``text``, with synonyms folded into KB vocabulary."""
    words = {word for word in re.findall(r"[a-z]+", text.lower()) if word not in STOP_WORDS}
    expanded = set(words)

    for word in words:
        if word in SYNONYMS:
            expanded.update(SYNONYMS[word].split())

    return frozenset(expanded)


def _hours_line(opening_hours: dict) -> str:
    """Return every day's opening hours as one line, in the order a week runs."""
    order = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    parts = [
        f"{day} {opening_hours[day]['open']}-{opening_hours[day]['close']}"
        for day in order
        if day in opening_hours
    ]

    return "Opening hours: " + ", ".join(parts) + "."


def _room_line(room: dict) -> str:
    """Return one room as one line: what it holds, what it has and when it is free."""
    features = ", ".join(room["features"]) if room["features"] else "no listed features"
    free = "; ".join(
        f"{day} {', '.join(hours)}" for day, hours in sorted(room["free_hour_starts"].items())
    )

    return (
        f"{room['name']} (floor {room['floor']}): holds {room['capacity']}, "
        f"has {features}. Free hour starts: {free}."
    )


class KnowledgeBase:
    """The venue file, as retrievable facts and as a room search."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        self.venue_name = self.data.get("venue", {}).get("name", "this place")
        self.rooms = list(self.data.get("rooms", []))
        self.facts = self._build_facts()

    def _build_facts(self) -> list[Fact]:
        """Return every fact in the file, each already reduced to the line a prompt would carry."""
        facts: list[Fact] = []

        def add(identifier: str, kind: str, title: str, line: str, extra: str = "") -> None:
            facts.append(
                Fact(
                    id=identifier,
                    kind=kind,
                    title=title,
                    line=line,
                    title_words=content_words(f"{title} {extra}"),
                    text_words=content_words(line),
                )
            )

        if "opening_hours" in self.data:
            add(
                "opening_hours",
                "hours",
                "opening hours",
                _hours_line(self.data["opening_hours"]),
                extra="open close hours day week " + " ".join(self.data["opening_hours"]),
            )

        for area in self.data.get("areas", []):
            add(
                area["id"],
                "area",
                area["name"],
                f"{area['name']} (floor {area['floor']}): {area['text']}",
                extra="where floor located",
            )

        for service in self.data.get("services", []):
            add(
                service["id"],
                "service",
                service["name"],
                f"{service['name']} (floor {service['floor']}): {service['text']}",
                extra="price cost service",
            )

        for policy in self.data.get("policies", []):
            add(
                policy["id"],
                "policy",
                policy["topic"],
                f"Policy on {policy['topic']}: {policy['text']}",
                extra="rule policy allowed",
            )

        for room in self.rooms:
            add(
                room["id"],
                "room",
                room["name"],
                _room_line(room),
                # Deliberately NOT "room" or "book": a booking turn does not retrieve its rooms
                # by keyword, it gets them from rooms_matching() below, and leaving those words in
                # the title made every room outrank the room-booking POLICY on "how far ahead can
                # I book a room". The room's own name and its features are what should retrieve it.
                extra="capacity " + " ".join(room["features"]),
            )

        return facts

    def relevant(self, text: str, limit: int = 4) -> list[Fact]:
        """Return the facts that best match ``text``, best first, or nothing when none match."""
        asked = content_words(text)

        if not asked:
            return []

        scored: list[tuple[int, int, Fact]] = []

        for index, fact in enumerate(self.facts):
            score = TITLE_WEIGHT * len(asked & fact.title_words) + TEXT_WEIGHT * len(
                asked & fact.text_words - fact.title_words
            )

            if score >= MINIMUM_SCORE:
                # The index is the tie-breaker, so the same question always retrieves the same
                # facts in the same order: a prompt that changes between runs is not reproducible.
                scored.append((-score, index, fact))

        scored.sort()

        return [fact for _, _, fact in scored[:limit]]

    def all_lines(self) -> list[str]:
        """Return every fact as a line, for ``--llm-kb-mode full``."""
        return [fact.line for fact in self.facts]

    def rooms_matching(
        self,
        capacity: int | None = None,
        features: tuple[str, ...] = (),
        day: str | None = None,
        hour: str | None = None,
    ) -> list[dict]:
        """Return the rooms that satisfy every constraint given, in capacity order.

        A constraint that is ``None`` is not applied, so this answers "what is free on Thursday"
        and "what holds ten with a projector" with the same call. Smallest room first, because a
        two-person booking should not be offered the twenty-person hall while a nook is free.
        """
        matches = []

        for room in self.rooms:
            if capacity is not None and room["capacity"] < capacity:
                continue

            if any(feature not in room["features"] for feature in features):
                continue

            if day is not None:
                free_hours = room["free_hour_starts"].get(day, [])

                if not free_hours:
                    continue

                if hour is not None and hour not in free_hours:
                    continue

            matches.append(room)

        return sorted(matches, key=lambda room: room["capacity"])

    def describe_conflict(
        self,
        capacity: int | None,
        features: tuple[str, ...],
        day: str | None,
        hour: str | None,
    ) -> str:
        """Return why nothing matched, and the nearest thing that does exist.

        The booking family is explicit that a failure must name WHICH constraint cannot be met with
        which, and offer the closest alternative. Working that out in Python is the only way the
        model can say it without inventing a room.

        The order matters and was wrong at first. Asking "does any room hold twelve AND have a
        projector" is only the right question when the answer is no: Maple Hall holds twenty and has
        one, so a search that failed on the DAY was being explained as a capacity conflict, and the
        sentence that came out ("Maple Hall holds 12 but has no projector") was false about a room
        the file describes correctly. So the capacity-and-feature pair is tested on its own first,
        and only when it genuinely has no room does the explanation talk about capacity at all.
        """
        wanted = " and ".join(features) if features else "that"
        fits_ignoring_time = self.rooms_matching(capacity=capacity, features=features)

        if not fits_ignoring_time:
            by_size = self.rooms_matching(capacity=capacity)
            by_feature = self.rooms_matching(features=features)

            if capacity is not None and features and by_size and by_feature:
                return (
                    f"No room both holds {capacity} and has {wanted}. "
                    f"{by_feature[-1]['name']} has {wanted} but holds only {by_feature[-1]['capacity']}. "
                    f"{by_size[0]['name']} holds {by_size[0]['capacity']} but has no {wanted}."
                )

            if features and not by_feature:
                return f"No room here has {wanted}."

            if capacity is not None and not by_size:
                largest = max(self.rooms, key=lambda room: room["capacity"])

                return (
                    f"No room holds {capacity}. The largest is {largest['name']}, "
                    f"which holds {largest['capacity']}."
                )

            return "Nothing in the knowledge base matches that request."

        # The room exists; it is the day or the hour that does not work.
        if day is not None:
            free_that_day = [
                room for room in fits_ignoring_time if room["free_hour_starts"].get(day)
            ]

            if not free_that_day:
                room = fits_ignoring_time[0]
                days = ", ".join(sorted(room["free_hour_starts"]))

                return (
                    f"{room['name']} fits (holds {room['capacity']}, has "
                    f"{', '.join(room['features'])}) but is free only on {days}, not {day}."
                )

            return (
                f"Nothing is free at that hour on {day}. Free then: "
                + "; ".join(
                    # The room's size and equipment are given too: with the name alone the model
                    # guessed Maple Hall "holds only 10" (it holds 20; job 7174252).
                    f"{room['name']} (holds {room['capacity']}, has {', '.join(room['features'])}) "
                    f"at {', '.join(room['free_hour_starts'][day])}"
                    for room in free_that_day[:3]
                )
                + "."
            )

        return "Nothing in the knowledge base matches that request."


def load(path: Path | str = DEFAULT_PATH) -> KnowledgeBase:
    """Return the knowledge base, read once at start-up and kept in memory."""
    return KnowledgeBase(path)
