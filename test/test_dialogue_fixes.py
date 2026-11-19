"""Sanity checks for the parser and dialogue-state fixes found in PACE job 7162952.

    python3 test/test_dialogue_fixes.py

Each case is an ASR transcript that job 7162952 actually produced (or the reference it should
have been), and the value the dialogue state must read from it. No model and no server needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain  # noqa: E402
import knowledge_base  # noqa: E402
import spoken_text  # noqa: E402
from dialogue_state import DialogueState  # noqa: E402


def check(label: str, got, expected) -> None:
    status = "PASS" if got == expected else "FAIL"
    print(f"{status}  {label}: got {got!r}, expected {expected!r}")
    assert got == expected


def clock(text: str) -> str | None:
    parsed = spoken_text.parse_clock_time(text)
    return parsed.as_24_hour() if parsed else None


def run_dialogue(utterances: list[str]) -> DialogueState:
    state = DialogueState(knowledge_base.load(knowledge_base.DEFAULT_PATH))
    for utterance in utterances:
        state.observe(utterance)
    return state


def main() -> int:
    # 1. "p.m." survives normalise() as "p m"; it used to be read as no time at all.
    check("5 p.m.", clock("5 p.m. under the name Asa."), "17:00")
    check("2:30 p.m.", clock("2:30 p.m. today"), "14:30")
    check("at 6 a.m.", clock("at 6 a.m."), "06:00")
    check("ten o'clock", clock("Ten o'clock, under the name Ravi."), "10:00")

    # 2. "the 19th" keeps its suffix through digitise(); it used to be read as no date.
    check("on the 19th", spoken_text.parse_date("on the 19th."), "the 19th")

    # 3. A trailing "for NAME" is a booking name, but "for ten" / "for Thursday" are not.
    check("then for Mariel", spoken_text.parse_person_name("Saturday at 11 then for Mariel."), "Mariel")
    check("a room for ten", spoken_text.parse_person_name("a room for ten"), None)
    check("for Thursday", spoken_text.parse_person_name("book a room for Thursday"), None)

    # 4. Mid-booking, a turn with only a time and a name is part of the booking (d12 t04).
    state = run_dialogue([
        "Can I get a room with a projector for eight people on Friday?",
        "One in the afternoon works.",
        "Hang on, there'll be 12 of us now, not 8.",
        "5 p.m. under the name Asa.",
    ])
    check("d12 hour after '5 p.m.'", state.request.hour, "17:00")
    check("d12 name", state.booking_name, "Asa")
    check("d12 proposed room", state.proposed_room and state.proposed_room["name"], "Maple Hall")

    # 5. Name given before "yes" is kept, so "yes" books it (d11).
    state = run_dialogue([
        "I'd like to book your room on Monday morning.",
        "4 of us and we need the screen to watch a video",
        "channel clock under the name Ravey",
        "Yes, please.",
    ])
    check("d11 booked", [booking.name for booking in state.bookings], ["Ravey"])

    # 6. "started" sets a timer even without the word "timer" (ASR: "Also started 20 minute phone").
    state = run_dialogue(["Also started 20 minute phone for the laundry."])
    check("started timer seconds", [timer.seconds for timer in state.timers], [1200])

    # 7. Leaked prompt labels are cut out of a spoken sentence.
    check(
        "leak filter",
        brain.strip_leaked_prompt("Laminating costs 1 dollar per page. Values acted on: - length")[0],
        "Laminating costs 1 dollar per page.",
    )

    check(
        "lower-case 'reminders:' is not a leak",
        brain.strip_leaked_prompt("You have two reminders: water the plants at six.")[0],
        "You have two reminders: water the plants at six.",
    )
    check(
        "label leak",
        brain.strip_leaked_prompt("Booked. TIMERS: none running.")[0],
        "Booked.",
    )

    # 10. Clock and head-count phrasings found in the unseen-dialogue check.
    check("three thirty pm", clock("at three thirty pm"), "15:30")
    check("quarter to five", clock("quarter to five"), "16:45")
    check("first thing is no hour", clock("first thing in the morning"), None)
    check("just me", spoken_text.parse_people("Book the Oak Nook for Friday at 1, it's just me."), 1)
    check("just me and one other", spoken_text.parse_people("Just me and one other person."), 2)

    # 9. A bare name right after the agent asked for one completes the booking (probe 7167867).
    state = run_dialogue([
        "Book me something with a video call kit for Thursday at 2.",
        "Sounds good.",
        "Lee.",
    ])
    check("bare name books", [booking.name for booking in state.bookings], ["Lee"])

    # 8. Only 24-hour times are rewritten for speech; a 12-hour time the model wrote is kept.
    check("24-hour time spoken", brain.spoken_numbers("free on thursday at 15:00."), "free on thursday at 3 PM.")
    check("opening hours range", brain.spoken_numbers("sunday 10:00-16:00"), "sunday 10 AM to 4 PM")
    check("12-hour time kept", brain.spoken_numbers("at 3:30 in the afternoon"), "at 3:30 in the afternoon")
    check("10:00 with pm kept", brain.spoken_numbers("at 10:00 pm"), "at 10:00 pm")

    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
