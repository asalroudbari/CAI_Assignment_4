"""Sanity checks for DialogueState.apply_actions, the executor behind --llm-nlu llm.

    python3 test/test_intent_actions.py

The actions are written by hand, in the JSON the intent reader returns, for utterances the rules
miss ("Could you time my eggs for seven minutes?", "Just me and one other person"). No model and no
server needed: this checks that the Python side does the right thing with a correct reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import knowledge_base  # noqa: E402
from dialogue_state import NEEDED_BOOKING_NAME, NEEDED_TIMER_DURATION, DialogueState  # noqa: E402


class Clock:
    """A clock the test moves by hand, so "how much is left" has one right answer."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def check(label: str, got, expected) -> None:
    status = "PASS" if got == expected else "FAIL"
    print(f"{status}  {label}: got {got!r}, expected {expected!r}")
    assert got == expected


def main() -> int:
    kb = knowledge_base.load(knowledge_base.DEFAULT_PATH)
    clock = Clock()
    state = DialogueState(kb, clock=clock)

    # "Could you time my eggs for seven minutes?" / "And another one, twenty minutes for the laundry."
    state.apply_actions([{"type": "start_timer", "minutes": 7, "label": "eggs"}])
    state.apply_actions([{"type": "start_timer", "minutes": 20, "label": "laundry"}])
    check("two timers", [(t.label, t.seconds) for t in state.timers], [("eggs", 420), ("laundry", 1200)])

    # "Give the eggs two more minutes." two minutes after they were set: 7 - 2 + 2 = 7 left.
    clock.now += 120
    state.apply_actions([{"type": "add_to_timer", "minutes": 2, "label": "eggs"}])
    check("eggs extended", state.timers[0].seconds, 540)
    check("eggs left", round(state.timers[0].remaining_seconds(clock())), 420)

    # "Cancel the eggs one." cancels the eggs, not the most recent timer.
    state.apply_actions([{"type": "cancel_timer", "label": "eggs"}])
    check("eggs cancelled, laundry kept", [t.cancelled for t in state.timers], [True, False])

    # "Put a timer for me." then "Half an hour.": the length is asked for, then filled in.
    state = DialogueState(kb, clock=clock)
    state.apply_actions([{"type": "start_timer"}])
    check("length asked for", state.needed, NEEDED_TIMER_DURATION)
    state.apply_actions([{"type": "start_timer", "minutes": 30}])
    check("one 30-minute timer", [(t.seconds, t.cancelled) for t in state.timers], [(1800, False)])

    # "Remind me at 3:30 pm to call my mom." then "Make that four."
    state = DialogueState(kb, clock=clock)
    state.apply_actions([{"type": "set_reminder", "subject": "call my mom", "time": "15:30"}])
    state.apply_actions([{"type": "change_reminder", "subject": "call my mom", "time": "16:00"}])
    check("reminder moved", [(r.subject, r.when) for r in state.reminders],
          [("call my mom", "4 o'clock in the afternoon")])

    # "Book me something with a video call kit for Thursday at 2." / "Just me and one other person."
    # / "Yes." / "Lee.": Fir Studio is the only room with a video call kit, free Thursday at 14:00.
    state = DialogueState(kb, clock=clock)
    state.apply_actions([{"type": "book_room", "features": ["video call kit"], "day": "thursday", "time": "14:00"}])
    state.apply_actions([{"type": "book_room", "people": 2}])
    check("room proposed", state.proposed_room and (state.proposed_room["name"], state.proposed_room["hour"]),
          ("Fir Studio", "14:00"))
    state.apply_actions([{"type": "confirm_booking"}])
    check("name asked for", state.needed, NEEDED_BOOKING_NAME)
    state.apply_actions([{"type": "book_room", "name": "lee"}])
    state.apply_actions([{"type": "confirm_booking"}])
    check("booked", [b.describe() for b in state.bookings], ["Fir Studio (floor 2) on thursday at 14:00 for Lee"])

    # A lookup question changes nothing and keeps the pending question.
    state.apply_actions([{"type": "start_timer"}])
    state.apply_actions([{"type": "question"}])
    check("pending kept over a question", state.needed, NEEDED_TIMER_DURATION)

    # The plausibility filter: a room and a name the user never said are dropped, and so is a
    # "how much is left" when no timer is running (Llama 3.2 3B produced all three in job 7167867).
    import intent_reader  # noqa: E402

    state = DialogueState(kb, clock=clock)
    kept = intent_reader.plausible(
        [{"type": "book_room", "people": 5, "day": "thursday", "room": "Maple Hall", "name": "Meeting Room 1"},
         {"type": "time_left", "label": "printing"}],
        state,
        "I need a room for five on Thursday.",
    )
    check("invented room and name dropped, impossible action dropped", kept,
          [{"type": "book_room", "people": 5, "day": "thursday"}])

    kept = intent_reader.plausible([{"type": "book_room", "day": "friday", "name": "Oak Nook,"}], state,
                                   "Book the Oak Nook for Friday at 1, it's just me.")
    check("room written as a name becomes the room", kept,
          [{"type": "book_room", "day": "friday", "time": "13:00", "people": 1, "room": "Oak Nook"}])

    state.proposed_room = {"name": "Maple Hall", "hour": "17:00"}  # a room is on offer
    kept = intent_reader.plausible([{"type": "confirm_booking", "name": "Asa"}], state, "5 p.m. under the name Asa.")
    check("no yes, no confirmation", kept, [{"type": "book_room", "name": "Asa", "time": "17:00"}])

    kept = intent_reader.plausible([{"type": "book_room", "day": "friday", "time": "00:00", "name": "? (not specified)"}],
                                   state, "Can I get a room with a projector for eight people on Friday?")
    check("invented midnight and non-name dropped", kept,
          [{"type": "book_room", "day": "friday", "people": 8, "features": ["projector"]}])

    state.proposed_room = None
    kept = intent_reader.plausible([{"type": "book_room", "people": 8, "time": "08:00"}], state,
                                   "Can I get a room with a projector for eight people on Friday?")
    check("time from a head count dropped", kept,
          [{"type": "book_room", "people": 8, "day": "friday", "features": ["projector"]}])
    kept = intent_reader.plausible([{"type": "book_room", "name": "cafe"}], state,
                                   "Can I have my coffee at the cafe on Sunday?")
    check("name-only booking with nothing under way dropped", kept, [])

    # "Remind me to study today": a date with no time of day still asks for the time, then joins them.
    state = DialogueState(kb, clock=clock)
    state.apply_actions([{"type": "set_reminder", "subject": "study", "date": "today"}])
    check("time asked for", state.needed, "reminder_time")
    state.apply_actions([{"type": "set_reminder", "time": "18:00"}])
    check("date and time joined", [r.when for r in state.reminders], ["today at 6 o'clock in the evening"])

    # "the deadline for the homework" means the deadline reminder, whatever subject the model named.
    state.apply_actions([{"type": "set_reminder", "subject": "qualification exam", "time": "09:00"}])
    state.apply_actions([{"type": "set_reminder", "subject": "deadline for the CAI course", "time": "10:00"}])
    kept = intent_reader.plausible([{"type": "change_reminder", "subject": "qualification exam", "date": "18th"}], state,
                                   "Actually, the deadline for the homework is on the 18th as well.")
    check("reminder chosen by the words said", kept[0]["subject"], "deadline for the CAI course")

    # The rules' values win: "for eight" is 8 people even when the model left it out, "at 1" is
    # 13:00, and "morning" alone is a part of the day, not a made-up hour.
    state = DialogueState(kb, clock=clock)
    kept = intent_reader.plausible([{"type": "book_room", "day": "wednesday"}], state,
                                   "Could I grab a room for eight with a projector on Wednesday?")
    check("people and feature from the rules", kept,
          [{"type": "book_room", "day": "wednesday", "people": 8, "features": ["projector"]}])
    kept = intent_reader.plausible([{"type": "book_room", "time": "07:00"}], state, "Morning if possible.")
    check("morning is a part of the day", kept, [{"type": "book_room", "part_of_day": "morning"}])

    # A second reminder while the first waits for its time is a new reminder, not the first one's time.
    state = DialogueState(kb, clock=clock)
    state.apply_actions([{"type": "set_reminder", "subject": "qualification exam", "date": "18th of November"}])
    state.apply_actions([{"type": "set_reminder", "subject": "deadline for the CAI course", "date": "19th"}])
    check("two reminders kept", [r.subject for r in state.reminders], ["qualification exam", "deadline for the CAI course"])

    # Two timers in one sentence keep their own lengths; the rules read only one for the sentence.
    state = DialogueState(kb, clock=clock)
    kept = intent_reader.plausible(
        [{"type": "start_timer", "minutes": 5, "label": "pasta"}, {"type": "start_timer", "minutes": 10, "label": "sauce"}],
        state, "Can you set two timers, five minutes for the pasta and ten for the sauce?")
    check("two timers keep their lengths", [a["minutes"] for a in kept], [5, 10])

    # A time the parser cannot read is kept when the words say a time; a head count is not a time.
    kept = intent_reader.plausible([{"type": "set_reminder", "subject": "call", "time": "15:30"}], state,
                                   "Remind me to call at half past three.")
    check("parsed time kept", kept[0]["time"], "15:30")

    # Where the rules misread the sense, the model's value stands: the day after "not", and the
    # half of the day ("dinner at 8").
    state = DialogueState(kb, clock=clock)
    kept = intent_reader.plausible([{"type": "book_room", "day": "tuesday"}], state, "Not Monday, Tuesday.")
    check("day after a negation", kept[0]["day"], "tuesday")
    kept = intent_reader.plausible([{"type": "set_reminder", "subject": "dinner", "time": "20:00"}], state,
                                   "Remind me about dinner at 8.")
    check("evening kept", kept[0]["time"], "20:00")

    # A named room too small for the new head count is let go, and the search finds another.
    state = DialogueState(kb, clock=clock)
    state.apply_actions([{"type": "book_room", "room": "Oak Nook", "day": "friday", "time": "13:00", "people": 1}])
    check("Oak Nook proposed", state.proposed_room and state.proposed_room["name"], "Oak Nook")
    state.apply_actions([{"type": "book_room", "people": 4}])
    check("another room for four", state.proposed_room and state.proposed_room["name"], "Cedar Room")

    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
