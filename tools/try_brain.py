"""Talk to the brain in text, with no audio and no shim. Owned by Partner B.

The turn loop is a slow way to find out whether a reply is any good: it replays every recording at
real time. This runs the same :class:`brain.LlmBrain` over text, one dialogue at a time, and prints
what the model said and how long its first sentence took. It is for judging ANSWERS, not latency:
the numbers printed here are measured on this machine, outside the shim, and no report quotes them.

    python3 tools/try_brain.py --references references.json --speaker asal
    python3 tools/try_brain.py --say "set a timer for five minutes" --say "how long is left?"
    python3 tools/try_brain.py --references references.json --show-prompt --dialogue d11
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain as brain_module  # noqa: E402

NANOSECONDS_PER_MILLISECOND = 1_000_000


def dialogues_from_references(path: Path, speaker: str | None) -> dict[str, list[str]]:
    """Return the reference transcripts grouped into the dialogues their file names mark."""
    data = json.loads(path.read_text())
    speakers = {speaker: data[speaker]} if speaker else data
    grouped: dict[str, list[str]] = collections.defaultdict(list)

    for name, utterances in speakers.items():
        for file_name, said in sorted(utterances.items()):
            parts = file_name.split("_")
            dialogue = parts[1] if len(parts) > 2 else name
            grouped[f"{name}/{dialogue}"].append(said)

    return dict(grouped)


def main() -> int:
    """Run every dialogue through the brain and print the conversation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--speaker", default=None, help="Only this speaker's dialogues.")
    parser.add_argument("--dialogue", default=None, help="Only the dialogue whose id contains this.")
    parser.add_argument("--say", action="append", default=[], help="One turn; repeatable.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt of each turn.")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-kb-mode", choices=brain_module.KB_MODES, default=None)
    arguments = parser.parse_args()

    overrides = {
        name: value
        for name, value in (
            ("model", arguments.llm_model),
            ("base_url", arguments.llm_base_url),
            ("kb_mode", arguments.llm_kb_mode),
        )
        if value is not None
    }
    agent = brain_module.LlmBrain(brain_module.BrainSettings(**overrides))

    print(f"brain settings    {agent.describe()}")
    started = time.monotonic()
    agent.warmup()
    print(f"warmed up in      {(time.monotonic() - started) * 1000:.0f} ms\n")

    if arguments.say:
        dialogues = {"command line": list(arguments.say)}

    elif arguments.references:
        dialogues = dialogues_from_references(arguments.references, arguments.speaker)

    else:
        raise SystemExit("Give --references or at least one --say.")

    for name, turns in dialogues.items():
        if arguments.dialogue and arguments.dialogue not in name:
            continue

        print(f"=== {name} ===")
        agent.reset()

        for said in turns:
            print(f"  USER   {said}")

            if arguments.show_prompt:
                agent.on_prompt = lambda messages: [
                    print(f"         [{message['role']}] {message['content']}")
                    for message in messages
                ]

            started = time.monotonic_ns()
            first_sentence_milliseconds = None
            sentences = []

            for sentence in agent.reply(said):
                if first_sentence_milliseconds is None:
                    first_sentence_milliseconds = (
                        time.monotonic_ns() - started
                    ) / NANOSECONDS_PER_MILLISECOND

                sentences.append(sentence)

            whole = (time.monotonic_ns() - started) / NANOSECONDS_PER_MILLISECOND
            print(f"  AGENT  {' '.join(sentences)}")
            print(
                f"         first sentence {first_sentence_milliseconds:.0f} ms, "
                f"whole reply {whole:.0f} ms (this machine, not the trace)"
            )

        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
