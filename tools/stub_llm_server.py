"""A stand-in LLM server, so the brain can be proved without a GPU. Owned by Partner B.

NOTHING THIS SERVER PRODUCES IS A MEASUREMENT. It exists to exercise the parts of the pipeline that
have nothing to do with a model: the HTTP client, the streaming parser, the sentence splitter, the
prompt assembly and the turn loop. It speaks the OpenAI streaming protocol and writes its "answer"
out of the FACTS block of the prompt it was given.

It serves a model named ``stub/not-a-real-model``. That is on purpose and it is the honesty
mechanism: to run the agent against this server you must pass ``--llm-model stub/not-a-real-model``,
the banner prints it, and the trace carries it, so a run made against the stub can never be mistaken
for a run against Llama.

    python3 tools/stub_llm_server.py --port 8000 &
    PYTHONPATH=. python3 run_agent.py --fake-asr --fake-tts \\
        --llm-base-url http://127.0.0.1:8000/v1 --llm-model stub/not-a-real-model
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STUB_MODEL = "stub/not-a-real-model"

# How long the stub waits before its first delta and between deltas. It stands in for prefill and
# for decoding, so a stub run has a shape roughly like a real one, and it is a setting so that the
# turn loop can be tested at both ends of the range.
FIRST_DELTA_SECONDS = 0.12
INTER_DELTA_SECONDS = 0.01


def answer_from_prompt(prompt: str) -> str:
    """Return something plausible to say, read out of the FACTS the brain put in the prompt."""
    missing = re.search(r"^MISSING: (.+?)\. Ask", prompt, re.MULTILINE)

    if missing:
        return f"Sorry, I need one more thing. Could you tell me {missing.group(1)}?"

    happened = re.search(r"^WHAT JUST HAPPENED: (.+)$", prompt, re.MULTILINE)

    if happened:
        return happened.group(1).strip()

    knowledge = prompt.split("KNOWLEDGE BASE:\n", 1)

    if len(knowledge) == 2:
        first_line = knowledge[1].splitlines()[0].strip()

        if first_line and "Nothing in the knowledge base" not in first_line:
            return f"Here is what I have. {first_line}"

    # A question about state ("how much is left", "read my reminders back") is answered from the
    # FACTS block, which is where a real model would read it too.
    state = [
        line
        for line in prompt.splitlines()
        if line.startswith(("TIMERS:", "REMINDERS:", "BOOKINGS:")) and "none" not in line
    ]

    if state:
        return "From what I have. " + " ".join(state)

    return "I do not have that. I can tell you about the cafe, the rooms and the opening hours."


def deltas_of(answer: str) -> list[str]:
    """Return the answer cut into pieces, so the client sees a stream rather than one blob."""
    pieces = re.findall(r"\S+\s*", answer)

    return pieces or [answer]


class Handler(BaseHTTPRequestHandler):
    """The two endpoints the brain uses, and nothing else."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        """Keep the agent's own output readable; this server has nothing worth logging."""

    def do_GET(self) -> None:
        """Answer /v1/models, which is what warmup checks before the first turn."""
        if not self.path.endswith("/models"):
            self.send_error(404)

            return

        body = json.dumps({"data": [{"id": STUB_MODEL, "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """Answer /v1/chat/completions as a server-sent event stream."""
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)

            return

        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        prompt = "\n".join(message.get("content", "") for message in request.get("messages", []))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        time.sleep(FIRST_DELTA_SECONDS)

        for piece in deltas_of(answer_from_prompt(prompt)):
            self._write_event({"choices": [{"delta": {"content": piece}, "index": 0}]})
            time.sleep(INTER_DELTA_SECONDS)

        self._write_event({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]})
        self._write_raw(b"data: [DONE]\n\n")
        self._write_raw(b"")

    def _write_event(self, payload: dict) -> None:
        """Write one server-sent event."""
        self._write_raw(b"data: " + json.dumps(payload).encode() + b"\n\n")

    def _write_raw(self, data: bytes) -> None:
        """Write one chunk of the chunked response."""
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


def main() -> int:
    """Serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    arguments = parser.parse_args()

    server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
    print(
        f"stub LLM server on http://{arguments.host}:{arguments.port}/v1 "
        f"serving {STUB_MODEL} — NOT A MEASUREMENT",
        flush=True,
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
