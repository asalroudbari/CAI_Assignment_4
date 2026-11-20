"""Streaming chat over the OpenAI-compatible API that vLLM serves. Owned by Partner B.

This is the standard library and nothing else: ``http.client`` and ``json``. The ``openai`` package
would do the same work, and adding it would mean another pinned version in ``recipe/requirements.txt``
and another import to pay for at start-up, for a client that sends one POST and reads lines off a
socket. The server speaks the OpenAI protocol, not the OpenAI library's protocol.

Two things here are about latency rather than about tidiness:

* The connection is kept open between turns. vLLM is on 127.0.0.1, so a fresh connect is cheap, but
  it is not free and it happens on the critical path of every single turn, between the final
  transcript and the first token.
* The response is read line by line and each delta is yielded the moment it is parsed. Nothing is
  accumulated here, because the sentence that stops the clock is assembled one delta up, in
  ``brain.py``, and anything held back in this file is time added to ``llm_first_token``.

Reasoning models (gpt-oss-20b) send their thinking as ``reasoning_content`` deltas before the
answer. Those are never spoken, so they are yielded with ``is_reasoning`` set and the brain stamps
``first_reasoning_token_nanoseconds`` from them while it waits for the answer to start.
"""

from __future__ import annotations

import http.client
import json
import os
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

BASE_URL_ENVIRONMENT_VARIABLE = "LLM_BASE_URL"
MODEL_ENVIRONMENT_VARIABLE = "LLM_MODEL"
API_KEY_ENVIRONMENT_VARIABLE = "LLM_API_KEY"

# vLLM ignores the key unless it was started with one, but the header has to be there.
DEFAULT_API_KEY = "EMPTY"

# How much of a server error to put in the exception. Enough to see the cause, not enough to fill
# the job log with a stack trace per turn.
ERROR_BODY_CHARACTERS = 400


class LlmError(RuntimeError):
    """The LLM server could not be reached, or answered with something unusable."""


@dataclass(frozen=True)
class Delta:
    """One piece of text from the stream, and whether it is thinking rather than answer."""

    text: str
    is_reasoning: bool = False


def default_base_url() -> str:
    """Return the server URL from the environment, or the local default.

    The README is explicit that the server URL is a setting and never a constant in the code. It
    arrives either here, from the environment the sbatch script exports, or on the command line.
    """
    return os.environ.get(BASE_URL_ENVIRONMENT_VARIABLE, DEFAULT_BASE_URL)


def default_model() -> str | None:
    """Return the model name from the environment, if the job set one."""
    return os.environ.get(MODEL_ENVIRONMENT_VARIABLE) or None


class ChatClient:
    """One keep-alive connection to an OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self.api_key = api_key or os.environ.get(API_KEY_ENVIRONMENT_VARIABLE, DEFAULT_API_KEY)
        self.timeout_seconds = timeout_seconds

        parts = urlsplit(self.base_url)

        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise LlmError(f"{self.base_url!r} is not a usable server URL.")

        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.prefix = parts.path.rstrip("/")
        self._connection: http.client.HTTPConnection | None = None

    def describe(self) -> str:
        """Return the server this client talks to, for the banner."""
        return f"{self.scheme}://{self.host}:{self.port}{self.prefix}"

    # -- the connection -------------------------------------------------------------------------

    def _connect(self) -> http.client.HTTPConnection:
        """Return the open connection, opening one if there is none."""
        if self._connection is None:
            connection_class = (
                http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            )
            self._connection = connection_class(self.host, self.port, timeout=self.timeout_seconds)

        return self._connection

    def close(self) -> None:
        """Drop the connection. The next request opens a new one."""
        if self._connection is not None:
            try:
                self._connection.close()

            finally:
                self._connection = None

    def _send(self, method: str, path: str, body: dict | None, stream: bool):
        """Send one request and return the response, reconnecting once if the socket was stale.

        A kept-open connection can be closed by the server between turns. That is not an error
        worth failing a turn over, so one reconnect is attempted; a second failure is real.
        """
        payload = json.dumps(body).encode() if body is not None else None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

        for attempt in (1, 2):
            connection = self._connect()

            try:
                connection.request(method, f"{self.prefix}{path}", body=payload, headers=headers)

                return connection.getresponse()

            except (http.client.HTTPException, OSError) as error:
                self.close()

                if attempt == 2:
                    raise LlmError(
                        f"cannot reach the LLM server at {self.describe()}: {error}"
                    ) from error

        raise LlmError("unreachable")

    def _fail(self, response) -> None:
        """Turn a non-200 response into an error that says what the server actually said."""
        body = response.read().decode("utf-8", "replace")[:ERROR_BODY_CHARACTERS]
        self.close()

        raise LlmError(f"the LLM server answered {response.status}: {body}")

    # -- the two calls --------------------------------------------------------------------------

    def models(self) -> list[str]:
        """Return the model names the server is serving."""
        response = self._send("GET", "/models", None, stream=False)

        if response.status != 200:
            self._fail(response)

        body = json.loads(response.read())

        return [entry.get("id", "") for entry in body.get("data", [])]

    def stream_chat(self, messages: list[dict], **options: Any) -> Iterator[Delta]:
        """Yield the answer to ``messages`` delta by delta, as the server produces them."""
        body = {"messages": messages, "stream": True, **options}
        response = self._send("POST", "/chat/completions", body, stream=True)

        if response.status != 200:
            self._fail(response)

        finished = False

        try:
            for raw_line in response:
                line = raw_line.strip()

                # Server-sent events: blank lines separate records, ":" starts a comment.
                if not line or line.startswith(b":"):
                    continue

                if not line.startswith(b"data:"):
                    continue

                data = line[len(b"data:"):].strip()

                if data == b"[DONE]":
                    finished = True
                    break

                try:
                    chunk = json.loads(data)

                except json.JSONDecodeError as error:
                    raise LlmError(f"the LLM server sent a line that is not JSON: {data!r}") from error

                for choice in chunk.get("choices", ()):
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")

                    if reasoning:
                        yield Delta(reasoning, is_reasoning=True)

                    content = delta.get("content")

                    if content:
                        yield Delta(content)

        finally:
            # A connection is only reusable when its response was read to the end. Anything else —
            # an error, a caller that stopped early — leaves bytes on the socket that the next turn
            # would read as its own answer.
            if finished:
                response.read()
            else:
                self.close()
