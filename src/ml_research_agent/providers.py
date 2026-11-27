"""Talk to a language model. The agent needs one thing from it: a reply to a list of
chat messages, and how many tokens that cost. Tests use ``ScriptedProvider``; real runs
use an OpenAI-compatible chat endpoint configured from the environment, so the same code
works with a hosted API or a local server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class Reply:
    text: str
    # Zero when the server does not report usage.
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(RuntimeError):
    """The model could not be reached or answered in a shape we cannot use."""


class Provider(Protocol):
    def complete(self, messages: Sequence[Message]) -> Reply: ...


class ScriptedProvider:
    """Returns canned replies in order and keeps every request it was given."""

    def __init__(self, replies: Iterable[str | Reply]) -> None:
        self._replies: deque[Reply] = deque(
            r if isinstance(r, Reply) else Reply(r) for r in replies
        )
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> Reply:
        self.calls.append([dict(m) for m in messages])  # type: ignore[misc]
        if not self._replies:
            raise ProviderError("scripted provider has no reply left")
        return self._replies.popleft()


ENV_BASE_URL = "MLAGENT_BASE_URL"
ENV_MODEL = "MLAGENT_MODEL"
ENV_API_KEY = "MLAGENT_API_KEY"


@dataclass(frozen=True)
class HTTPConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        scheme = urllib.parse.urlsplit(self.base_url).scheme
        if scheme not in ("http", "https"):
            raise ProviderError(f"{ENV_BASE_URL} must start with http:// or https://")

    def __repr__(self) -> str:  # the key must never end up in a log or a traceback
        key = "set" if self.api_key else "not set"
        return f"HTTPConfig(base_url={self.base_url!r}, model={self.model!r}, api_key={key})"


def config_from_env(env: Mapping[str, str]) -> HTTPConfig:
    """Read the endpoint, model and optional key. Local servers need no key."""
    missing = [name for name in (ENV_BASE_URL, ENV_MODEL) if not env.get(name)]
    if missing:
        raise ProviderError("missing environment variables: " + ", ".join(missing))
    return HTTPConfig(
        base_url=env[ENV_BASE_URL],
        model=env[ENV_MODEL],
        api_key=env.get(ENV_API_KEY) or None,
    )


# The transport sends one POST and returns the response body. Injected so tests never
# touch the network.
Transport = Callable[[str, dict[str, str], bytes, float], bytes]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data: bytes = response.read()
        return data


class OpenAICompatibleProvider:
    """A client for any server that implements ``POST /chat/completions``."""

    def __init__(self, config: HTTPConfig, transport: Transport = _urllib_transport) -> None:
        self.config = config
        self._transport = transport

    def complete(self, messages: Sequence[Message]) -> Reply:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        body = json.dumps(
            {"model": self.config.model, "messages": list(messages), "temperature": 0}
        ).encode("utf-8")
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        try:
            raw = self._transport(url, headers, body, self.config.timeout_s)
        except urllib.error.HTTPError as err:
            raise ProviderError(f"model endpoint answered HTTP {err.code}") from err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise ProviderError(f"model endpoint could not be reached: {err}") from err
        return _reply(raw)


def _count(usage: Any, name: str) -> int:
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _reply(raw: bytes) -> Reply:
    try:
        data: Any = json.loads(raw)
        text = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as err:
        raise ProviderError("model response has no message content") from err
    if not isinstance(text, str):
        raise ProviderError("model response content is not text")
    usage = data.get("usage")
    return Reply(text, _count(usage, "prompt_tokens"), _count(usage, "completion_tokens"))
