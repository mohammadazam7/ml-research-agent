import io
import json
import urllib.error

import pytest

from ml_research_agent.providers import (
    HTTPConfig,
    Message,
    OpenAICompatibleProvider,
    ProviderError,
    Reply,
    ScriptedProvider,
    config_from_env,
)

HELLO: list[Message] = [{"role": "user", "content": "hello"}]


def _body(content: object, usage: object = None) -> bytes:
    payload: dict[str, object] = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload).encode()


def test_scripted_provider_replies_in_order_and_keeps_what_it_was_asked() -> None:
    provider = ScriptedProvider(["one", Reply("two", 10, 3)])
    assert provider.complete(HELLO) == Reply("one")
    assert provider.complete([*HELLO, {"role": "user", "content": "again"}]) == Reply("two", 10, 3)
    assert [len(call) for call in provider.calls] == [1, 2]
    with pytest.raises(ProviderError, match="no reply left"):
        provider.complete(HELLO)


def test_a_recorded_call_does_not_change_when_the_caller_reuses_its_list() -> None:
    provider = ScriptedProvider(["one"])
    messages = list(HELLO)
    provider.complete(messages)
    messages.append({"role": "user", "content": "later"})
    assert provider.calls == [HELLO]


def test_the_request_carries_the_model_messages_and_key() -> None:
    seen: dict[str, object] = {}

    def transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
        seen.update(url=url, headers=headers, body=json.loads(body), timeout=timeout)
        return _body("hi", {"prompt_tokens": 12, "completion_tokens": 4})

    config = HTTPConfig("https://models.example/v1/", "small", api_key="k-123", timeout_s=5)
    reply = OpenAICompatibleProvider(config, transport).complete(HELLO)
    assert reply == Reply("hi", 12, 4)
    assert seen["url"] == "https://models.example/v1/chat/completions"
    assert seen["headers"] == {"Content-Type": "application/json", "Authorization": "Bearer k-123"}
    assert seen["body"] == {"model": "small", "messages": HELLO, "temperature": 0}
    assert seen["timeout"] == 5


def test_a_local_server_needs_no_key_and_may_omit_usage() -> None:
    def transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
        assert "Authorization" not in headers
        return _body("hi")

    provider = OpenAICompatibleProvider(HTTPConfig("http://localhost:8080/v1", "tiny"), transport)
    assert provider.complete(HELLO) == Reply("hi", 0, 0)


@pytest.mark.parametrize("usage", [{"prompt_tokens": "many"}, {"prompt_tokens": -4}, [1, 2], True])
def test_usage_that_is_not_a_count_is_read_as_zero(usage: object) -> None:
    provider = OpenAICompatibleProvider(
        HTTPConfig("http://localhost/v1", "tiny"), lambda *_: _body("hi", usage)
    )
    assert provider.complete(HELLO) == Reply("hi", 0, 0)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"not json", "no message content"),
        (b'{"choices": []}', "no message content"),
        (b'{"error": "overloaded"}', "no message content"),
        (_body(None), "not text"),
        (_body(["a"]), "not text"),
    ],
)
def test_an_unusable_answer_is_an_error(raw: bytes, message: str) -> None:
    provider = OpenAICompatibleProvider(HTTPConfig("http://localhost/v1", "tiny"), lambda *_: raw)
    with pytest.raises(ProviderError, match=message):
        provider.complete(HELLO)


def test_network_failures_become_provider_errors_without_the_key() -> None:
    def refused(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
        raise urllib.error.URLError("connection refused")

    def unauthorised(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, io.BytesIO(b""))  # type: ignore[arg-type]

    config = HTTPConfig("https://models.example/v1", "small", api_key="k-secret")
    with pytest.raises(ProviderError, match="could not be reached") as reached:
        OpenAICompatibleProvider(config, refused).complete(HELLO)
    with pytest.raises(ProviderError, match="HTTP 401") as status:
        OpenAICompatibleProvider(config, unauthorised).complete(HELLO)
    assert "k-secret" not in str(reached.value) + str(status.value) + repr(config)
    assert "api_key=set" in repr(config)


def test_configuration_comes_from_the_environment() -> None:
    env = {"MLAGENT_BASE_URL": "http://localhost:11434/v1", "MLAGENT_MODEL": "local"}
    assert config_from_env(env) == HTTPConfig("http://localhost:11434/v1", "local")
    assert config_from_env({**env, "MLAGENT_API_KEY": "k"}).api_key == "k"
    with pytest.raises(ProviderError, match="MLAGENT_BASE_URL, MLAGENT_MODEL"):
        config_from_env({})
    with pytest.raises(ProviderError, match="http:// or https://"):
        config_from_env({**env, "MLAGENT_BASE_URL": "file:///etc/passwd"})
