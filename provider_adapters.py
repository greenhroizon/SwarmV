"""Dependency-free adapters for supported chat provider APIs.

The module deliberately keeps provider credentials in memory only.  API keys are
sent in request headers and are never included in exception messages, catalog
entries, or response objects.  ``urllib.request.urlopen`` is injectable so the
adapters can be tested with an in-process HTTP server or a small fake opener.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import socket
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ALLOW_INSECURE_HTTP_ENV = "SWARM_SIGNAL_ALLOW_INSECURE_HTTP"
DEFAULT_TIMEOUT = 20.0
MIN_TIMEOUT = 0.1
MAX_TIMEOUT = 60.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_TOKENS = 512
MAX_MAX_TOKENS = 8_192


class ProviderError(Exception):
    """Base class for safe, user-facing provider failures."""

    code = "provider_error"

    def __init__(self, message: str = "the provider request could not be completed") -> None:
        # Callers should only pass fixed, credential-free messages here.
        super().__init__(message)
        self.message = message


class ProviderValidationError(ProviderError):
    """A provider configuration or chat input could not be validated."""

    code = "invalid_provider"

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class ProviderTimeoutError(ProviderError):
    """The bounded provider request exceeded its timeout."""

    code = "provider_timeout"


class ProviderTransportError(ProviderError):
    """The provider could not be reached or returned an HTTP failure."""

    code = "provider_unavailable"


class ProviderResponseError(ProviderError):
    """The provider returned a response outside its documented shape."""

    code = "provider_response_error"


# These are the only values exposed by the catalog.  In particular, no key,
# authorization header, or user-supplied URL is ever part of this structure.
PROVIDER_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "openai",
        "name": "OpenAI",
        "protocol": "openai-chat-completions",
        "default_base_url": "https://api.openai.com/v1",
        "supports_custom_base_url": True,
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "protocol": "anthropic-messages",
        "default_base_url": "https://api.anthropic.com",
        "supports_custom_base_url": True,
    },
    {
        "id": "google_gemini",
        "name": "Google Gemini",
        "protocol": "gemini-generate-content",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "supports_custom_base_url": True,
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "protocol": "openai-chat-completions",
        "default_base_url": "https://openrouter.ai/api/v1",
        "supports_custom_base_url": True,
    },
    {
        "id": "custom_openai_compatible",
        "name": "Custom OpenAI-compatible endpoint",
        "protocol": "openai-chat-completions",
        "default_base_url": None,
        "supports_custom_base_url": True,
    },
)


_PROVIDER_DEFAULTS = {entry["id"]: entry["default_base_url"] for entry in PROVIDER_CATALOG}
_PROVIDER_ALIASES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google_gemini",
    "google": "google_gemini",
    "google_gemini": "google_gemini",
    "google_gemini_api": "google_gemini",
    "google_generative_ai": "google_gemini",
    "openrouter": "openrouter",
    "custom": "custom_openai_compatible",
    "custom_openai": "custom_openai_compatible",
    "custom_openai_compatible": "custom_openai_compatible",
    "custom_openai_compatible_endpoint": "custom_openai_compatible",
    "openai_compatible": "custom_openai_compatible",
}


def provider_catalog() -> list[dict[str, Any]]:
    """Return a copy of the public provider catalog, without any secrets."""

    return [
        {key: list(value) if isinstance(value, tuple) else value for key, value in entry.items()}
        for entry in PROVIDER_CATALOG
    ]


# Descriptive aliases make the small public module convenient for callers that
# use either noun in their integration code.
get_provider_catalog = provider_catalog


@dataclass(frozen=True)
class ProviderConfig:
    """Validated in-memory provider configuration.

    ``repr=False`` is intentional: accidental debug logging must not expose an
    API key.  The server never serializes this object.
    """

    provider_id: str
    model: str
    api_key: str = field(repr=False)
    base_url: str
    timeout: float = DEFAULT_TIMEOUT


def _mapping_value(
    payload: Mapping[str, Any],
    canonical: str,
    aliases: Sequence[str] = (),
    *,
    required: bool = False,
) -> Any:
    names = (canonical, *aliases)
    present = [(name, payload[name]) for name in names if name in payload]
    if not present:
        if required:
            raise ProviderValidationError(f"{canonical} is required", canonical)
        return None
    first_name, first_value = present[0]
    for name, value in present[1:]:
        if value != first_value:
            raise ProviderValidationError(
                f"provide only one value for {canonical} (conflicting fields)", canonical
            )
    return first_value


def _clean_string(value: Any, field_name: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ProviderValidationError(f"{field_name} must be a string", field_name)
    cleaned = value.strip()
    if not cleaned:
        raise ProviderValidationError(f"{field_name} must not be empty", field_name)
    if len(cleaned) > max_length:
        raise ProviderValidationError(
            f"{field_name} must be at most {max_length} characters", field_name
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in cleaned):
        raise ProviderValidationError(f"{field_name} contains unsupported control characters", field_name)
    return cleaned


def _normalize_provider_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ProviderValidationError("provider_id must be a string", "provider_id")
    key = value.strip().casefold().replace("-", "_").replace(" ", "_")
    provider_id = _PROVIDER_ALIASES.get(key)
    if provider_id is None:
        raise ProviderValidationError("provider_id is not supported", "provider_id")
    return provider_id


def _allow_insecure_http() -> bool:
    value = os.environ.get(ALLOW_INSECURE_HTTP_ENV, "")
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _is_loopback_or_test_host(hostname: str) -> bool:
    host = hostname.casefold().rstrip(".")
    if host in {"localhost", "localhost.localdomain", "test"} or host.endswith(".test"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_base_url(value: Any, provider_id: str) -> str:
    base_url = _clean_string(value, "base_url", max_length=2_048)
    if any(character.isspace() for character in base_url):
        raise ProviderValidationError("base_url must not contain whitespace", "base_url")
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        # Do not include parser details: malformed URLs can contain secrets in
        # userinfo, and all callers need is a stable safe validation message.
        raise ProviderValidationError("base_url is not a valid URL", "base_url") from exc
    if parsed.scheme.casefold() not in {"https", "http"}:
        raise ProviderValidationError("base_url must use HTTPS", "base_url")
    if not hostname:
        raise ProviderValidationError("base_url must include a hostname", "base_url")
    if parsed.username is not None or parsed.password is not None:
        raise ProviderValidationError("base_url must not include credentials", "base_url")
    if parsed.query or parsed.fragment:
        raise ProviderValidationError("base_url must not include a query or fragment", "base_url")
    if port is not None and not 1 <= port <= 65_535:
        raise ProviderValidationError("base_url port is out of range", "base_url")

    scheme = parsed.scheme.casefold()
    if scheme == "http":
        if not _allow_insecure_http() or not _is_loopback_or_test_host(hostname):
            raise ProviderValidationError(
                "base_url must use HTTPS; HTTP is allowed only for loopback/test endpoints "
                f"when {ALLOW_INSECURE_HTTP_ENV}=1",
                "base_url",
            )

    # Reject literal private, link-local, multicast, reserved, or unspecified
    # destinations.  Loopback is deliberately retained for local test servers;
    # HTTP still requires the explicit environment flag above.
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_loopback:
        if (
            address.is_private
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            raise ProviderValidationError("base_url points to an unsafe network address", "base_url")

    # A root slash is not meaningful to endpoint joining and can otherwise
    # produce double slashes in request paths.
    normalized = base_url.rstrip("/")
    if not normalized:
        raise ProviderValidationError("base_url is not a valid URL", "base_url")
    return normalized


def _validate_model(value: Any) -> str:
    return _clean_string(value, "model", max_length=256)


def _validate_api_key(value: Any) -> str:
    # Keep this validation intentionally format-agnostic: providers issue keys
    # with different alphabets and lengths.  It is never interpolated into an
    # error message.
    return _clean_string(value, "api_key", max_length=4_096)


def _validate_timeout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderValidationError("timeout must be a number of seconds", "timeout")
    timeout = float(value)
    if not math.isfinite(timeout) or not MIN_TIMEOUT <= timeout <= MAX_TIMEOUT:
        raise ProviderValidationError(
            f"timeout must be between {MIN_TIMEOUT:g} and {MAX_TIMEOUT:g} seconds", "timeout"
        )
    return timeout


def validate_provider_config(payload: Mapping[str, Any] | ProviderConfig) -> ProviderConfig:
    """Validate and normalize a provider configuration without making a call."""

    if isinstance(payload, ProviderConfig):
        # Revalidation also protects callers that construct the dataclass
        # directly and keeps URL policy consistent across entry points.
        return validate_provider_config(
            {
                "provider_id": payload.provider_id,
                "model": payload.model,
                "api_key": payload.api_key,
                "base_url": payload.base_url,
                "timeout": payload.timeout,
            }
        )
    if not isinstance(payload, Mapping):
        raise ProviderValidationError("provider config must be a JSON object")

    provider_value = _mapping_value(payload, "provider_id", ("provider", "id"), required=True)
    provider_id = _normalize_provider_id(provider_value)
    model = _validate_model(_mapping_value(payload, "model", required=True))
    api_key = _validate_api_key(_mapping_value(payload, "api_key", ("apiKey",), required=True))
    timeout_value = _mapping_value(payload, "timeout", ("timeout_seconds",), required=False)
    timeout = DEFAULT_TIMEOUT if timeout_value is None else _validate_timeout(timeout_value)

    base_value = _mapping_value(payload, "base_url", ("baseUrl",), required=False)
    if base_value is None:
        base_value = _PROVIDER_DEFAULTS[provider_id]
    if base_value is None:
        raise ProviderValidationError(
            "base_url is required for the custom OpenAI-compatible provider", "base_url"
        )
    base_url = _validate_base_url(base_value, provider_id)
    return ProviderConfig(
        provider_id=provider_id,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


def _join_endpoint(base_url: str, path: str) -> str:
    normalized_base = base_url.rstrip("/")
    normalized_path = path.lstrip("/")
    # Anthropic's documented base is the host root, while many compatible
    # test/proxy URLs already end in /v1.  Avoid producing /v1/v1/messages.
    if normalized_path.startswith("v1/") and normalized_base.casefold().endswith("/v1"):
        normalized_path = normalized_path.removeprefix("v1/")
    return f"{normalized_base}/{normalized_path}"


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


def _content_to_text(content: Any) -> str:
    """Extract text from common provider content block representations."""

    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes, bytearray)) or not isinstance(messages, Sequence):
        raise ProviderValidationError("messages must be a non-empty list", "messages")
    if not messages:
        raise ProviderValidationError("messages must be a non-empty list", "messages")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ProviderValidationError(f"messages[{index}] must be an object", "messages")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ProviderValidationError(
                f"messages[{index}].role is unsupported", "messages"
            )
        content = _content_to_text(message.get("content"))
        if not content.strip():
            raise ProviderValidationError(
                f"messages[{index}].content must contain text", "messages"
            )
        if len(content) > 50_000:
            raise ProviderValidationError(
                f"messages[{index}].content is too long", "messages"
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _validate_generation_options(max_tokens: Any, temperature: Any) -> tuple[int, float | None]:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ProviderValidationError("max_tokens must be an integer", "max_tokens")
    if not 1 <= max_tokens <= MAX_MAX_TOKENS:
        raise ProviderValidationError("max_tokens is out of range", "max_tokens")
    if temperature is None:
        return max_tokens, None
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ProviderValidationError("temperature must be a number", "temperature")
    normalized_temperature = float(temperature)
    if not math.isfinite(normalized_temperature) or not 0.0 <= normalized_temperature <= 2.0:
        raise ProviderValidationError("temperature must be between 0 and 2", "temperature")
    return max_tokens, normalized_temperature


Transport = Callable[[Request, float], Any]


class ProviderAdapter:
    """Base adapter implementing safe HTTP and JSON handling."""

    def __init__(
        self,
        config: ProviderConfig | Mapping[str, Any],
        opener: Transport | None = None,
        *,
        transport: Transport | None = None,
    ) -> None:
        if opener is not None and transport is not None:
            raise ProviderValidationError("provide only one HTTP transport", "transport")
        self.config = validate_provider_config(config)
        self._opener = opener or transport or _default_opener

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
    ) -> str:
        normalized_messages = _normalize_messages(messages)
        checked_max_tokens, checked_temperature = _validate_generation_options(
            max_tokens, temperature
        )
        payload = self._build_payload(normalized_messages, checked_max_tokens, checked_temperature)
        response = self._post_json(self._endpoint_url(), payload, self._headers())
        text = self._extract_text(response)
        if not text.strip():
            raise ProviderResponseError("provider returned no text")
        return text.strip()

    def _endpoint_url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _build_payload(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float | None
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _extract_text(self, response: Any) -> str:
        raise NotImplementedError

    def _post_json(self, endpoint: str, payload: Mapping[str, Any], headers: Mapping[str, str]) -> Any:
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProviderValidationError("request payload could not be encoded") from exc
        request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        request_headers.update(headers)
        request = Request(endpoint, data=encoded, headers=request_headers, method="POST")

        try:
            response = self._opener(request, self.config.timeout)
        except HTTPError as exc:
            try:
                exc.close()
            except OSError:
                pass
            raise ProviderTransportError("provider returned an HTTP error") from None
        except (socket.timeout, TimeoutError):
            raise ProviderTimeoutError() from None
        except URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise ProviderTimeoutError() from None
            raise ProviderTransportError("provider request failed") from None
        except Exception:
            # The broad final branch keeps implementation-specific transport
            # details (which may contain a URL) out of the public error.
            raise ProviderTransportError("provider request failed") from None

        try:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if status is not None and int(status) >= 400:
                raise ProviderTransportError("provider returned an HTTP error")
            try:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            except TypeError:
                # A tiny injected fake may expose read() without a size
                # parameter.  Real urllib responses take the bounded form.
                raw = response.read()
        except ProviderError:
            raise
        except (socket.timeout, TimeoutError):
            raise ProviderTimeoutError() from None
        except Exception:
            raise ProviderTransportError("provider response could not be read") from None
        finally:
            try:
                response.close()
            except (AttributeError, OSError):
                pass

        if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_RESPONSE_BYTES:
            raise ProviderResponseError("provider response was too large")
        try:
            text = bytes(raw).decode("utf-8")
            return json.loads(text, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            raise ProviderResponseError("provider returned invalid JSON") from None


def _default_opener(request: Request, timeout: float) -> Any:
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep a provider from redirecting a validated URL to an unsafe host."""

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str):
        return None


_NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler)


class OpenAICompatibleAdapter(ProviderAdapter):
    """Adapter for OpenAI, OpenRouter, and custom compatible endpoints."""

    def _endpoint_url(self) -> str:
        return _join_endpoint(self.config.base_url, "chat/completions")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}

    def _build_payload(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    def _extract_text(self, response: Any) -> str:
        if not isinstance(response, Mapping):
            raise ProviderResponseError("provider returned an unexpected response")
        choices = response.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise ProviderResponseError("provider returned an unexpected response")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ProviderResponseError("provider returned an unexpected response")
        message = first.get("message")
        if isinstance(message, Mapping):
            text = _content_to_text(message.get("content"))
        else:
            text = _content_to_text(first.get("text"))
        if not text:
            raise ProviderResponseError("provider returned an unexpected response")
        return text


class OpenAIAdapter(OpenAICompatibleAdapter):
    """OpenAI chat-completions adapter."""


class OpenRouterAdapter(OpenAICompatibleAdapter):
    """OpenRouter's OpenAI-compatible chat-completions adapter."""


class CustomOpenAICompatibleAdapter(OpenAICompatibleAdapter):
    """Adapter for a user-supplied OpenAI-compatible endpoint."""


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic's ``/v1/messages`` API."""

    def _endpoint_url(self) -> str:
        return _join_endpoint(self.config.base_url, "v1/messages")

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _build_payload(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float | None
    ) -> dict[str, Any]:
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        conversation = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] != "system"
        ]
        if not conversation:
            raise ProviderValidationError("messages must contain a user or assistant message", "messages")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": conversation,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    def _extract_text(self, response: Any) -> str:
        if not isinstance(response, Mapping):
            raise ProviderResponseError("provider returned an unexpected response")
        text = _content_to_text(response.get("content"))
        if not text:
            raise ProviderResponseError("provider returned an unexpected response")
        return text


class GoogleGeminiAdapter(ProviderAdapter):
    """Adapter for Google's Gemini ``generateContent`` API."""

    def _endpoint_url(self) -> str:
        model = self.config.model.removeprefix("models/")
        return _join_endpoint(
            self.config.base_url,
            f"models/{quote(model, safe='')}:generateContent",
        )

    def _headers(self) -> dict[str, str]:
        # Using a header rather than a query parameter keeps the credential out
        # of URLs, proxy access logs, and exception representations.
        return {"x-goog-api-key": self.config.api_key}

    def _build_payload(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float | None
    ) -> dict[str, Any]:
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        contents: list[dict[str, Any]] = []
        for message in messages:
            if message["role"] == "system":
                continue
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message["content"]}]})
        if not contents:
            raise ProviderValidationError("messages must contain a user or assistant message", "messages")
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        return payload

    def _extract_text(self, response: Any) -> str:
        if not isinstance(response, Mapping):
            raise ProviderResponseError("provider returned an unexpected response")
        candidates = response.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
            raise ProviderResponseError("provider returned an unexpected response")
        first = candidates[0]
        if not isinstance(first, Mapping):
            raise ProviderResponseError("provider returned an unexpected response")
        content = first.get("content")
        if isinstance(content, Mapping):
            text = _content_to_text(content.get("parts"))
        else:
            text = _content_to_text(first.get("text"))
        if not text:
            raise ProviderResponseError("provider returned an unexpected response")
        return text


GeminiAdapter = GoogleGeminiAdapter
CustomOpenAIAdapter = CustomOpenAICompatibleAdapter


def create_adapter(
    config: ProviderConfig | Mapping[str, Any],
    opener: Transport | None = None,
    *,
    transport: Transport | None = None,
) -> ProviderAdapter:
    """Validate configuration and construct the matching provider adapter."""

    if opener is not None and transport is not None:
        raise ProviderValidationError("provide only one HTTP transport", "transport")
    checked = validate_provider_config(config)
    adapter_type: type[ProviderAdapter]
    if checked.provider_id == "openai":
        adapter_type = OpenAIAdapter
    elif checked.provider_id == "anthropic":
        adapter_type = AnthropicAdapter
    elif checked.provider_id == "google_gemini":
        adapter_type = GoogleGeminiAdapter
    elif checked.provider_id == "openrouter":
        adapter_type = OpenRouterAdapter
    else:
        adapter_type = CustomOpenAICompatibleAdapter
    # Pass the already checked dataclass; the adapter's constructor performs a
    # second policy check but never serializes or logs the key.
    return adapter_type(checked, opener=opener, transport=transport)


build_adapter = create_adapter


def provider_summary(config: ProviderConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return the safe provider fields suitable for an API response."""

    checked = validate_provider_config(config)
    return {
        "provider_id": checked.provider_id,
        "model": checked.model,
    }


__all__ = [
    "ALLOW_INSECURE_HTTP_ENV",
    "AnthropicAdapter",
    "CustomOpenAICompatibleAdapter",
    "CustomOpenAIAdapter",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT",
    "GoogleGeminiAdapter",
    "GeminiAdapter",
    "MAX_MAX_TOKENS",
    "MAX_RESPONSE_BYTES",
    "MAX_TIMEOUT",
    "MIN_TIMEOUT",
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "OpenRouterAdapter",
    "PROVIDER_CATALOG",
    "ProviderAdapter",
    "ProviderConfig",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderTransportError",
    "ProviderValidationError",
    "build_adapter",
    "create_adapter",
    "get_provider_catalog",
    "provider_catalog",
    "provider_summary",
    "validate_provider_config",
]
