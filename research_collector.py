"""Bounded, dependency-free collection of public web evidence.

The collector intentionally has a narrow contract.  A statement-driven request
may use one fixed public search endpoint to discover a small result set before
making one request per source URL without following redirects.  It returns only
bounded, readable text; it does not crawl links, execute page content, resolve
arbitrary URLs supplied by a page, or claim that the returned packet represents
all available information.

``urllib`` is used for transport and URL parsing, ``HTMLParser`` extracts
visible text, and ``json`` is used only by callers that serialize the returned
packet.  The opener/transport parameters make all fetch behavior replaceable
in offline tests.
"""

from __future__ import annotations

import datetime as _datetime
import ipaddress
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_URLS = 8
MAX_URL_COUNT = MAX_URLS
MAX_DISCOVERED_URLS = 5
MAX_DISCOVERY_QUERIES = 4
MAX_URL_LENGTH = 2_048
MAX_STATEMENT_LENGTH = 2_000
DEFAULT_TIMEOUT = 8.0
MIN_TIMEOUT = 0.1
MAX_TIMEOUT = 15.0
MAX_BODY_BYTES = 1 * 1024 * 1024
MAX_FETCH_BYTES = MAX_BODY_BYTES
MAX_TEXT_CHARS = 12_000
MAX_EXCERPT_CHARS = 1_200
MAX_TITLE_CHARS = 300
SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"

_TEXT_CONTENT_TYPES = {
    "application/json",
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}
_UNSAFE_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "metadata",
    "metadata.google.internal",
    "host.docker.internal",
}
_UNSAFE_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".localdomain",
    ".internal",
    ".intranet",
    ".lan",
    ".home.arpa",
    ".test",
    ".invalid",
)
_WHITESPACE_RE = re.compile(r"\s+")
_CHARSET_RE = re.compile(r"(?:^|;)\s*charset\s*=\s*['\"]?([^;\s'\"]+)", re.IGNORECASE)


class ResearchError(Exception):
    """Base class for safe research collection failures."""

    code = "research_error"

    def __init__(self, message: str, field: str | None = None) -> None:
        # Messages passed here are fixed policy text.  Never pass through an
        # exception string, URL, response body, or provider payload.
        super().__init__(message)
        self.message = message
        self.field = field


class ResearchValidationError(ResearchError):
    """The statement, URL list, or timeout violates the collection contract."""

    code = "invalid_research"


class ResearchFetchError(ResearchError):
    """A source could not be fetched or parsed."""

    code = "research_fetch_failed"


class ResearchTimeoutError(ResearchFetchError):
    """A source exceeded the bounded fetch timeout."""

    code = "research_timeout"


class _NoRedirectHandler(HTTPRedirectHandler):
    """Prevent urllib from following a Location header."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class _VisibleHTMLParser(HTMLParser):
    """Extract a title, canonical link, and visible text from HTML."""

    _ignored_tags = {"script", "style", "noscript", "template", "svg", "canvas"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.canonical_href: str | None = None
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.casefold()
        attrs_map = {key.casefold(): value for key, value in attrs}
        if name in self._ignored_tags:
            self._ignored_depth += 1
            return
        if name == "title" and self._ignored_depth == 0:
            self._in_title = True
        if name == "link" and self._ignored_depth == 0 and self.canonical_href is None:
            rel = (attrs_map.get("rel") or "").casefold().split()
            href = attrs_map.get("href")
            if "canonical" in rel and href:
                self.canonical_href = href.strip()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        name = tag.casefold()
        if name in self._ignored_tags:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if name == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = data.strip()
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)


class _SearchResultParser(HTMLParser):
    """Read result links from the fixed public search endpoint only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str]] = []
        self._href: str | None = None
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a" or self._href is not None:
            return
        attrs_map = {key.casefold(): value for key, value in attrs}
        classes = (attrs_map.get("class") or "").casefold().split()
        href = attrs_map.get("href")
        if "result__a" in classes and href:
            self._href = href.strip()
            self._title_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "a" or self._href is None:
            return
        title = _clean_text(" ".join(self._title_parts), MAX_TITLE_CHARS)
        if title:
            self.results.append((self._href, title))
        self._href = None
        self._title_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._title_parts.append(data.strip())

    def handle_entityref(self, name: str) -> None:
        # convert_charrefs=True normally handles these in data; retaining this
        # hook makes malformed/partial input harmless if the parser delegates.
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds")


def _clean_text(value: str, max_length: int) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()[:max_length].strip()


def _safe_hostname(hostname: str) -> str:
    # URL parsing returns a Unicode hostname.  IDNA conversion makes the host
    # checked here match the ASCII host sent by urllib for international names.
    try:
        return hostname.encode("idna").decode("ascii").casefold().rstrip(".")
    except (UnicodeError, UnicodeEncodeError):
        raise ResearchValidationError("URL host is invalid", "urls") from None


def validate_url(value: Any, field: str = "urls") -> str:
    """Validate one fetch URL and return a trimmed, safe URL string."""

    if not isinstance(value, str):
        raise ResearchValidationError("each URL must be a string", field)
    url = value.strip()
    if not url:
        raise ResearchValidationError("URLs must not be empty", field)
    if len(url) > MAX_URL_LENGTH:
        raise ResearchValidationError(
            f"each URL must be at most {MAX_URL_LENGTH} characters", field
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        raise ResearchValidationError("URL contains unsupported control characters", field)

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        # Accessing .port also validates malformed bracketed/port syntax.
        port = parsed.port
    except ValueError:
        raise ResearchValidationError("URL is invalid", field) from None

    if parsed.scheme.casefold() != "https":
        raise ResearchValidationError("URLs must use HTTPS", field)
    if not hostname:
        raise ResearchValidationError("URL must include a host", field)
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ResearchValidationError("URL credentials are not allowed", field)
    if parsed.query or parsed.fragment or "?" in url or "#" in url:
        raise ResearchValidationError("URL queries and fragments are not allowed", field)
    if port == 0:
        raise ResearchValidationError("URL port is invalid", field)

    safe_host = _safe_hostname(hostname)
    if safe_host in _UNSAFE_HOSTS or safe_host.endswith(_UNSAFE_HOST_SUFFIXES):
        raise ResearchValidationError("URL host is not allowed", field)
    numeric_labels = safe_host.split(".")
    if safe_host.isdigit() or (
        len(numeric_labels) <= 4 and all(label.isdigit() for label in numeric_labels)
    ) or re.fullmatch(r"0[xX][0-9a-fA-F]+", safe_host):
        # Numeric dword/short IPv4 spellings can be interpreted as loopback or
        # private addresses by resolvers even when ipaddress does not parse
        # them as canonical dotted notation.
        raise ResearchValidationError("URL host is not allowed", field)
    if "%" in hostname:
        # IPv6 zone identifiers may select a link-local interface.
        raise ResearchValidationError("URL host is not allowed", field)
    try:
        address = ipaddress.ip_address(safe_host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        or not address.is_global
    ):
        raise ResearchValidationError("URL host is not allowed", field)

    # Preserve the caller's path and host spelling, while ensuring a stable
    # path exists for final-URL comparison.  Query/fragment were rejected.
    return urlunsplit(("https", parsed.netloc, parsed.path or "/", "", ""))


def validate_urls(urls: Any) -> list[str]:
    """Validate the bounded URL list used by :func:`collect_research`."""

    if isinstance(urls, (str, bytes)) or not isinstance(urls, Sequence):
        raise ResearchValidationError("urls must be an array of strings", "urls")
    if len(urls) > MAX_URLS:
        raise ResearchValidationError(f"urls may contain at most {MAX_URLS} items", "urls")
    return [validate_url(value, f"urls[{index}]") for index, value in enumerate(urls)]


def validate_statement(statement: Any) -> str:
    """Validate the forecasting statement used to frame the evidence packet."""

    if not isinstance(statement, str):
        raise ResearchValidationError("statement must be a string", "statement")
    cleaned = statement.strip()
    if not cleaned:
        raise ResearchValidationError("statement must not be empty", "statement")
    if len(cleaned) > MAX_STATEMENT_LENGTH:
        raise ResearchValidationError(
            f"statement must be at most {MAX_STATEMENT_LENGTH} characters", "statement"
        )
    if any(ord(character) < 32 and character not in "\n\t" or ord(character) == 127 for character in cleaned):
        raise ResearchValidationError("statement contains unsupported control characters", "statement")
    return cleaned


def validate_timeout(timeout: Any) -> float:
    """Validate and normalize the per-source timeout."""

    if timeout is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ResearchValidationError("timeout must be a number", "timeout")
    normalized = float(timeout)
    if not math.isfinite(normalized) or not MIN_TIMEOUT <= normalized <= MAX_TIMEOUT:
        raise ResearchValidationError(
            f"timeout must be between {MIN_TIMEOUT:g} and {MAX_TIMEOUT:g} seconds", "timeout"
        )
    return normalized


def validate_discover(value: Any) -> bool:
    """Validate the optional statement-driven public source discovery flag."""

    if value is None:
        # Research requests are statement-driven by default.  Callers that do
        # not want the fixed public search endpoint must opt out explicitly
        # with ``discover: false``.
        return True
    if not isinstance(value, bool):
        raise ResearchValidationError("discover must be a boolean", "discover")
    return value


def _header(response: Any, name: str) -> str:
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get(name)
        except AttributeError:
            value = None
        if not value and isinstance(headers, Mapping):
            wanted = name.casefold()
            for header_name, header_value in headers.items():
                if str(header_name).casefold() == wanted:
                    value = header_value
                    break
        if value:
            return str(value)
    getheader = getattr(response, "getheader", None)
    if callable(getheader):
        value = getheader(name)
        if value:
            return str(value)
    return ""


def _response_url(response: Any, requested_url: str) -> str:
    geturl = getattr(response, "geturl", None)
    if callable(geturl):
        value = geturl()
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = getattr(response, "url", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return requested_url


def _same_fetch_url(requested_url: str, response_url: str) -> bool:
    try:
        requested = urlsplit(requested_url)
        returned = urlsplit(response_url)
        returned_host = returned.hostname
        if returned_host is None:
            return False
        normalized_returned = validate_url(response_url)
        normalized_requested = validate_url(requested_url)
    except ResearchValidationError:
        return False
    return normalized_requested.casefold() == normalized_returned.casefold()


def _content_type(raw_content_type: str, body: bytes) -> str:
    media_type = raw_content_type.split(";", 1)[0].strip().casefold()
    if media_type:
        return media_type
    # A missing header is common in simple test transports.  Only infer a
    # harmless textual type from an obvious HTML/text prefix.
    prefix = body[:256].lstrip().casefold()
    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html") or b"<title" in prefix:
        return "text/html"
    return "text/plain"


def _decode_body(body: bytes, raw_content_type: str) -> str:
    match = _CHARSET_RE.search(raw_content_type)
    charset = match.group(1) if match else "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except (LookupError, TypeError):
        return body.decode("utf-8", errors="replace")


def _extract_document(body: bytes, content_type: str) -> tuple[str, str, str | None]:
    text = _decode_body(body, content_type)
    if content_type == "application/json":
        try:
            parsed = json.loads(text)
            readable = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError, ValueError):
            readable = text
        return "", _clean_text(readable, MAX_TEXT_CHARS), None
    if content_type in {"text/html", "application/xhtml+xml"}:
        parser = _VisibleHTMLParser()
        try:
            parser.feed(text)
            parser.close()
        except Exception:
            # HTMLParser should not make a source unavailable merely because a
            # public page is malformed.  Its partial output is still bounded.
            pass
        title = _clean_text(" ".join(parser.title_parts), MAX_TITLE_CHARS)
        readable = _clean_text(" ".join(parser.text_parts), MAX_TEXT_CHARS)
        return title, readable, parser.canonical_href
    readable = _clean_text(text, MAX_TEXT_CHARS)
    return "", readable, None


def _safe_canonical_url(canonical_href: str | None, base_url: str) -> str | None:
    if not canonical_href:
        return None
    try:
        candidate = urljoin(base_url, canonical_href.strip())
        return validate_url(candidate, "canonical_url")
    except ResearchValidationError:
        return None


def _error_source(
    source_id: str,
    url: str,
    fetched_at: str,
    *,
    message: str,
    code: str,
    http_status: int | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    # Keep all error text fixed and deliberately omit exception/URL details.
    return {
        "id": source_id,
        "url": url,
        "final_url": None,
        "canonical_url": None,
        "title": "",
        "text": "",
        "readable_text": "",
        "excerpt": "",
        "summary": "",
        "status": "error",
        "status_code": http_status,
        "http_status": http_status,
        "content_type": content_type,
        "fetched_at": fetched_at,
        "error": message,
        "error_code": code,
    }


def _success_source(
    source_id: str,
    url: str,
    response_url: str,
    status_code: int,
    content_type: str,
    title: str,
    readable: str,
    canonical_url: str | None,
    fetched_at: str,
) -> dict[str, Any]:
    excerpt = readable[:MAX_EXCERPT_CHARS].strip()
    return {
        "id": source_id,
        "url": url,
        "final_url": response_url,
        "canonical_url": canonical_url,
        "title": title,
        "text": readable,
        "readable_text": readable,
        "excerpt": excerpt,
        "summary": excerpt,
        "status": "ok",
        "status_code": status_code,
        "http_status": status_code,
        "content_type": content_type,
        "fetched_at": fetched_at,
        "error": None,
        "error_code": None,
    }


def _is_timeout_exception(error: BaseException) -> bool:
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, URLError):
        reason = getattr(error, "reason", None)
        return isinstance(reason, (TimeoutError,))
    return False


def _open_response(
    request: Request,
    timeout: float,
    *,
    opener: Any | None,
    transport: Callable[..., Any] | None,
) -> Any:
    def call_injected(callback: Callable[..., Any], value: Any) -> Any:
        # Small offline doubles commonly expose one of these shapes:
        # transport(Request, timeout=...), transport(Request, timeout),
        # transport(url, timeout=...), or transport(url).  Try only argument
        # mismatches; network/HTTP failures have explicit source handling.
        first_error: BaseException | None = None
        candidates = (value, request.full_url) if value is request else (value,)
        for candidate in candidates:
            for call in (
                lambda: callback(candidate, timeout=timeout),
                lambda: callback(candidate, timeout),
                lambda: callback(candidate),
            ):
                try:
                    return call()
                except (AssertionError, AttributeError, KeyError, TypeError) as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error
        raise ResearchFetchError("source could not be fetched")

    if transport is not None:
        # The documented injectable shape is transport(request, timeout).  A
        # keyword-friendly call is useful for small test doubles and does not
        # expose any response detail when the callable rejects it.
        return call_injected(transport, request)
    selected = opener if opener is not None else build_opener(_NoRedirectHandler)
    open_method = getattr(selected, "open", None)
    if callable(open_method):
        return open_method(request, timeout=timeout)
    if callable(selected):
        return call_injected(selected, request)
    raise ResearchFetchError("source could not be fetched")


def _search_endpoint_url(statement: str) -> str:
    return f"{SEARCH_ENDPOINT}?{urlencode({'q': statement, 'kl': 'wt-wt'})}"


def _discovery_query_variants(statement: str) -> list[str]:
    """Derive a small, deterministic set of public-search query variants."""

    base = _clean_text(statement, MAX_STATEMENT_LENGTH)
    candidates = (
        base,
        f"{base} evidence data",
        f"{base} risks criticism",
        f"{base} policy impact",
    )
    variants: list[str] = []
    seen: set[str] = set()
    for candidate in candidates[:MAX_DISCOVERY_QUERIES]:
        normalized = _clean_text(candidate, MAX_STATEMENT_LENGTH)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        variants.append(normalized)
    return variants


# Public descriptive aliases for offline callers that want to inspect the
# bounded discovery policy without depending on private names.
derive_query_variants = _discovery_query_variants
build_query_variants = _discovery_query_variants


def _safe_search_response_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold().rstrip(".") == "html.duckduckgo.com"
        and parsed.path.rstrip("/") == "/html"
        and parsed.username is None
        and parsed.password is None
    )


def _discovered_result_url(value: str) -> str | None:
    """Convert a search result link to a source URL without following it."""

    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.hostname and parsed.hostname.casefold().endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        candidate = target
    try:
        return validate_url(candidate, "discovered_urls")
    except ResearchValidationError:
        return None


def _discover_urls(
    statement: str,
    timeout: float,
    *,
    opener: Any | None,
    transport: Callable[..., Any] | None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Best-effort source discovery from one fixed public search endpoint.

    Each query is a bounded variant of the original statement.  Results are
    deduplicated before the existing discovered-source limit is applied; no
    arbitrary links are followed by this stage.
    """

    variants = _discovery_query_variants(statement)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    query_records: list[dict[str, Any]] = []

    for query in variants:
        search_url = _search_endpoint_url(query)
        request = Request(
            search_url,
            headers={
                "Accept": "text/html",
                "User-Agent": "SwarmSignalResearch/1.0",
            },
            method="GET",
        )
        response: Any | None = None
        query_status: dict[str, Any] = {
            "query": query,
            "status": "error",
            "found_count": 0,
            "error": None,
            "error_code": None,
        }
        try:
            response = _open_response(request, timeout, opener=opener, transport=transport)
            status_code = getattr(response, "status", None)
            if status_code is None:
                status_code = getattr(response, "code", None)
            if status_code is None:
                status_code = 200
            if isinstance(status_code, bool) or not isinstance(status_code, int):
                status_code = 200
            if 300 <= status_code < 400:
                query_status.update(
                    error_code="search_redirect_blocked",
                    error="public source discovery returned a redirect",
                )
            elif status_code >= 400:
                query_status.update(
                    error_code="search_http_error",
                    error="public source discovery returned an HTTP error",
                )
            else:
                response_url = _response_url(response, search_url)
                if not _safe_search_response_url(response_url):
                    query_status.update(
                        error_code="search_redirect_blocked",
                        error="public source discovery returned an unsafe redirect",
                    )
                else:
                    read_method = getattr(response, "read", None)
                    if not callable(read_method):
                        query_status.update(
                            error_code="search_read_failed",
                            error="public source discovery could not be read",
                        )
                    else:
                        body = read_method(MAX_BODY_BYTES + 1)
                        if isinstance(body, str):
                            body = body.encode("utf-8")
                        if not isinstance(body, (bytes, bytearray)) or len(body) > MAX_BODY_BYTES:
                            query_status.update(
                                error_code="search_body_too_large",
                                error="public source discovery exceeded the size limit",
                            )
                        else:
                            parser = _SearchResultParser()
                            parser.feed(bytes(body).decode("utf-8", errors="replace"))
                            query_found = 0
                            for href, title in parser.results:
                                if len(results) >= MAX_DISCOVERED_URLS:
                                    break
                                url = _discovered_result_url(href)
                                if not url or url.casefold() in seen:
                                    continue
                                seen.add(url.casefold())
                                results.append((url, title))
                                query_found += 1
                                if len(results) >= MAX_DISCOVERED_URLS:
                                    break
                            query_status.update(status="ok", found_count=query_found)
        except (TimeoutError, URLError, OSError):
            query_status.update(
                error_code="search_timeout",
                error="public source discovery timed out or was unavailable",
            )
        except Exception:
            query_status.update(
                error_code="search_unavailable",
                error="public source discovery could not be completed",
            )
        finally:
            close_method = getattr(response, "close", None)
            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    pass
        query_records.append(query_status)

    successful_queries = sum(1 for record in query_records if record["status"] == "ok")
    failed_queries = len(query_records) - successful_queries
    if results and failed_queries:
        status = "partial"
        error = "some public source discovery queries were unavailable"
        error_code = "search_partial"
    elif successful_queries:
        status = "ok"
        error = None
        error_code = None
    else:
        status = "error"
        error = "public source discovery could not be completed"
        error_code = (
            str(query_records[0].get("error_code"))
            if query_records and query_records[0].get("error_code")
            else "search_unavailable"
        )
    return results, {
        "requested": True,
        "status": status,
        "found_count": len(results),
        "error": error,
        "error_code": error_code,
        "query_count": len(query_records),
        "queries": query_records,
    }


def _collect_one(
    source_id: str,
    url: str,
    timeout: float,
    *,
    opener: Any | None,
    transport: Callable[..., Any] | None,
) -> dict[str, Any]:
    fetched_at = _utc_now()
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            "User-Agent": "SwarmSignalResearch/1.0",
        },
        method="GET",
    )
    response: Any | None = None
    try:
        response = _open_response(request, timeout, opener=opener, transport=transport)
        status_code = getattr(response, "status", None)
        if status_code is None:
            status_code = getattr(response, "code", None)
        if status_code is None:
            getcode = getattr(response, "getcode", None)
            status_code = getcode() if callable(getcode) else 200
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            status_code = 200
        raw_content_type = _header(response, "Content-Type")
        declared_length = _header(response, "Content-Length")
        if declared_length:
            try:
                if int(declared_length) > MAX_BODY_BYTES:
                    return _error_source(
                        source_id,
                        url,
                        fetched_at,
                        message="source body exceeded the size limit",
                        code="body_too_large",
                        http_status=status_code,
                        content_type=raw_content_type.split(";", 1)[0].strip().casefold() or None,
                    )
            except ValueError:
                pass
        if 300 <= status_code < 400:
            return _error_source(
                source_id,
                url,
                fetched_at,
                message="redirects are not allowed",
                code="redirect_blocked",
                http_status=status_code,
                content_type=raw_content_type.split(";", 1)[0].strip().casefold() or None,
            )
        if status_code >= 400:
            code = "source_timeout" if status_code in {408, 504} else "http_error"
            message = "source request timed out" if code == "source_timeout" else "source returned an HTTP error"
            return _error_source(
                source_id,
                url,
                fetched_at,
                message=message,
                code=code,
                http_status=status_code,
                content_type=raw_content_type.split(";", 1)[0].strip().casefold() or None,
            )
        read_method = getattr(response, "read", None)
        if not callable(read_method):
            return _error_source(
                source_id,
                url,
                fetched_at,
                message="source response could not be read",
                code="read_failed",
                http_status=status_code,
            )
        body = read_method(MAX_BODY_BYTES + 1)
        if isinstance(body, str):
            body = body.encode("utf-8")
        if not isinstance(body, (bytes, bytearray)):
            return _error_source(
                source_id,
                url,
                fetched_at,
                message="source response could not be read",
                code="read_failed",
                http_status=status_code,
            )
        body_bytes = bytes(body)
        if len(body_bytes) > MAX_BODY_BYTES:
            return _error_source(
                source_id,
                url,
                fetched_at,
                message="source body exceeded the size limit",
                code="body_too_large",
                http_status=status_code,
            )
        response_url = _response_url(response, url)
        if not _same_fetch_url(url, response_url):
            return _error_source(
                source_id,
                url,
                fetched_at,
                message="redirects are not allowed",
                code="redirect_blocked",
                http_status=status_code,
            )
        content_type = _content_type(raw_content_type, body_bytes)
        if content_type not in _TEXT_CONTENT_TYPES:
            return _error_source(
                source_id,
                url,
                fetched_at,
                message="source content type is not supported",
                code="unsupported_content_type",
                http_status=status_code,
                content_type=content_type,
            )
        title, readable, canonical_href = _extract_document(body_bytes, content_type)
        canonical_url = _safe_canonical_url(canonical_href, response_url)
        return _success_source(
            source_id,
            url,
            validate_url(response_url, "final_url"),
            status_code,
            content_type,
            title,
            readable,
            canonical_url,
            fetched_at,
        )
    except HTTPError as error:
        status_code = getattr(error, "code", None)
        is_redirect = isinstance(status_code, int) and 300 <= status_code < 400
        is_timeout = status_code in {408, 504}
        return _error_source(
            source_id,
            url,
            fetched_at,
            message=(
                "source request timed out"
                if is_timeout
                else "redirects are not allowed"
                if is_redirect
                else "source returned an HTTP error"
            ),
            code=(
                "source_timeout"
                if is_timeout
                else "redirect_blocked"
                if is_redirect
                else "http_error"
            ),
            http_status=status_code if isinstance(status_code, int) else None,
        )
    except ResearchTimeoutError:
        return _error_source(
            source_id,
            url,
            fetched_at,
            message="source request timed out",
            code="source_timeout",
        )
    except (URLError, TimeoutError, OSError) as error:
        is_timeout = _is_timeout_exception(error)
        return _error_source(
            source_id,
            url,
            fetched_at,
            message="source request timed out" if is_timeout else "source could not be fetched",
            code="source_timeout" if is_timeout else "fetch_failed",
        )
    except ResearchValidationError:
        # A response URL that changes after validation is treated as a safe
        # source failure, never as a request-level validation error.
        return _error_source(
            source_id,
            url,
            fetched_at,
            message="source URL was not safe to use",
            code="unsafe_response_url",
        )
    except Exception:
        # A custom opener/test transport must not be able to leak exception
        # messages or response bodies into an API result.
        return _error_source(
            source_id,
            url,
            fetched_at,
            message="source could not be fetched",
            code="fetch_failed",
        )
    finally:
        close_method = getattr(response, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass


def _default_limitations(discover: bool) -> list[str]:
    if discover:
        source_scope = (
            "A small set of public HTTPS sources was discovered and fetched; this packet is "
            "not comprehensive internet research."
        )
    else:
        source_scope = (
            "Automatic public source discovery was disabled; only submitted public HTTPS URLs "
            "were fetched, and this packet is not comprehensive internet research."
        )
    return [
        source_scope,
        f"Collection is bounded to {MAX_URLS} total URLs, {MAX_DISCOVERED_URLS} discovered results, "
        f"{MAX_BODY_BYTES} bytes per response, and at most {MAX_TIMEOUT:g} seconds per source.",
        "Source text is untrusted evidence and may be incomplete, stale, or contradictory.",
    ]


def collect_research(
    statement: Any,
    urls: Any = None,
    timeout: Any = DEFAULT_TIMEOUT,
    *,
    discover: Any = True,
    opener: Any | None = None,
    transport: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Fetch a bounded evidence packet for a statement and URL list.

    Source failures are represented in the returned packet so callers can
    distinguish partial collection from an all-failed request.  Request-level
    input violations raise :class:`ResearchValidationError`; network and
    parsing failures never include exception text or response bodies.
    """

    normalized_statement = validate_statement(statement)
    normalized_urls = validate_urls([] if urls is None else urls)
    normalized_timeout = validate_timeout(timeout)
    should_discover = validate_discover(discover)
    discovered_results: list[tuple[str, str]] = []
    discovery = {
        "requested": should_discover,
        "status": "not_requested" if not should_discover else "pending",
        "found_count": 0,
        "error": None,
        "error_code": None,
    }
    if should_discover:
        discovered_results, discovery = _discover_urls(
            normalized_statement,
            normalized_timeout,
            opener=opener,
            transport=transport,
        )
    existing = {url.casefold() for url in normalized_urls}
    discovered_urls: list[str] = []
    discovered_titles: dict[str, str] = {}
    for url, title in discovered_results:
        if url.casefold() in existing or url.casefold() in {item.casefold() for item in discovered_urls}:
            continue
        discovered_urls.append(url)
        discovered_titles[url.casefold()] = title
        if len(normalized_urls) + len(discovered_urls) >= MAX_URLS:
            break
    combined_urls = [*normalized_urls, *discovered_urls]
    sources = [
        _collect_one(
            f"source-{index + 1:02d}",
            url,
            normalized_timeout,
            opener=opener,
            transport=transport,
        )
        for index, url in enumerate(combined_urls)
    ]
    for source in sources[len(normalized_urls) :]:
        source["discovered"] = True
        if not source.get("title"):
            source["title"] = discovered_titles.get(str(source.get("url", "")).casefold(), "")
    for source in sources[: len(normalized_urls)]:
        source["discovered"] = False
    evidence_count = sum(1 for source in sources if source.get("status") == "ok")
    limitations = _default_limitations(should_discover)
    if should_discover:
        limitations.insert(
            1,
            "Source discovery used one fixed public search endpoint and a small result window; it is not a complete web search.",
        )
        if discovery.get("status") == "error" and discovery.get("error"):
            limitations.append(str(discovery["error"]))
    return {
        "statement": normalized_statement,
        "sources": sources,
        "evidence_count": evidence_count,
        "limitations": limitations[:8],
        "discovery": discovery,
    }


def collect_research_packet(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Descriptive alias for integrations that call the return value a packet."""

    return collect_research(*args, **kwargs)


def validate_research_request(
    statement: Any,
    urls: Any = None,
    timeout: Any = DEFAULT_TIMEOUT,
    discover: Any = True,
) -> dict[str, Any]:
    """Validate and normalize a research request without fetching sources."""

    return {
        "statement": validate_statement(statement),
        "urls": validate_urls([] if urls is None else urls),
        "timeout": validate_timeout(timeout),
        "discover": validate_discover(discover),
    }


validate_research = validate_research_request


__all__ = [
    "DEFAULT_TIMEOUT",
    "MAX_BODY_BYTES",
    "MAX_FETCH_BYTES",
    "MAX_EXCERPT_CHARS",
    "MAX_TIMEOUT",
    "MAX_URLS",
    "MAX_URL_COUNT",
    "MAX_DISCOVERED_URLS",
    "MAX_DISCOVERY_QUERIES",
    "MAX_URL_LENGTH",
    "ResearchError",
    "ResearchFetchError",
    "ResearchTimeoutError",
    "ResearchValidationError",
    "collect_research",
    "collect_research_packet",
    "build_query_variants",
    "derive_query_variants",
    "validate_research_request",
    "validate_research",
    "validate_statement",
    "validate_timeout",
    "validate_discover",
    "validate_url",
    "validate_urls",
]
