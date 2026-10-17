"""Same-origin HTTP server for Swarm Signal live and demo forecasting."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import live_orchestration
import provider_adapters
import research_collector
from swarm_engine import ValidationError, simulate_forecast


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PROJECT_ROOT / "static"
MAX_BODY_BYTES = 64 * 1024


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _error_payload(code: str, message: str, field: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if field:
        error["field"] = field
    return {"error": error}


class SwarmRequestHandler(BaseHTTPRequestHandler):
    """Handle JSON API requests and files rooted below ``static/``."""

    # HTTP/1.0 closes each connection after the response, which avoids having
    # to implement request-body draining and keep-alive edge cases in this MVP.
    protocol_version = "HTTP/1.0"
    server_version = "SwarmForecast/1.0"
    sys_version = ""

    @property
    def static_root(self) -> Path:
        """Allow tests or embedding applications to replace the static root."""

        configured = getattr(self.server, "static_root", STATIC_ROOT)
        return Path(configured).resolve()

    def _send_json(
        self,
        status: int | HTTPStatus,
        payload: Any,
        extra_headers: dict[str, str] | None = None,
        send_body: bool = True,
    ) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _send_error(
        self,
        status: int | HTTPStatus,
        code: str,
        message: str,
        field: str | None = None,
        extra_headers: dict[str, str] | None = None,
        send_body: bool = True,
    ) -> None:
        self._send_json(
            status,
            _error_payload(code, message, field),
            extra_headers,
            send_body=send_body,
        )

    def _path(self) -> str:
        return urlsplit(self.path).path

    def _api_method_not_allowed(self, allow: str) -> None:
        self._send_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "method is not allowed for this endpoint",
            extra_headers={"Allow": allow},
        )

    def _read_json_body(self) -> Any:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise _RequestError(
                HTTPStatus.LENGTH_REQUIRED,
                "length_required",
                "Content-Length is required for JSON requests",
            )
        try:
            length = int(content_length)
        except ValueError as exc:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must be a non-negative integer",
            ) from exc
        if length < 0:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must be a non-negative integer",
            )
        if length > MAX_BODY_BYTES:
            raise _RequestError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                f"request body must be at most {MAX_BODY_BYTES} bytes",
            )

        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().casefold()
        if media_type and media_type != "application/json":
            raise _RequestError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json",
            )

        raw_body = self.rfile.read(length)
        if len(raw_body) != length:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "incomplete_body",
                "request body ended before Content-Length",
            )
        try:
            text = raw_body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "invalid_utf8",
                "request body must be UTF-8 JSON",
            ) from exc
        try:
            return json.loads(text, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise _RequestError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "request body must contain valid JSON",
            ) from exc

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = self._path()
        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "status": "ok",
                    "service": "swarm-signal",
                    "modes": ["live-provider-panel", "bundled-demo"],
                },
            )
            return
        if path == "/api/providers":
            self._send_json(HTTPStatus.OK, {"providers": provider_adapters.provider_catalog()})
            return
        if path == "/api/research/limits":
            self._send_json(
                HTTPStatus.OK,
                {
                    "max_urls": research_collector.MAX_URLS,
                    "max_discovered_urls": research_collector.MAX_DISCOVERED_URLS,
                    "max_discovery_queries": research_collector.MAX_DISCOVERY_QUERIES,
                    "max_url_length": research_collector.MAX_URL_LENGTH,
                    "max_body_bytes": research_collector.MAX_BODY_BYTES,
                    "default_timeout": research_collector.DEFAULT_TIMEOUT,
                    "min_timeout": research_collector.MIN_TIMEOUT,
                    "max_timeout": research_collector.MAX_TIMEOUT,
                    "default_discover": True,
                    "https_only": True,
                    "redirects": False,
                },
            )
            return
        if path.startswith("/api/") or path == "/api":
            if path in {
                "/api/simulate",
                "/api/providers/test",
                "/api/live-simulate",
                "/api/research",
            }:
                self._api_method_not_allowed("POST")
            else:
                self._send_error(
                    HTTPStatus.NOT_FOUND,
                    "not_found",
                    "API endpoint was not found",
                )
            return
        self._serve_static(path, send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = self._path()
        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "status": "ok",
                    "service": "swarm-signal",
                    "modes": ["live-provider-panel", "bundled-demo"],
                },
                send_body=False,
            )
            return
        if path == "/api/providers":
            self._send_json(
                HTTPStatus.OK,
                {"providers": provider_adapters.provider_catalog()},
                send_body=False,
            )
            return
        if path == "/api/research/limits":
            self._send_json(
                HTTPStatus.OK,
                {
                    "max_urls": research_collector.MAX_URLS,
                    "max_discovered_urls": research_collector.MAX_DISCOVERED_URLS,
                    "max_discovery_queries": research_collector.MAX_DISCOVERY_QUERIES,
                    "max_url_length": research_collector.MAX_URL_LENGTH,
                    "max_body_bytes": research_collector.MAX_BODY_BYTES,
                    "default_timeout": research_collector.DEFAULT_TIMEOUT,
                    "min_timeout": research_collector.MIN_TIMEOUT,
                    "max_timeout": research_collector.MAX_TIMEOUT,
                    "default_discover": True,
                    "https_only": True,
                    "redirects": False,
                },
                send_body=False,
            )
            return
        if path.startswith("/api/") or path == "/api":
            allow = {
                "/api/simulate": "POST",
                "/api/providers/test": "POST",
                "/api/live-simulate": "POST",
                "/api/research": "POST",
                "/api/research/limits": "GET, HEAD",
            }.get(path, "GET")
            self._send_error(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "method_not_allowed",
                "method is not allowed for this endpoint",
                extra_headers={"Allow": allow},
                send_body=False,
            )
            return
        self._serve_static(path, send_body=False)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = self._path()
        supported_paths = {
            "/api/simulate",
            "/api/providers/test",
            "/api/live-simulate",
            "/api/research",
        }
        if path not in supported_paths:
            if path.startswith("/api/") or path == "/api":
                self._send_error(
                    HTTPStatus.NOT_FOUND,
                    "not_found",
                    "API endpoint was not found",
                )
            else:
                self._api_method_not_allowed("GET, HEAD")
            return

        try:
            payload = self._read_json_body()
        except _RequestError as exc:
            self._send_error(exc.status, exc.code, exc.message)
            return

        if path == "/api/providers/test":
            self._handle_provider_test(payload)
            return
        if path == "/api/live-simulate":
            self._handle_live_simulation(payload)
            return
        if path == "/api/research":
            self._handle_research(payload)
            return

        try:
            result = simulate_forecast(payload)
        except ValidationError as exc:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                exc.message,
                exc.field,
            )
            return
        except (TypeError, KeyError) as exc:
            # Keep malformed input from becoming a traceback or an HTML error
            # response if the engine's request contract changes in the future.
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "request body did not match the simulation schema",
            )
            return
        except Exception:
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "the simulation could not be completed",
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def _provider_adapter_factory(self) -> Any:
        return getattr(self.server, "provider_adapter_factory", provider_adapters.create_adapter)

    @staticmethod
    def _safe_provider_error_message(error: provider_adapters.ProviderError) -> str:
        """Return a fixed message for provider failures that may cross the API."""

        if isinstance(error, provider_adapters.ProviderValidationError):
            # Validation messages are generated locally and contain only field
            # names and policy text, never the rejected value.
            return error.message
        if isinstance(error, provider_adapters.ProviderTimeoutError):
            return "the provider request timed out"
        if isinstance(error, provider_adapters.ProviderResponseError):
            return "the provider returned an invalid response"
        return "the provider request could not be completed"

    @staticmethod
    def _provider_test_config_payload(payload: Any) -> Any:
        """Accept a direct config or a conventional nested config wrapper."""

        if not isinstance(payload, dict):
            return payload
        for name in ("provider_config", "providerConfig", "config"):
            value = payload.get(name)
            if isinstance(value, dict):
                return value
        if isinstance(payload.get("provider"), dict):
            return payload["provider"]
        return payload

    def _handle_provider_test(self, payload: Any) -> None:
        """Make one minimal provider call and return only safe metadata."""

        try:
            config = provider_adapters.validate_provider_config(
                self._provider_test_config_payload(payload)
            )
            adapter = self._provider_adapter_factory()(config)
            started = time.monotonic()
            adapter.chat(
                [
                    {
                        "role": "user",
                        "content": "Reply with the single word OK.",
                    }
                ],
                max_tokens=8,
                temperature=0.0,
            )
            latency_ms = round((time.monotonic() - started) * 1_000, 1)
        except provider_adapters.ProviderValidationError as exc:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                exc.code,
                exc.message,
                exc.field,
            )
            return
        except provider_adapters.ProviderTimeoutError as exc:
            self._send_error(
                HTTPStatus.GATEWAY_TIMEOUT,
                exc.code,
                self._safe_provider_error_message(exc),
            )
            return
        except provider_adapters.ProviderError as exc:
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                exc.code,
                self._safe_provider_error_message(exc),
            )
            return
        except Exception:
            # Keep adapter implementation details and any provider payload out
            # of the public response.
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "provider_unavailable",
                "the provider test could not be completed",
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "status": "ok",
                "provider_id": config.provider_id,
                "model": config.model,
                "latency_ms": latency_ms,
            },
        )

    def _handle_live_simulation(self, payload: Any) -> None:
        """Run a bounded live panel and map failures to structured API errors."""

        try:
            research_opener = self._research_opener()
            research_transport = self._research_transport()
            result = live_orchestration.orchestrate_live_simulation(
                payload,
                adapter_factory=self._provider_adapter_factory(),
                research_opener=research_opener,
                research_transport=research_transport,
                # Keep the endpoint default explicit so embedding callers can
                # distinguish the HTTP workflow's automatic research policy.
                research_enabled_default=True,
            )
        except ValidationError as exc:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                exc.message,
                exc.field,
            )
            return
        except provider_adapters.ProviderValidationError as exc:
            field = f"provider.{exc.field}" if exc.field else "provider"
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                exc.code,
                exc.message,
                field,
            )
            return
        except provider_adapters.ProviderTimeoutError as exc:
            self._send_error(
                HTTPStatus.GATEWAY_TIMEOUT,
                exc.code,
                self._safe_provider_error_message(exc),
            )
            return
        except provider_adapters.ProviderError as exc:
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                exc.code,
                self._safe_provider_error_message(exc),
            )
            return
        except live_orchestration.LiveOrchestrationError as exc:
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                exc.code,
                "the live simulation could not be completed",
            )
            return
        except (TypeError, KeyError):
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                "request body did not match the live simulation schema",
            )
            return
        except Exception:
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "live_simulation_failed",
                "the live simulation could not be completed",
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def _research_opener(self) -> Any | None:
        return getattr(self.server, "research_opener", None)

    def _research_transport(self) -> Any | None:
        return getattr(self.server, "research_transport", None)

    def _handle_research(self, payload: Any) -> None:
        """Collect only the caller's bounded URL list and return safe metadata."""

        if not isinstance(payload, dict):
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_research",
                "request body must be a JSON object",
            )
            return
        try:
            collection_kwargs: dict[str, Any] = {
                "timeout": payload.get("timeout", research_collector.DEFAULT_TIMEOUT),
                "discover": payload.get("discover", True),
            }
            opener = self._research_opener()
            transport = self._research_transport()
            if opener is not None:
                collection_kwargs["opener"] = opener
            if transport is not None:
                collection_kwargs["transport"] = transport
            statement = payload.get(
                "statement",
                payload.get(
                    "question",
                    payload.get("forecasting_statement", payload.get("scenario")),
                ),
            )
            urls = payload.get("urls", payload.get("source_urls"))
            result = research_collector.collect_research(
                statement,
                urls,
                **collection_kwargs,
            )
        except research_collector.ResearchValidationError as exc:
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                exc.code,
                exc.message,
                exc.field,
            )
            return
        except research_collector.ResearchTimeoutError:
            self._send_error(
                HTTPStatus.GATEWAY_TIMEOUT,
                "research_timeout",
                "the research request timed out",
            )
            return
        except research_collector.ResearchFetchError:
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "research_unavailable",
                "the research request could not be completed",
            )
            return
        except Exception:
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "research_unavailable",
                "the research request could not be completed",
            )
            return

        sources = result.get("sources", [])
        if not isinstance(sources, list):
            self._send_error(
                HTTPStatus.BAD_GATEWAY,
                "research_unavailable",
                "the research request could not be completed",
            )
            return
        evidence_count = result.get("evidence_count")
        if not isinstance(evidence_count, int) or isinstance(evidence_count, bool):
            evidence_count = sum(
                1
                for source in sources
                if isinstance(source, dict)
                and (source.get("status") == "ok" or source.get("status") == 200)
            )
        discovery = result.get("discovery")
        discovery_failed = isinstance(discovery, dict) and discovery.get("status") == "error"
        if (sources and evidence_count == 0) or (not sources and discovery_failed):
            source_error_codes: set[str] = set()
            for source in sources:
                if not isinstance(source, dict):
                    continue
                code = source.get("error_code")
                if isinstance(code, str):
                    source_error_codes.add(code)
                error_value = source.get("error")
                if isinstance(error_value, dict) and isinstance(error_value.get("code"), str):
                    source_error_codes.add(error_value["code"])
            discovery_code = discovery.get("error_code") if isinstance(discovery, dict) else None
            timeout_codes = {
                "source_timeout",
                "timeout",
                "research_timeout",
                "search_timeout",
            }
            # A caller-supplied source can time out even when the separate
            # default discovery attempt is unavailable.  Preserve the more
            # actionable timeout status when every fetched source timed out;
            # discovery-only failures remain a safe 502.
            source_timeouts_only = bool(source_error_codes) and source_error_codes <= timeout_codes
            discovery_timeout_only = not source_error_codes and discovery_code in timeout_codes
            if source_timeouts_only or discovery_timeout_only:
                self._send_error(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    "research_timeout",
                    "all research sources timed out",
                )
            else:
                self._send_error(
                    HTTPStatus.BAD_GATEWAY,
                    "research_unavailable",
                    "all research sources failed",
                )
            return
        self._send_json(HTTPStatus.OK, result)

    def _serve_static(self, path: str, send_body: bool) -> None:
        try:
            decoded_path = unquote(path)
        except (UnicodeDecodeError, ValueError):
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_path", "URL path is invalid")
            return
        if "\x00" in decoded_path or "\\" in decoded_path:
            self._send_error(HTTPStatus.BAD_REQUEST, "invalid_path", "URL path is invalid")
            return

        root = self.static_root
        relative_path = decoded_path.lstrip("/") or "index.html"
        candidate = (root / relative_path).resolve()
        if candidate != root and root not in candidate.parents:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "static file was not found")
            return
        if candidate.is_dir():
            candidate = (candidate / "index.html").resolve()
        if candidate != root and root not in candidate.parents:
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "static file was not found")
            return
        if not candidate.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "not_found", "static file was not found")
            return

        try:
            body = candidate.read_bytes()
        except OSError:
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "static_read_error",
                "static file could not be read",
            )
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep normal access logging, but avoid exposing request bodies or
        # credentials (the server never logs them in the first place).
        super().log_message(format, *args)


class _RequestError(Exception):
    def __init__(self, status: int | HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


class SwarmHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler]):
        super().__init__(server_address, handler_class)
        self.static_root = STATIC_ROOT
        # Tests and embedding applications can inject a fake adapter factory;
        # production uses the stdlib urllib-backed factory.
        self.provider_adapter_factory = provider_adapters.create_adapter
        # Research transport is injectable for offline tests and embedding
        # applications; production uses research_collector's safe opener.
        self.research_opener = None
        self.research_transport = None


def create_server(host: str = "127.0.0.1", port: int = 8000) -> SwarmHTTPServer:
    """Create, but do not start, the local HTTP server."""

    return SwarmHTTPServer((host, port), SwarmRequestHandler)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve until interrupted."""

    server = create_server(host, port)
    print(f"Swarm forecast server listening at http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="TCP port to bind (default: 8000)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_argument_parser().parse_args(argv)
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
