"""Synapse Built-in Standard Library: Native HTTP Client Module.

Provides pure Python standard-library based HTTP client capabilities
using urllib.request, urllib.parse, urllib.error, http.client, and ssl.
Handles redirects, SSL certificates, UTF-8 decoding, and timeout errors
gracefully without crashing.
"""

from __future__ import annotations

import http.client as _http_client
import json as _json
import socket as _socket
import ssl as _ssl
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import urllib.error as _url_err
import urllib.parse as _url_parse
import urllib.request as _url_req


class CaseInsensitiveDict(dict):
    """A dictionary with case-insensitive string keys, preserving original key casing."""

    def __init__(self, data: Optional[Union[Dict[str, Any], Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        self._key_map: Dict[str, str] = {}
        if data:
            if hasattr(data, "items"):
                for k, v in data.items():
                    self[k] = v
            else:
                for k, v in data:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key: str, value: Any) -> None:
        norm_key = key.lower() if isinstance(key, str) else key
        old_key = self._key_map.get(norm_key)
        if old_key is not None and old_key != key:
            super().__delitem__(old_key)
        self._key_map[norm_key] = key
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        norm_key = key.lower() if isinstance(key, str) else key
        actual_key = self._key_map.get(norm_key, key)
        return super().__getitem__(actual_key)

    def __delitem__(self, key: str) -> None:
        norm_key = key.lower() if isinstance(key, str) else key
        actual_key = self._key_map.pop(norm_key, key)
        super().__delitem__(actual_key)

    def __contains__(self, key: object) -> bool:
        norm_key = key.lower() if isinstance(key, str) else key
        return norm_key in self._key_map

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class HttpResponse:
    """Represents an HTTP response from an HTTP client request.

    Attributes:
        status_code: HTTP status code integer (e.g. 200, 404, 500, 408 on timeout, 0 on failure)
        headers: Case-insensitive dictionary of response headers
        body: Raw response content as bytes
        text: Decoded UTF-8 string content
        ok: True if status_code is in 200..399 range and no network error occurred
        url: The final URL of the response (after any redirects)
        error: Error description string if request failed or timed out, else None
    """

    def __init__(
        self,
        status_code: int = 0,
        headers: Optional[Union[Dict[str, str], Any]] = None,
        body: bytes = b"",
        url: str = "",
        error: Optional[str] = None,
    ) -> None:
        self.status_code: int = status_code
        self.headers: CaseInsensitiveDict = (
            headers if isinstance(headers, CaseInsensitiveDict) else CaseInsensitiveDict(headers or {})
        )
        self.body: bytes = body if isinstance(body, (bytes, bytearray)) else bytes(body or b"")
        self.url: str = url
        self.error: Optional[str] = error
        self._text_cached: Optional[str] = None

    @property
    def ok(self) -> bool:
        """Returns True if status_code indicates success (200-399) and no network error."""
        return 200 <= self.status_code < 400 and self.error is None

    @property
    def text(self) -> str:
        """Returns body decoded as string. Handles UTF-8 and other charsets gracefully."""
        if self._text_cached is None:
            if not self.body:
                self._text_cached = ""
            else:
                encoding = "utf-8"
                ct = self.headers.get("content-type", "")
                if "charset=" in ct.lower():
                    try:
                        encoding = ct.lower().split("charset=")[-1].split(";")[0].strip().strip('"\'')
                    except Exception:
                        encoding = "utf-8"
                try:
                    self._text_cached = self.body.decode(encoding)
                except Exception:
                    try:
                        self._text_cached = self.body.decode("utf-8", errors="replace")
                    except Exception:
                        self._text_cached = self.body.decode("latin-1", errors="replace")
        return self._text_cached

    def json(self) -> Any:
        """Parses the response body as JSON. Returns parsed data, or None if empty."""
        txt = self.text.strip()
        if not txt:
            return None
        try:
            return _json.loads(txt)
        except _json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse response body as JSON: {exc}") from exc

    def __repr__(self) -> str:
        status_text = "OK" if self.ok else ("ERR" if self.error else "HTTP")
        return f"<HttpResponse [{self.status_code} {status_text}]>"

    def __bool__(self) -> bool:
        return self.ok


def request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    query: Optional[Union[Dict[str, Any], str]] = None,
    json: Any = None,
    data: Any = None,
    timeout: float = 15,
    verify_ssl: bool = True,
) -> HttpResponse:
    """Executes an HTTP request with redirect following, SSL verification, and timeout handling.

    Returns an HttpResponse object without raising exceptions on network errors or status codes.
    """
    target_url = str(url)

    # 1. Append query parameters if provided
    if query:
        if isinstance(query, dict):
            clean_query = {k: v for k, v in query.items() if v is not None}
            query_str = _url_parse.urlencode(clean_query, doseq=True)
        elif isinstance(query, str):
            query_str = query.lstrip("?")
        else:
            query_str = _url_parse.urlencode(dict(query), doseq=True)

        if query_str:
            separator = "&" if "?" in target_url else "?"
            target_url = f"{target_url}{separator}{query_str}"

    # 2. Build headers
    req_headers: Dict[str, str] = dict(headers) if headers else {}
    if not any(k.lower() == "user-agent" for k in req_headers):
        req_headers["User-Agent"] = "Synapse-HTTP-Client/1.0"

    # 3. Serialize request body
    body_bytes: Optional[bytes] = None
    if json is not None:
        if not any(k.lower() == "content-type" for k in req_headers):
            req_headers["Content-Type"] = "application/json; charset=utf-8"
        body_bytes = _json.dumps(json).encode("utf-8")
    elif data is not None:
        if isinstance(data, (bytes, bytearray)):
            body_bytes = bytes(data)
        elif isinstance(data, str):
            body_bytes = data.encode("utf-8")
        elif isinstance(data, (dict, list)):
            if not any(k.lower() == "content-type" for k in req_headers):
                req_headers["Content-Type"] = "application/x-www-form-urlencoded"
            body_bytes = _url_parse.urlencode(data, doseq=True).encode("utf-8")
        else:
            body_bytes = str(data).encode("utf-8")
    elif method.upper() in ("POST", "PUT", "PATCH"):
        body_bytes = b""

    # 4. Construct Request object
    req = _url_req.Request(
        url=target_url,
        data=body_bytes,
        headers=req_headers,
        method=method.upper(),
    )

    # 5. Configure SSL context
    ssl_context: Optional[_ssl.SSLContext] = None
    if target_url.lower().startswith("https"):
        try:
            if verify_ssl:
                ssl_context = _ssl.create_default_context()
            else:
                ssl_context = _ssl._create_unverified_context()
        except Exception:
            ssl_context = None

    # 6. Execute HTTP request safely
    try:
        kwargs: Dict[str, Any] = {"timeout": timeout}
        if ssl_context is not None:
            kwargs["context"] = ssl_context

        with _url_req.urlopen(req, **kwargs) as resp:
            resp_body = resp.read()
            status_code = getattr(resp, "status", None) or resp.getcode()
            resp_headers = dict(resp.headers.items()) if hasattr(resp, "headers") else {}
            final_url = resp.geturl() if hasattr(resp, "geturl") else target_url
            return HttpResponse(
                status_code=int(status_code),
                headers=CaseInsensitiveDict(resp_headers),
                body=resp_body,
                url=final_url,
            )
    except _url_err.HTTPError as exc:
        err_body = b""
        try:
            err_body = exc.read()
        except Exception:
            pass
        err_headers = dict(exc.headers.items()) if hasattr(exc, "headers") and exc.headers else {}
        final_url = exc.geturl() if hasattr(exc, "geturl") else target_url
        return HttpResponse(
            status_code=int(exc.code),
            headers=CaseInsensitiveDict(err_headers),
            body=err_body,
            url=final_url,
            error=str(exc),
        )
    except (_socket.timeout, TimeoutError) as exc:
        return HttpResponse(
            status_code=408,
            headers=CaseInsensitiveDict(),
            body=b"",
            url=target_url,
            error=f"Request Timeout: {exc}",
        )
    except _url_err.URLError as exc:
        is_timeout = (
            isinstance(exc.reason, (_socket.timeout, TimeoutError))
            or "timed out" in str(exc.reason).lower()
        )
        return HttpResponse(
            status_code=408 if is_timeout else 0,
            headers=CaseInsensitiveDict(),
            body=b"",
            url=target_url,
            error=f"Timeout: {exc.reason}" if is_timeout else f"Network Error: {exc.reason}",
        )
    except _ssl.SSLError as exc:
        return HttpResponse(
            status_code=0,
            headers=CaseInsensitiveDict(),
            body=b"",
            url=target_url,
            error=f"SSL Error: {exc}",
        )
    except Exception as exc:
        return HttpResponse(
            status_code=0,
            headers=CaseInsensitiveDict(),
            body=b"",
            url=target_url,
            error=f"Request Error: {exc}",
        )


def get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    query: Optional[Union[Dict[str, Any], str]] = None,
    timeout: float = 15,
    verify_ssl: bool = True,
) -> HttpResponse:
    """Sends a GET request to the specified URL."""
    return request(
        method="GET",
        url=url,
        headers=headers,
        query=query,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )


def post(
    url: str,
    json: Any = None,
    data: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15,
    verify_ssl: bool = True,
) -> HttpResponse:
    """Sends a POST request with optional JSON or form data payload."""
    return request(
        method="POST",
        url=url,
        headers=headers,
        json=json,
        data=data,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )


def put(
    url: str,
    json: Any = None,
    data: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15,
    verify_ssl: bool = True,
) -> HttpResponse:
    """Sends a PUT request with optional JSON or form data payload."""
    return request(
        method="PUT",
        url=url,
        headers=headers,
        json=json,
        data=data,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )


def delete(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15,
    verify_ssl: bool = True,
) -> HttpResponse:
    """Sends a DELETE request to the specified URL."""
    return request(
        method="DELETE",
        url=url,
        headers=headers,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )


def patch(
    url: str,
    json: Any = None,
    data: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15,
    verify_ssl: bool = True,
) -> HttpResponse:
    """Sends a PATCH request with optional JSON or form data payload."""
    return request(
        method="PATCH",
        url=url,
        headers=headers,
        json=json,
        data=data,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )


def head(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15,
    verify_ssl: bool = True,
) -> HttpResponse:
    """Sends a HEAD request to the specified URL."""
    return request(
        method="HEAD",
        url=url,
        headers=headers,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )


__all__ = [
    "HttpResponse",
    "CaseInsensitiveDict",
    "request",
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "head",
]
