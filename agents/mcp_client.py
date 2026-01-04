from __future__ import annotations

import datetime
import json
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import httpx
from fastmcp import Client
from fastmcp.client import transports as fastmcp_transports
from mcp.client.streamable_http import streamable_http_client as _modern_streamable_http_client
from mcp.shared._httpx_utils import McpHttpClientFactory, create_mcp_http_client

MCP_URL = "http://127.0.0.1:8000/mcp"


@asynccontextmanager
async def _compat_streamable_http_client(
    url: str,
    headers: dict[str, str] | None = None,
    *,
    sse_read_timeout: float | datetime.timedelta = 60 * 5,
    terminate_on_close: bool = True,
    httpx_client_factory: McpHttpClientFactory = create_mcp_http_client,
    auth: httpx.Auth | None = None,
    **legacy_kwargs: Any,
):
    """Proxy FastMCP's legacy signature to the modern helper without new warnings."""

    timeout_value = legacy_kwargs.pop("timeout", 60)
    timeout_seconds = (
        timeout_value.total_seconds()
        if isinstance(timeout_value, datetime.timedelta)
        else float(timeout_value)
    )
    sse_read_timeout_seconds = (
        sse_read_timeout.total_seconds()
        if isinstance(sse_read_timeout, datetime.timedelta)
        else float(sse_read_timeout)
    )

    client = httpx_client_factory(
        headers=headers,
        timeout=httpx.Timeout(timeout_seconds, read=sse_read_timeout_seconds),
        auth=auth,
    )

    async with client:
        async with _modern_streamable_http_client(
            url,
            http_client=client,
            terminate_on_close=terminate_on_close,
        ) as streams:
            yield streams


def ensure_streamable_http_transport_patch() -> None:
    """Ensure FastMCP uses the compatibility wrapper without deprecation warnings."""
    current = getattr(fastmcp_transports, "streamablehttp_client", None)
    if current is not _compat_streamable_http_client:
        fastmcp_transports.streamablehttp_client = _compat_streamable_http_client


ensure_streamable_http_transport_patch()


def _create_client() -> Client:
    return Client(MCP_URL)


class SyntaxCheckResult(TypedDict):
    is_valid: bool
    error_type: str | None
    line: int | None
    column: int | None
    message: str | None
    context: str | None
    expected: list[str] | None

class CompilationResult(TypedDict):
    compiled: bool
    info: str
    warning: str
    errors: str

class TestResult(TypedDict):
    passed: bool
    stdout: str
    stderr: str



def _try_pydantic_dump(raw: Any) -> Any:
    """Try to extract dict from pydantic model."""
    if not hasattr(raw, "model_dump"):
        return None
    try:
        dumped = raw.model_dump()
        if isinstance(dumped, dict):
            return dumped
    except Exception:
        pass
    return None


def _try_direct_attributes(raw: Any) -> Any:
    """Check for common dict attributes: result, data, output."""
    for attr in ("result", "data", "output"):
        if hasattr(raw, attr):
            try:
                v = getattr(raw, attr)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return None


def _extract_text_parts(content: Any) -> list[str]:
    """Extract text parts from content list."""
    texts: list[str] = []
    try:
        for part in content:
            if part is None:
                continue
            if isinstance(part, str):
                texts.append(part)
                continue

            t = getattr(part, "text", None)
            if t is None and isinstance(part, dict):
                t = part.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t)
    except Exception:
        pass
    return texts


def _parse_json_or_string(merged: str) -> Any:
    """Try JSON parsing, fallback to plain string."""
    try:
        return json.loads(merged)
    except Exception:
        return merged


def _normalize_tool_result(raw: Any) -> Any:
    """Normalize FastMCP `call_tool` responses.

    FastMCP commonly returns a `CallToolResult` whose `.content` is a list of
    content parts (e.g., `TextContent`). Many MCP servers encode tool outputs as
    a single JSON string inside the first text part.

    This helper:
      - returns dicts unchanged
      - tries `.model_dump()` (pydantic) when present
      - extracts/concatenates text parts from `.content`
      - JSON-decodes the resulting string when possible
      - otherwise returns the best-effort string / object
    """
    if raw is None:
        return None

    if isinstance(raw, dict):
        return raw

    # Some versions return pydantic models
    pydantic_result = _try_pydantic_dump(raw)
    if pydantic_result is not None:
        return pydantic_result

    # Some versions stash a dict directly
    attr_result = _try_direct_attributes(raw)
    if attr_result is not None:
        return attr_result

    content = getattr(raw, "content", None)
    if content is None:
        return raw

    texts = _extract_text_parts(content)
    
    merged = "\n".join(texts).strip()
    if not merged:
        return raw

    return _parse_json_or_string(merged)


async def grammo_lark_mcp(code: str) -> SyntaxCheckResult:
    async with _create_client() as client:
        raw = await client.call_tool("grammo_lark", {"code": code})
        return _normalize_tool_result(raw)


async def grammo_compiler_mcp(code: str) -> CompilationResult:
    async with _create_client() as client:
        raw = await client.call_tool("grammo_compiler", {"code": code})
        return _normalize_tool_result(raw)


async def grammo_test_mcp(code: str, tests: str) -> TestResult:
    async with _create_client() as client:
        raw = await client.call_tool("grammo_test", {"code": code, "tests": tests})
        return _normalize_tool_result(raw)
