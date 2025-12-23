from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from orchestrator import build_app


IDLP_MODEL_NAME = os.getenv("IDLP_MODEL_NAME", "idlp")
UPSTREAM_OLLAMA = os.getenv("UPSTREAM_OLLAMA", "http://127.0.0.1:11434")

app = FastAPI(title="IDLP Ollama-Compatible Server")
_graph = build_app()


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _base_model_name(name: str) -> str:
    return (name or "").strip().split(":", 1)[0]


def _is_idlp_model(name: str) -> bool:
    return _base_model_name(name) == IDLP_MODEL_NAME


def _to_lc_messages(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in messages or []:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _extract_final_text(final_state: Dict[str, Any]) -> str:
    msgs = final_state.get("messages") or []
    if msgs:
        last = msgs[-1]
        content = getattr(last, "content", "") or ""
        if content.strip():
            return content.strip()

    for key in ("code", "assembled_code", "validated_code"):
        v = final_state.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    parts: List[str] = []
    summary = final_state.get("validation_summary")
    safety = final_state.get("safety_notes")
    code = final_state.get("code")
    if isinstance(summary, str) and summary.strip():
        parts.append(f"SUMMARY: {summary.strip()}")
    if isinstance(safety, str) and safety.strip():
        parts.append(f"SAFETY: {safety.strip()}")
    if isinstance(code, str) and code.strip():
        parts.append(code.strip())
    return "\n".join(parts).strip()


def _thread_id_from_payload(payload: Dict[str, Any]) -> str:
    # Ollama clients may send "session". If not, keep a stable default.
    session = payload.get("session")
    if isinstance(session, str) and session.strip():
        return session.strip()
    return "ollama-idlp-default"


async def _proxy_request(path: str, req: Request) -> Response:
    url = f"{UPSTREAM_OLLAMA}{path}"
    method = req.method.upper()

    body = None
    if method not in ("GET", "HEAD"):
        try:
            body = await req.json()
        except Exception:
            body = None

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.request(method, url, json=body)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )


@app.get("/")
def root_get() -> Response:
    return PlainTextResponse("ok", status_code=200)


@app.head("/")
def root_head() -> Response:
    return Response(status_code=200)


@app.get("/api/version")
def version_get() -> Dict[str, str]:
    return {"version": "idlp-ollama-compat-0.6"}


@app.head("/api/version")
def version_head() -> Response:
    return Response(status_code=200)


@app.get("/api/tags")
async def tags() -> Dict[str, Any]:
    models: List[Dict[str, Any]] = [
        {
            "name": f"{IDLP_MODEL_NAME}:latest",
            "model": f"{IDLP_MODEL_NAME}:latest",
            "modified_at": _iso_now(),
            "size": 0,
            "digest": "idlp-proxy",
            "details": {
                "format": "proxy",
                "family": "idlp",
                "parameter_size": "",
                "quantization_level": "",
            },
        }
    ]

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{UPSTREAM_OLLAMA}/api/tags")
            if r.status_code == 200:
                data = r.json()
                for m in data.get("models", []):
                    if _base_model_name(m.get("name", "")) != IDLP_MODEL_NAME:
                        models.append(m)
    except Exception:
        pass

    return {"models": models}


@app.post("/api/me")
async def me(_: Request) -> Response:
    return JSONResponse(content={"name": "local", "authenticated": True}, status_code=200)


@app.post("/api/show")
async def show(req: Request) -> Response:
    payload = await req.json()
    model = payload.get("model") or ""

    if model and not _is_idlp_model(model):
        return await _proxy_request("/api/show", req)

    return JSONResponse(
        content={
            "license": "",
            "modelfile": f"FROM {IDLP_MODEL_NAME}\n# proxy model\n",
            "parameters": "",  # MUST be string for Ollama Go client
            "template": "",
            "details": {
                "format": "proxy",
                "family": "idlp",
                "parameter_size": "",
                "quantization_level": "",
            },
            "model_info": {},
        },
        status_code=200,
    )


@app.post("/api/chat")
async def chat(req: Request) -> Response:
    payload = await req.json()
    model = payload.get("model") or ""
    stream = bool(payload.get("stream", False))

    if model and not _is_idlp_model(model):
        return await _proxy_request("/api/chat", req)

    try:
        lc_messages = _to_lc_messages(payload.get("messages", []))
        thread_id = _thread_id_from_payload(payload)

        final_state = _graph.invoke(
            {"messages": lc_messages, "iterations": 0, "max_iters": 12},
            config={"configurable": {"thread_id": thread_id}},
        )
        text = _extract_final_text(final_state)

        resp_obj = {
            "model": f"{IDLP_MODEL_NAME}:latest",
            "created_at": _iso_now(),
            "message": {"role": "assistant", "content": text},
            "done": True,
        }

        if not stream:
            return JSONResponse(content=resp_obj, status_code=200)

        def ndjson():
            yield (JSONResponse(content=resp_obj).body + b"\n")

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    except Exception as e:
        return JSONResponse(
            content={
                "model": f"{IDLP_MODEL_NAME}:latest",
                "created_at": _iso_now(),
                "message": {"role": "assistant", "content": f"Internal error: {e}"},
                "done": True,
            },
            status_code=200,
        )


@app.post("/api/generate")
async def generate(req: Request) -> Response:
    payload = await req.json()
    model = payload.get("model") or ""
    stream = bool(payload.get("stream", False))
    prompt = payload.get("prompt") or ""

    if model and not _is_idlp_model(model):
        return await _proxy_request("/api/generate", req)

    try:
        thread_id = _thread_id_from_payload(payload)

        final_state = _graph.invoke(
            {"messages": [HumanMessage(content=prompt)], "iterations": 0, "max_iters": 12},
            config={"configurable": {"thread_id": thread_id}},
        )
        text = _extract_final_text(final_state)

        resp_obj = {
            "model": f"{IDLP_MODEL_NAME}:latest",
            "created_at": _iso_now(),
            "response": text,
            "done": True,
        }

        if not stream:
            return JSONResponse(content=resp_obj, status_code=200)

        def ndjson():
            yield (JSONResponse(content=resp_obj).body + b"\n")

        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    except Exception as e:
        return JSONResponse(
            content={
                "model": f"{IDLP_MODEL_NAME}:latest",
                "created_at": _iso_now(),
                "response": f"Internal error: {e}",
                "done": True,
            },
            status_code=200,
        )


@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE", "HEAD"])
async def api_fallback(path: str, req: Request) -> Response:
    return await _proxy_request(f"/api/{path}", req)
