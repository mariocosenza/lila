from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from orchestrator import build_app


TraceLevel = Literal["off", "basic", "debug"]


# Keys we consider useful to surface during execution.
# NOTE: message patches are handled specially (we print compact previews).
TRACE_KEYS = {
    # routing / planning
    "route",
    "task",
    "original_task",
    "planner_used",
    "plan",
    "plan_step",
    "awaiting_approval",
    # iteration controls
    "iterations",
    "max_iters",
    # compilation
    "compile_attempts",
    "compile_result",
    "compile_errors",
    # testing
    "test_attempts",
    "max_test_attempts",
    "tests",
    "test_result",
    "tester_error",
    # validation
    "validated_code",
    "validation_summary",
    "safety_notes",
    # code artifacts
    "code",
    "assembled_code",
    # misc
    "diagnostics",
    "workspace",
}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _safe_preview(v: Any, max_len: int = 220) -> str:
    if v is None:
        return ""

    # Compact printing for LangChain messages
    if isinstance(v, BaseMessage):
        return _preview_message(v, max_len=max_len)

    if isinstance(v, list) and v and all(isinstance(x, BaseMessage) for x in v):
        # only show the last couple of messages to keep logs readable
        tail = v[-2:] if len(v) >= 2 else v
        return " | ".join(_preview_message(m, max_len=max_len) for m in tail)

    if isinstance(v, (dict, list, tuple)):
        s = str(v)
    else:
        s = str(v)

    s = s.replace("\n", "\\n")
    return s if len(s) <= max_len else (s[: max_len - 3] + "...")


def _preview_message(m: BaseMessage, max_len: int = 220) -> str:
    role = getattr(m, "type", None) or m.__class__.__name__
    name = getattr(m, "name", None)
    content = getattr(m, "content", "") or ""

    # tool_calls is present on some message types (e.g. AIMessage)
    tool_calls = getattr(m, "tool_calls", None)
    tc = ""
    try:
        if tool_calls:
            tc = f" tool_calls={len(tool_calls)}"
    except Exception:
        tc = ""

    if isinstance(content, (dict, list)):
        content_str = str(content)
    else:
        content_str = str(content)

    content_str = content_str.replace("\n", "\\n")
    if len(content_str) > max_len:
        content_str = content_str[: max_len - 3] + "..."

    label = role
    if isinstance(name, str) and name.strip():
        label = f"{role}({name.strip()})"

    return f"{label}{tc}: {content_str}".strip()


def _extract_answer(final_state: Dict[str, Any]) -> str:
    msgs: List[BaseMessage] = final_state.get("messages") or []
    if msgs:
        last = msgs[-1]
        content = getattr(last, "content", "") or ""
        if content.strip():
            return content.strip()

    parts: List[str] = []
    summary = final_state.get("validation_summary")
    safety = final_state.get("safety_notes")
    code = final_state.get("code") or final_state.get("assembled_code") or ""

    if isinstance(summary, str) and summary.strip():
        parts.append(f"SUMMARY: {summary.strip()}")
    if isinstance(safety, str) and safety.strip():
        parts.append(f"SAFETY: {safety.strip()}")
    if isinstance(code, str) and code.strip():
        parts.append(code.strip())

    return "\n".join(parts).strip()


def _fmt_namespace(ns: Tuple[str, ...]) -> str:
    if not ns:
        return "root"
    # Each namespace element is often like "node_name:<task_id>".
    # We keep only the node_name to show a readable phase path.
    parts: List[str] = []
    for item in ns:
        raw = str(item)
        parts.append(raw.split(":", 1)[0])
    return "/".join(parts)


def _print_updates(ns: Tuple[str, ...], node: str, updates: Dict[str, Any]) -> None:
    # Always surface message patches compactly (these are frequently the most useful signal)
    msg_patch: Optional[Any] = updates.get("messages") if isinstance(updates, dict) else None
    interrupt_patch: Optional[Any] = updates.get("__interrupt__") if isinstance(updates, dict) else None

    interesting = {k: v for k, v in (updates or {}).items() if k in TRACE_KEYS}
    has_any = bool(interesting) or (msg_patch is not None) or (interrupt_patch is not None)
    if not has_any:
        return

    phase = _fmt_namespace(ns)
    print(f"[{_ts()}] [{phase}] {node}")

    if msg_patch is not None:
        print(f"  - messages: {_safe_preview(msg_patch)}")

    if interrupt_patch is not None:
        print(f"  - __interrupt__: {_safe_preview(interrupt_patch, max_len=500)}")

    for k, v in interesting.items():
        print(f"  - {k}: {_safe_preview(v)}")
    print("")


def _print_debug(ns: Tuple[str, ...], chunk: Any) -> None:
    # Debug chunks can be large; we keep this intentionally compact.
    phase = _fmt_namespace(ns)
    if isinstance(chunk, dict):
        # Common keys seen in debug streams across LangGraph versions
        node = chunk.get("name") or chunk.get("node") or chunk.get("id") or "debug"
        typ = chunk.get("type") or chunk.get("event") or "debug"
        payload = chunk.get("payload") or chunk.get("data") or chunk
        print(f"[{_ts()}] [{phase}] DEBUG {typ} ({node})")
        if payload is not chunk:
            print(f"  - payload: {_safe_preview(payload, max_len=500)}")
        else:
            # avoid printing the same dict twice
            keys = list(chunk.keys())
            print(f"  - keys: {keys[:30]}")
        print("")
        return

    print(f"[{_ts()}] [{phase}] DEBUG: {_safe_preview(chunk, max_len=500)}\n")


def _split_stream_item(item: Any) -> Tuple[Tuple[str, ...], Optional[str], Any]:
    """Normalize LangGraph stream outputs.

    Supported shapes (see LangGraph streaming docs):
      - updates only: chunk is a dict
      - subgraphs=True: (namespace, chunk)
      - multiple stream modes: (mode, chunk)
      - multiple stream modes + subgraphs=True: (namespace, mode, chunk)
    """

    ns: Tuple[str, ...] = ()
    mode: Optional[str] = None
    chunk: Any = item

    if isinstance(item, tuple):
        if len(item) == 2:
            a, b = item
            if isinstance(a, tuple):
                ns = a
                chunk = b
            elif isinstance(a, str):
                mode = a
                chunk = b
        elif len(item) == 3:
            a, b, c = item
            if isinstance(a, tuple) and isinstance(b, str):
                ns = a
                mode = b
                chunk = c

    return ns, mode, chunk


def run_turn(
    app,
    thread_id: str,
    messages: List[BaseMessage],
    trace_level: TraceLevel = "basic",
) -> Tuple[Dict[str, Any], List[BaseMessage]]:
    state_in = {"messages": messages, "iterations": 0, "max_iters": 12}
    final_state: Dict[str, Any] = {}

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 150  
    }

    if trace_level == "off":
        final_state = app.invoke(state_in, config=config)
    else:
        # Use LangGraph built-in streaming.
        # - subgraphs=True streams outputs from nested graphs (planner/coder/integrator/validator internals).
        # - "updates" surfaces state patches after each node.
        # - "debug" (optional) surfaces additional execution details.
        stream_mode: Any = "updates" if trace_level == "basic" else ["updates", "debug"]

        for item in app.stream(
            state_in,
            config=config,
            stream_mode=stream_mode,
            subgraphs=True,
        ):
            ns, mode, chunk = _split_stream_item(item)

            # When streaming multiple modes, we may get (namespace, mode, chunk) events.
            # When streaming a single mode, mode will be None.
            eff_mode = mode or ("updates" if isinstance(chunk, dict) else "debug")

            if eff_mode == "updates":
                if isinstance(chunk, dict):
                    # updates: { "<node_name>": {state_patch...}, ... }
                    for node, patch in chunk.items():
                        if node == "__final__":
                            if isinstance(patch, dict):
                                final_state = patch
                            continue
                        if isinstance(patch, dict):
                            _print_updates(ns, str(node), patch)
                continue

            if eff_mode == "debug":
                _print_debug(ns, chunk)

        # Prefer pulling the final state from the LangGraph checkpointer (no re-run).
        if not final_state:
            try:
                snap = app.get_state(config)
                final_state = getattr(snap, "values", None) or {}
            except Exception:
                final_state = {}

        # Fallback: invoke if get_state is unavailable in the installed LangGraph version.
        if not final_state:
            final_state = app.invoke(state_in, config=config)

    # Keep local transcript in sync with the graph state.
    # (This preserves approval prompts and intermediate messages that may not be the very last message.)
    out_msgs: List[BaseMessage] = final_state.get("messages") or []
    if out_msgs:
        messages[:] = out_msgs

    return final_state, messages


def _parse_trace_cmd(user_in: str, current: TraceLevel) -> Tuple[bool, TraceLevel, str]:
    """Return (handled, new_level, message)."""

    parts = user_in.strip().split()
    if not parts or parts[0] != ":trace":
        return False, current, ""

    if len(parts) == 1:
        return True, current, f"(trace {current})\n"

    arg = parts[1].lower()
    if arg in ("off", "0", "false"):
        return True, "off", "(trace off)\n"

    if arg in ("on", "1", "true", "basic"):
        return True, "basic", "(trace basic)\n"

    if arg == "debug":
        return True, "debug", "(trace debug)\n"

    return True, current, "(usage: :trace off|basic|debug)\n"


def main() -> None:
    app = build_app()
    thread_id = str(uuid.uuid4())

    print("IDLP console client (LangGraph streaming trace)")
    print("Commands: ':quit' to exit, ':new' new session, ':trace off|basic|debug'\n")

    messages: List[BaseMessage] = []
    trace_level: TraceLevel = "basic"

    while True:
        user_in = input("> ").strip()
        if not user_in:
            continue

        if user_in in (":quit", ":q", "exit"):
            break

        if user_in == ":new":
            thread_id = str(uuid.uuid4())
            messages = []
            print("(new session)\n")
            continue

        handled, trace_level, msg = _parse_trace_cmd(user_in, trace_level)
        if handled:
            print(msg)
            continue

        messages.append(HumanMessage(content=user_in))

        if trace_level != "off":
            print("\n--- execution trace ---\n")

        final_state, messages = run_turn(app, thread_id, messages, trace_level=trace_level)

        if trace_level != "off":
            print("--- end trace ---\n")

        answer = _extract_answer(final_state)
        print(answer + "\n")


if __name__ == "__main__":
    main()
