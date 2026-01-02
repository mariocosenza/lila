from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

# UI Improvements
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# LangChain / Graph imports
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from orchestrator import build_app, check_tester_service, TESTER_URL

# --- 1. Fix Deprecation Warnings ---
# Silence the specific datetime warning and Pydantic-related ones
warnings.filterwarnings(
    "ignore", 
    message=".*datetime.datetime.utcfromtimestamp().*", 
    category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore", 
    message=".*support for `datetime.datetime.utcfromtimestamp()`.*", 
    category=DeprecationWarning
)

# --- 2. UI Configuration ---
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "green",
    "trace.header": "bold blue",
    "trace.content": "dim white",
    "timestamp": "dim white",
})
console = Console(theme=custom_theme)

TraceLevel = Literal["off", "basic", "debug"]

# Keys we consider useful to surface during execution.
TRACE_KEYS = {
    "route", "task", "original_task", "planner_used", "plan", "plan_step",
    "awaiting_approval", "iterations", "max_iters", "compile_attempts",
    "compile_result", "compile_errors", "test_attempts", "max_test_attempts",
    "tests", "test_result", "tester_error", "validated_code", "validation_summary",
    "safety_notes", "code", "assembled_code", "diagnostics", "workspace",
}


def _ts() -> str:
    # UPDATED: Use timezone.utc to fix the AttributeError and avoid warnings
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _safe_preview(v: Any, max_len: int = 220) -> str:
    """Creates a compact preview string for trace logs."""
    if v is None:
        return ""

    if isinstance(v, BaseMessage):
        return _preview_message(v, max_len=max_len)

    if isinstance(v, list) and v and all(isinstance(x, BaseMessage) for x in v):
        tail = v[-2:] if len(v) >= 2 else v
        return " | ".join(_preview_message(m, max_len=max_len) for m in tail)

    s = str(v).replace("\n", "\\n")
    return s if len(s) <= max_len else (s[: max_len - 3] + "...")


def _preview_message(m: BaseMessage, max_len: int = 220) -> str:
    role = getattr(m, "type", None) or m.__class__.__name__
    name = getattr(m, "name", None)
    content = getattr(m, "content", "") or ""

    tool_calls = getattr(m, "tool_calls", None)
    tc = f" tool_calls={len(tool_calls)}" if tool_calls else ""

    content_str = str(content).replace("\n", "\\n")
    if len(content_str) > max_len:
        content_str = content_str[: max_len - 3] + "..."

    label = f"{role}({name.strip()})" if (name and name.strip()) else role
    return f"{label}{tc}: {content_str}".strip()


def _build_structured_answer(summary: str, safety: str, code: str) -> str:
    """Build answer from structured fields (summary, safety, code)."""
    parts: List[str] = []
    
    if isinstance(summary, str) and summary.strip():
        parts.append(f"**SUMMARY**: {summary.strip()}")
    
    if isinstance(safety, str) and safety.strip():
        parts.append(f"**SAFETY**: {safety.strip()}")
        
    if isinstance(code, str) and code.strip():
        clean_code = code.strip('\n')
        parts.append(f"```c\n{clean_code}\n```")
    
    return "\n\n".join(parts)


def _extract_message_answer(final_state: Dict[str, Any]) -> str:
    """Fallback to the last message if no structured data."""
    msgs: List[BaseMessage] = final_state.get("messages") or []
    if msgs:
        last = msgs[-1]
        content = getattr(last, "content", "") or ""
        if content.strip():
            return content.strip()
    return "No output generated."


def _extract_answer(final_state: Dict[str, Any]) -> str:
    """
    Extracts the final answer from state.
    Code is wrapped in Markdown blocks to preserve indentation.
    """
    summary = final_state.get("validation_summary")
    safety = final_state.get("safety_notes")
    code = final_state.get("code") or final_state.get("assembled_code") or ""

    has_structured = bool(summary or safety or code)
    if has_structured:
        answer = _build_structured_answer(summary, safety, code)
        if answer:
            return answer

    return _extract_message_answer(final_state)


def _fmt_namespace(ns: Tuple[str, ...]) -> str:
    if not ns:
        return "root"
    parts: List[str] = []
    for item in ns:
        raw = str(item)
        parts.append(raw.split(":", 1)[0])
    return "/".join(parts)


def _print_updates(ns: Tuple[str, ...], node: str, updates: Dict[str, Any]) -> None:
    msg_patch = updates.get("messages") if isinstance(updates, dict) else None
    interrupt_patch = updates.get("__interrupt__") if isinstance(updates, dict) else None

    interesting = {k: v for k, v in (updates or {}).items() if k in TRACE_KEYS}
    has_any = bool(interesting) or (msg_patch is not None) or (interrupt_patch is not None)
    
    if not has_any:
        return

    phase = _fmt_namespace(ns)
    header = Text(f"[{_ts()}] ", style="timestamp")
    header.append(f"[{phase}] ", style="info")
    header.append(node, style="trace.header")
    console.print(header)

    if msg_patch is not None:
        console.print(f"  [trace.content]• messages:[/trace.content] {_safe_preview(msg_patch)}")

    if interrupt_patch is not None:
        console.print(f"  [warning]• __interrupt__:[/warning] {_safe_preview(interrupt_patch, max_len=500)}")

    for k, v in interesting.items():
        console.print(f"  [trace.content]• {k}:[/trace.content] {_safe_preview(v)}")


def _print_debug(ns: Tuple[str, ...], chunk: Any) -> None:
    phase = _fmt_namespace(ns)
    
    if isinstance(chunk, dict):
        node = chunk.get("name") or chunk.get("node") or chunk.get("id") or "debug"
        typ = chunk.get("type") or chunk.get("event") or "debug"
        payload = chunk.get("payload") or chunk.get("data") or chunk
        
        header = Text(f"[{_ts()}] ", style="timestamp")
        header.append(f"[{phase}] ", style="info")
        header.append(f"DEBUG {typ} ({node})", style="dim magenta")
        console.print(header)
        
        if payload is not chunk:
            console.print(f"  [dim]payload:[/dim] {_safe_preview(payload, max_len=500)}")
        else:
            keys = list(chunk.keys())
            console.print(f"  [dim]keys:[/dim] {keys[:30]}")
        console.print("")
        return

    console.print(f"[{_ts()}] [{phase}] DEBUG: {_safe_preview(chunk, max_len=500)}\n", style="dim")


def _split_stream_item(item: Any) -> Tuple[Tuple[str, ...], Optional[str], Any]:
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


def _handle_updates_mode(chunk: Dict, ns: Tuple[str, ...], final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle updates from stream item."""
    for node, patch in chunk.items():
        if node == "__final__":
            if isinstance(patch, dict):
                return patch
            continue
        if isinstance(patch, dict):
            _print_updates(ns, str(node), patch)
    return final_state


def _process_stream_item(item: Any, final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single stream item and update final_state if needed."""
    ns, mode, chunk = _split_stream_item(item)
    eff_mode = mode or ("updates" if isinstance(chunk, dict) else "debug")

    if eff_mode == "updates" and isinstance(chunk, dict):
        return _handle_updates_mode(chunk, ns, final_state)

    if eff_mode == "debug":
        _print_debug(ns, chunk)
    
    return final_state


def _retrieve_final_state(app, config: Dict, state_in: Dict) -> Dict[str, Any]:
    """Retrieve final state with fallbacks."""
    try:
        snap = app.get_state(config)
        return getattr(snap, "values", None) or {}
    except Exception:
        pass
    
    return app.invoke(state_in, config=config)


def _execute_with_trace(app, state_in: Dict, config: Dict, stream_mode: Any) -> Dict[str, Any]:
    """Execute app with tracing enabled."""
    final_state: Dict[str, Any] = {}
    
    console.rule("[bold blue]Execution Trace", align="left", style="blue")

    for item in app.stream(
        state_in,
        config=config,
        stream_mode=stream_mode,
        subgraphs=True,
    ):
        final_state = _process_stream_item(item, final_state)

    # Fallbacks for state retrieval
    if not final_state:
        final_state = _retrieve_final_state(app, config, state_in)
    
    return final_state


def run_turn(
    app,
    thread_id: str,
    messages: List[BaseMessage],
    trace_level: TraceLevel = "basic",
) -> Tuple[Dict[str, Any], List[BaseMessage]]:
    state_in = {
        "messages": messages, 
        "iterations": 0, 
        "max_iters": 8,
        "global_iterations": 0,
        "max_global_iters": 50
    }
    final_state: Dict[str, Any] = {}

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 150  
    }

    if trace_level == "off":
        with console.status("[bold green]Processing...", spinner="dots"):
            final_state = app.invoke(state_in, config=config)
    else:
        stream_mode: Any = "updates" if trace_level == "basic" else ["updates", "debug"]
        final_state = _execute_with_trace(app, state_in, config, stream_mode)

    # Sync local transcript
    out_msgs: List[BaseMessage] = final_state.get("messages") or []
    if out_msgs:
        messages[:] = out_msgs

    return final_state, messages


def _parse_trace_cmd(user_in: str, current: TraceLevel) -> Tuple[bool, TraceLevel, str]:
    parts = user_in.strip().split()
    if not parts or parts[0] != ":trace":
        return False, current, ""

    if len(parts) == 1:
        return True, current, f"Current trace level: [bold]{current}[/bold]"

    arg = parts[1].lower()
    if arg in ("off", "0", "false"):
        return True, "off", "Trace level set to [bold]off[/bold]"
    if arg in ("on", "1", "true", "basic"):
        return True, "basic", "Trace level set to [bold]basic[/bold]"
    if arg == "debug":
        return True, "debug", "Trace level set to [bold]debug[/bold]"

    return True, current, "[danger]Usage:[/danger] :trace off|basic|debug"


def main() -> None:
    # --- Health Check ---
    console.print("[dim]Checking Tester Service status...[/dim]")
    if check_tester_service():
        console.print("[bold green]✅ Tester Service is ONLINE[/bold green]")
    else:
        console.print(f"[bold red]❌ Tester Service is UNREACHABLE at {TESTER_URL}[/bold red]")
        console.print("[yellow]Warning: Testing capabilities will be disabled.[/yellow]")

    app = build_app()
    thread_id = str(uuid.uuid4())

    console.print(Panel.fit(
        "[bold magenta]IDLP Console Client[/bold magenta]\n[dim]LangGraph Streaming Interface[/dim]", 
        border_style="magenta"
    ))
    console.print("[dim]Commands: ':quit' to exit, ':new' new session, ':trace off|basic|debug'[/dim]\n")

    messages: List[BaseMessage] = []
    trace_level: TraceLevel = "basic"

    while True:
        try:
            user_in = console.input("[bold green]User[/bold green] > ").strip()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            break

        if not user_in:
            continue

        if user_in in (":quit", ":q", "exit"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if user_in == ":new":
            thread_id = str(uuid.uuid4())
            messages = []
            console.print(Panel("[bold green]New Session Started[/bold green]", expand=False))
            console.print("")
            continue

        handled, trace_level, msg = _parse_trace_cmd(user_in, trace_level)
        if handled:
            console.print(f"[info]{msg}[/info]\n")
            continue

        messages.append(HumanMessage(content=user_in))

        final_state, messages = run_turn(app, thread_id, messages, trace_level=trace_level)

        if trace_level != "off":
            console.rule("[bold blue]End Trace", align="left", style="blue")
            console.print("")

        answer = _extract_answer(final_state)
        
        if answer:
            console.print(Panel(Markdown(answer), title="[bold]Assistant[/bold]", border_style="green"))
            console.print("")


if __name__ == "__main__":
    main()