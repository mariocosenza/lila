from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Literal, TypedDict
import re
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from generator import grammo_compile, TOOLS as GENERATOR_TOOLS
from prompts.integrator_prompts import INTEGRATOR_SYSTEM, build_integrator_compile_failure_message, GRAMMO_LARK_SPEC


class IntegratorState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    task: str
    plan: list[str]
    workspace: dict[str, str]

    code: str
    assembled_code: str

    iterations: int
    max_iters: int

    compile_attempts: int
    compile_result: dict[str, Any]
    compile_errors: list[str]


@dataclass(frozen=True)
class IntegratorContext:
    llm_with_tools: object


def _sanitize_grammo_source(text: str) -> str:
    """Best-effort sanitizer to ensure only Grammo source is compiled.

    Removes markdown fences and drops leading natural-language/meta lines until
    the first plausible top-level declaration (`func` or `var`).
    """
    if text is None:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    if '```' in s:
        m = re.search(r"```(?:\w+)?\s*\n(.*?)\n```", s, flags=re.DOTALL)
        if m:
            s = m.group(1).strip()
        else:
            s = s.replace('```', '').strip()

    lines = s.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        line_s = line.lstrip()
        if not line_s:
            continue
        if line_s.startswith(('func', 'var')):
            start_idx = i
            break

    if start_idx is not None and start_idx > 0:
        s = "\n".join(lines[start_idx:]).strip()

    return s


def ensure_system(messages: list[BaseMessage]) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [INTEGRATOR_SYSTEM, *messages]


def integrator_generate(ctx: IntegratorContext, state: IntegratorState) -> dict:
    msgs = ensure_system(state.get("messages", []))
    ai: AIMessage = ctx.llm_with_tools.invoke(msgs)

    iters = int(state.get("iterations", 0)) + 1
    max_iters = int(state.get("max_iters", 5))
    code = _sanitize_grammo_source(ai.content or "")

    return {
        "messages": [ai],
        "iterations": iters,
        "max_iters": max_iters,
        "assembled_code": code,
        "code": code,
    }


def integrator_route_after_generate(state: IntegratorState) -> Literal["tools", "compile", "__end__"]:
    iters = int(state.get("iterations", 0))
    max_iters = int(state.get("max_iters", 5))
    if iters >= max_iters:
        return "__end__"

    msgs = state.get("messages", [])
    if not msgs:
        return "__end__"

    last = msgs[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "compile"


def integrator_compile(state: IntegratorState) -> dict:
    code = _sanitize_grammo_source(state.get("assembled_code") or state.get("code") or "")
    result = grammo_compile.invoke({"code": code})

    attempts = int(state.get("compile_attempts", 0)) + 1
    compiled = bool(result.get("compiled", False))
    errors = (result.get("errors") or "").strip()

    compile_errors = list(state.get("compile_errors", []))
    if (not compiled) and errors:
        compile_errors.append(errors)

    out: dict[str, Any] = {
        "compile_attempts": attempts,
        "compile_result": result,
        "compile_errors": compile_errors,
    }

    if (not compiled) and attempts < 3:
        out["messages"] = [
            HumanMessage(
                content=build_integrator_compile_failure_message(errors)
            )
        ]

    return out


def integrator_route_after_compile(state: IntegratorState) -> Literal["generate", "__end__"]:
    result = state.get("compile_result") or {}
    compiled = bool(result.get("compiled", False))
    attempts = int(state.get("compile_attempts", 0))

    if compiled or attempts >= 3:
        return "__end__"
    return "generate"


def reset_iterations(state: IntegratorState) -> dict:
    current_iters = int(state.get("iterations", 0))
    global_iters = int(state.get("global_iterations", 0))
    new_global = global_iters + current_iters
    
    if new_global > int(state.get("max_global_iters", 30)):
        return {"iterations": 0, "global_iterations": new_global, "max_iters": 0}
        
    return {"iterations": 0, "global_iterations": new_global}


def build_integrator_subgraph(llm):
    ctx = IntegratorContext(llm_with_tools=llm.bind_tools(GENERATOR_TOOLS))

    g = StateGraph(IntegratorState)
    g.add_node("reset", reset_iterations)
    g.add_node("generate", partial(integrator_generate, ctx))
    g.add_node("tools", ToolNode(GENERATOR_TOOLS))
    g.add_node("compile", integrator_compile)

    g.add_edge(START, "reset")
    g.add_edge("reset", "generate")
    g.add_conditional_edges(
        "generate",
        integrator_route_after_generate,
        {"tools": "tools", "compile": "compile", "__end__": END},
    )
    g.add_edge("tools", "generate")
    g.add_conditional_edges(
        "compile",
        integrator_route_after_compile,
        {"generate": "generate", "__end__": END},
    )

    return g.compile()
