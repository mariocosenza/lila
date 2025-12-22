from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Dict, List, Literal, TypedDict, Any

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.client import grammo_lark_mcp


class GrammoCode(BaseModel):
    code: str = Field(
        description=(
            "A complete Grammo program conforming to the provided Lark grammar. "
            "No surrounding markdown fences; just the source text."
        )
    )


@tool("grammo_lark", args_schema=GrammoCode)
def grammo_lark(code: str) -> str:
    result = grammo_lark_mcp(code)
    return f"Lark syntax check result: {result}"


@tool("grammo_compile", args_schema=GrammoCode)
def grammo_compile(code: str) -> Dict[str, str]:
    # TODO: implement real compile call
    return {
        "compiled": True,
        "info": "",
        "warning": "",
        "errors": "",
    }


TOOLS = [grammo_lark]


class CoderState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    iterations: int
    max_iters: int

    code: str
    compile_attempts: int
    compile_result: Dict[str, Any]
    compile_errors: List[str]


GRAMMO_SYSTEM = SystemMessage(
    content=(
        "You are an expert Grammo Compilation Agent. Generate strictly valid Grammo code.\n"
        "You MUST call `grammo_lark` to validate syntax; if invalid, fix and call again.\n"
        "When you are done, output ONLY the Grammo source code (no markdown fences)."
    )
)


@dataclass(frozen=True)
class CoderContext:
    llm_with_tools: object


def ensure_system_message(messages: List[BaseMessage]) -> List[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [GRAMMO_SYSTEM, *messages]


def coder_generate(ctx: CoderContext, state: CoderState) -> Dict:
    messages = ensure_system_message(state.get("messages", []))
    ai: AIMessage = ctx.llm_with_tools.invoke(messages)

    iters = int(state.get("iterations", 0)) + 1
    max_iters = int(state.get("max_iters", 12))

    code = (ai.content or "").strip()

    return {
        "messages": [ai],
        "iterations": iters,
        "max_iters": max_iters,
        "code": code,
    }


def coder_route_after_generate(state: CoderState) -> Literal["tools", "compile", "__end__"]:
    iters = int(state.get("iterations", 0))
    max_iters = int(state.get("max_iters", 12))
    if iters >= max_iters:
        return "__end__"

    msgs = state.get("messages", [])
    if not msgs:
        return "__end__"

    last = msgs[-1]
    if getattr(last, "tool_calls", None):
        return "tools"

    return "compile"


def coder_compile(state: CoderState) -> Dict:
    code = (state.get("code") or "").strip()
    result = grammo_compile.invoke({"code": code})

    attempts = int(state.get("compile_attempts", 0)) + 1
    compiled = bool(result.get("compiled", False))
    errors = (result.get("errors") or "").strip()

    compile_errors = list(state.get("compile_errors", []))
    if (not compiled) and errors:
        compile_errors.append(errors)

    out: Dict[str, Any] = {
        "compile_attempts": attempts,
        "compile_result": result,
        "compile_errors": compile_errors,
    }

    if (not compiled) and attempts < 3:
        out["messages"] = [
            SystemMessage(
                content=(
                    "Compilation failed. Fix the code and try again.\n"
                    f"Errors:\n{errors or '(no details)'}"
                )
            )
        ]

    return out


def coder_route_after_compile(state: CoderState) -> Literal["generate", "__end__"]:
    result = state.get("compile_result") or {}
    compiled = bool(result.get("compiled", False))
    attempts = int(state.get("compile_attempts", 0))

    if compiled:
        return "__end__"
    if attempts >= 3:
        return "__end__"
    return "generate"


def build_coder_subgraph(llm) -> CompiledGraph:
    ctx = CoderContext(llm_with_tools=llm.bind_tools(TOOLS))

    g = StateGraph(CoderState)
    g.add_node("generate", partial(coder_generate, ctx))
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("compile", coder_compile)

    g.add_edge(START, "generate")
    g.add_conditional_edges(
        "generate",
        coder_route_after_generate,
        {"tools": "tools", "compile": "compile", "__end__": END},
    )
    g.add_edge("tools", "generate")
    g.add_conditional_edges(
        "compile",
        coder_route_after_compile,
        {"generate": "generate", "__end__": END},
    )

    return g.compile()
