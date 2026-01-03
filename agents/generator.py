from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Literal, TypedDict, Any

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from mcp_client import grammo_compiler_mcp, grammo_lark_mcp
from prompts.generator_prompts import GRAMMO_SYSTEM, build_generator_compile_failure_message
from utils import sanitize_grammo_source, run_async_in_sync
import re

# ==========================================
# 1. Grammar Specification (FULL)
# ==========================================

# ==========================================
# 2. Helper Functions
# ==========================================

class GrammoCode(BaseModel):
    code: str = Field(
        description=(
            "A complete Grammo program conforming to the provided Lark grammar. "
            "No surrounding markdown fences; just the source text."
        )
    )

# ==========================================
# 3. Tools
# ==========================================

@tool("grammo_lark", args_schema=GrammoCode)
def grammo_lark(code: str) -> str:
    """
    Check the given source string with the Lark syntax checker.
    """
    result = run_async_in_sync(grammo_lark_mcp(code))
    return f"Lark syntax check result: {result}"


@tool("grammo_compile", args_schema=GrammoCode)
def grammo_compile(code: str) -> dict[str, str]:
    """
    Compile a source string into Grammo format.
    """
    result = run_async_in_sync(grammo_compiler_mcp(code))

    # Ensure a stable dict shape even if upstream returns weird types
    if not isinstance(result, dict):
        return {"compiled": False, "info": "", "warning": "", "errors": str(result)}

    return {
        "compiled": bool(result.get("compiled", False)),
        "info": str(result.get("info", "") or ""),
        "warning": str(result.get("warning", "") or ""),
        "errors": str(result.get("errors", "") or ""),
    }

    

TOOLS = [grammo_lark, grammo_compile]


# ==========================================
# 4. State & Prompts
# ==========================================

class GeneratorState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    max_iters: int

    code: str
    compile_attempts: int
    compile_result: dict[str, Any]
    compile_errors: list[str]


@dataclass(frozen=True)
class GeneratorContext:
    llm_with_tools: object


def ensure_system_message(messages: list[BaseMessage]) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [GRAMMO_SYSTEM, *messages]


def generator_generate(ctx: GeneratorContext, state: GeneratorState) -> dict:
    messages = ensure_system_message(state.get("messages", []))
    ai: AIMessage = ctx.llm_with_tools.invoke(messages)

    iters = int(state.get("iterations", 0)) + 1
    max_iters = int(state.get("max_iters", 5))

    code = sanitize_grammo_source(ai.content or "")
    
    # If code is empty and no tool calls, try to recover by asking explicitly next time
    # But we can't easily inject a message here without returning it.
    # The next step (compile) will catch the empty code and inject the error message.

    return {
        "messages": [ai],
        "iterations": iters,
        "max_iters": max_iters,
        "code": code,
    }


def generator_route_after_generate(state: GeneratorState) -> Literal["tools", "compile", "__end__"]:
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


def generator_compile(state: GeneratorState) -> dict:
    code = sanitize_grammo_source(state.get("code") or "")

    # Safety check: If code is empty, don't even try to compile, just fail.
    if not code or len(code.strip()) < 10:
        attempts = int(state.get("compile_attempts", 0)) + 1
        return {
            "compile_attempts": attempts,
            "compile_result": {"compiled": False, "errors": "No code generated or code too short."},
            "compile_errors": list(state.get("compile_errors", [])) + ["No code generated."],
            "code": code,
            "messages": [
                HumanMessage(content="Error: No code found. Please output the full Grammo code.")
            ]
        }

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
        "code": code
    }

    # Use max_iters from state or default to 5
    max_retries = int(state.get("max_iters", 5))
    
    if (not compiled) and attempts < max_retries:
        out["messages"] = [
            HumanMessage(
                content=build_generator_compile_failure_message(errors)
            )
        ]

    return out


def generator_route_after_compile(state: GeneratorState) -> Literal["generate", "__end__"]:
    result = state.get("compile_result") or {}
    compiled = bool(result.get("compiled", False))
    attempts = int(state.get("compile_attempts", 0))
    max_retries = int(state.get("max_iters", 5))

    if compiled:
        return "__end__"
    if attempts >= max_retries:
        return "__end__"
    return "generate"


def reset_iterations(state: GeneratorState) -> dict:
    current_iters = int(state.get("iterations", 0))
    global_iters = int(state.get("global_iterations", 0))
    new_global = global_iters + current_iters
    
    if new_global > int(state.get("max_global_iters", 30)):
        # Stop execution if global limit reached
        # We can't easily stop the whole graph from here, but we can set max_iters to 0 to force exit
        return {
            "iterations": 0, 
            "global_iterations": new_global, 
            "max_iters": 0,
            "compile_attempts": 0,
            "compile_errors": []
        }
        
    return {
        "iterations": 0, 
        "global_iterations": new_global,
        "compile_attempts": 0,
        "compile_errors": []
    }


def build_generator_subgraph(llm):
    ctx = GeneratorContext(llm_with_tools=llm.bind_tools(TOOLS))

    g = StateGraph(GeneratorState)
    g.add_node("reset", reset_iterations)
    g.add_node("generate", partial(generator_generate, ctx))
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("compile", generator_compile)

    g.add_edge(START, "reset")
    g.add_edge("reset", "generate")
    g.add_conditional_edges(
        "generate",
        generator_route_after_generate,
        {"tools": "tools", "compile": "compile", "__end__": END},
    )
    g.add_edge("tools", "generate")
    g.add_conditional_edges(
        "compile",
        generator_route_after_compile,
        {"generate": "generate", "__end__": END},
    )

    return g.compile()