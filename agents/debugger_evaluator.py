from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Literal, TypedDict
import re

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from integrator import GRAMMO_LARK_SPEC
from generator import grammo_compile, TOOLS as GENERATOR_TOOLS
from prompts.debugger_evaluator_prompts import (
    DEBUGGER_EVALUATOR_SYSTEM,
    build_debugger_evaluator_user_payload,
    build_debugger_evaluator_compile_failure_message
)


class DebuggerEvaluatorState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    task: str
    original_task: str

    code: str
    assembled_code: str

    tests: str
    test_result: dict[str, Any]

    iteration_count: int

    compile_attempts: int
    compile_result: dict[str, Any]
    compile_errors: list[str]

    validated_code: str
    validation_summary: str
    test_summary: str  # New field to store the text summary of tests
    error_summary: str # New field to store the text summary of compilation errors

    global_iterations: int
    max_global_iters: int
    max_iters: int


@dataclass(frozen=True)
class DebuggerEvaluatorContext:
    llm_with_tools: object


def _get_candidate_code(state: DebuggerEvaluatorState) -> str:
    return (state.get("assembled_code") or state.get("code") or "").strip()


def _parse_debugger_evaluator_output(text: str) -> tuple[str, str, str, str]:
    """
    Parses the output to extract Summary, Test Report, Error Report, and Code.
    Returns: (summary, test_summary, error_summary, code)
    """
    summary = ""
    test_summary = ""
    error_summary = ""
    
    lines = text.splitlines()
    clean_lines = []
    
    # 1. Extract Headers
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SUMMARY:"):
            summary = stripped[len("SUMMARY:") :].strip()
        elif stripped.startswith("TESTS:"):
            test_summary = stripped[len("TESTS:") :].strip()
        elif stripped.startswith("ERRORS:"):
            error_summary = stripped[len("ERRORS:") :].strip()
        elif stripped.startswith(("GRAMMO SPECIFICATION:", "// ===", "start: program")):
            # Stop if we hit hallucinated grammar spec
            break
        else:
            # Keep line if it's not a header
            clean_lines.append(line)

    text = "\n".join(clean_lines).strip()

    # 2. Extract Code (Handle Markdown Fences)
    if "```" in text:
        match = re.search(r"```(?:\w+)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            return summary, test_summary, error_summary, code

    # 3. Fallback: Return whatever is left after stripping headers
    # Strip leading empty lines
    code = text.strip()
    
    return summary, test_summary, error_summary, code


def debugger_evaluator_generate(ctx: DebuggerEvaluatorContext, state: DebuggerEvaluatorState) -> dict:
    code = _get_candidate_code(state)
    task = (state.get("task") or state.get("original_task") or "").strip()
    test_result = state.get("test_result") or {}
    
    current_iter = state.get("iteration_count", 0) + 1

    user_payload = build_debugger_evaluator_user_payload(task, test_result, code)

    msgs = state.get("messages", [])
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [DEBUGGER_EVALUATOR_SYSTEM, *msgs]
    else:
        msgs = msgs[:]

    msgs.append(HumanMessage(content=user_payload))

    ai: AIMessage = ctx.llm_with_tools.invoke(msgs)

    # Simple text extraction from content
    raw_content = ai.content
    content = ""
    if isinstance(raw_content, list):
        for item in raw_content:
            if isinstance(item, str): content += item
            elif isinstance(item, dict): content += item.get("text", "")
    else:
        content = (raw_content or "").strip()
    
    return {
        "messages": [ai],
        "validated_code": content,
        "iteration_count": current_iter
    }


def debugger_evaluator_route_after_generate(state: DebuggerEvaluatorState) -> Literal["tools", "compile", "__end__"]:
    if state.get("iteration_count", 0) > 10:
        return "__end__"

    msgs = state.get("messages", [])
    if not msgs:
        return "__end__"
    
    last = msgs[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    
    return "compile"


def debugger_evaluator_compile(state: DebuggerEvaluatorState) -> dict:
    full_text = (state.get("validated_code") or "").strip()
    
    # Strip headers (SUMMARY / TESTS / ERRORS) before compiling
    _, _, _, code_only = _parse_debugger_evaluator_output(full_text)

    result = grammo_compile.invoke({"code": code_only})
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

    if (not compiled) and attempts < 5:
        out["messages"] = [
            HumanMessage(
                content=build_debugger_evaluator_compile_failure_message(errors)
            )
        ]

    return out


def debugger_evaluator_route_after_compile(state: DebuggerEvaluatorState) -> Literal["generate", "__end__"]:
    if state.get("iteration_count", 0) > 10:
        return "__end__"

    result = state.get("compile_result") or {}
    compiled = bool(result.get("compiled", False))
    attempts = int(state.get("compile_attempts", 0))
    
    if compiled or attempts >= 5:
        return "__end__"
        
    return "generate"


def debugger_evaluator_finalize(state: DebuggerEvaluatorState) -> dict:
    full_text = (state.get("validated_code") or "").strip()
    
    # Parse all fields
    summary, test_summary, error_summary, code = _parse_debugger_evaluator_output(full_text)

    return {
        "validation_summary": summary,
        "test_summary": test_summary, # Captured in state
        "error_summary": error_summary, # Captured in state
        "code": code or _get_candidate_code(state),
        "assembled_code": code or state.get("assembled_code", ""),
    }


def reset_iterations(state: DebuggerEvaluatorState) -> dict:
    current_iters = int(state.get("iteration_count", 0))
    global_iters = int(state.get("global_iterations", 0))
    new_global = global_iters + current_iters
    
    if new_global > int(state.get("max_global_iters", 30)):
        # Force exit by setting iteration_count high
        return {"iteration_count": 999, "global_iterations": new_global}
        
    return {"iteration_count": 0, "global_iterations": new_global}


def build_debugger_evaluator_subgraph(llm):
    ctx = DebuggerEvaluatorContext(llm_with_tools=llm.bind_tools(GENERATOR_TOOLS))

    g = StateGraph(DebuggerEvaluatorState)
    g.add_node("reset", reset_iterations)
    g.add_node("generate", partial(debugger_evaluator_generate, ctx))
    g.add_node("tools", ToolNode(GENERATOR_TOOLS))
    g.add_node("compile", debugger_evaluator_compile)
    g.add_node("finalize", debugger_evaluator_finalize)

    g.add_edge(START, "reset")
    g.add_edge("reset", "generate")
    g.add_conditional_edges(
        "generate",
        debugger_evaluator_route_after_generate,
        {"tools": "tools", "compile": "compile", "__end__": END},
    )
    g.add_edge("tools", "generate")
    g.add_conditional_edges(
        "compile",
        debugger_evaluator_route_after_compile,
        {"generate": "generate", "__end__": "finalize"},
    )
    g.add_edge("finalize", END)

    return g.compile()