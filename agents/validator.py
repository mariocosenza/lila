from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from integrator import GRAMMO_LARK_SPEC
from coder import grammo_compile, TOOLS as CODER_TOOLS


class ValidatorState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    task: str
    original_task: str

    code: str
    assembled_code: str

    tests: str
    test_result: Dict[str, Any]

    compile_attempts: int
    compile_result: Dict[str, Any]
    compile_errors: List[str]

    validated_code: str
    validation_summary: str
    safety_notes: str



VALIDATOR_SYSTEM = SystemMessage(
    content=(
        "You are VALIDATOR.\n"
        "You receive a Grammo program produced by other agents.\n"
        "You must:\n"
        "1) Validate it against the Lark spec (below) and fix only syntax/semantic issues.\n"
        "2) Do NOT change the program logic; apply the minimum edits needed.\n"
        "3) Use `grammo_lark` tool until syntax is valid.\n"
        "4) Then compilation will run; if it fails, apply minimal fixes (max 3 attempts).\n"
        "5) Add a short summary of how generation went and a safety validation note.\n"
        "Output format:\n"
        "- First line: `SUMMARY: ...`\n"
        "- Second line: `SAFETY: ...`\n"
        "- Then the full corrected Grammo code.\n\n"
        "LARK SPECIFICATION:\n"
        f"{GRAMMO_LARK_SPEC}"
    )
)


@dataclass(frozen=True)
class ValidatorContext:
    llm_with_tools: object


def _get_candidate_code(state: ValidatorState) -> str:
    return (state.get("assembled_code") or state.get("code") or "").strip()


def validator_generate(ctx: ValidatorContext, state: ValidatorState) -> Dict:
    code = _get_candidate_code(state)
    task = (state.get("task") or state.get("original_task") or "").strip()
    test_result = state.get("test_result") or {}

    user_payload = (
        "Validate and minimally fix this Grammo program.\n"
        "Do not change logic.\n\n"
        f"TASK:\n{task}\n\n"
        f"TEST_RESULT:\n{test_result}\n\n"
        "GRAMMO CODE:\n"
        f"{code}"
    )

    msgs = state.get("messages", [])
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [VALIDATOR_SYSTEM, *msgs]
    else:
        msgs = msgs[:]

    msgs.append(HumanMessage(content=user_payload))

    ai: AIMessage = ctx.llm_with_tools.invoke(msgs)
    content = (ai.content or "").strip()

    return {
        "messages": [ai],
        "validated_code": content,
    }


def validator_route_after_generate(state: ValidatorState) -> Literal["tools", "compile", "__end__"]:
    msgs = state.get("messages", [])
    if not msgs:
        return "__end__"
    last = msgs[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "compile"


def validator_compile(state: ValidatorState) -> Dict:
    text = (state.get("validated_code") or "").strip()
    code = text

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
            HumanMessage(
                content=(
                    "Compilation failed. Apply the smallest possible fix without changing logic.\n"
                    f"Errors:\n{errors or '(no details)'}\n\n"
                    "Return the same output format:\n"
                    "SUMMARY: ...\nSAFETY: ...\n<code>"
                )
            )
        ]

    return out


def validator_route_after_compile(state: ValidatorState) -> Literal["generate", "__end__"]:
    result = state.get("compile_result") or {}
    compiled = bool(result.get("compiled", False))
    attempts = int(state.get("compile_attempts", 0))
    if compiled or attempts >= 3:
        return "__end__"
    return "generate"


def validator_finalize(state: ValidatorState) -> Dict:
    text = (state.get("validated_code") or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    summary = ""
    safety = ""
    if lines and lines[0].startswith("SUMMARY:"):
        summary = lines[0][len("SUMMARY:") :].strip()
    if len(lines) > 1 and lines[1].startswith("SAFETY:"):
        safety = lines[1][len("SAFETY:") :].strip()

    code_start = 0
    if summary:
        code_start = 1
    if safety:
        code_start = 2
    code = "\n".join(lines[code_start:]).strip()

    return {
        "validation_summary": summary,
        "safety_notes": safety,
        "code": code or _get_candidate_code(state),
        "assembled_code": code or state.get("assembled_code", ""),
    }


def build_validator_subgraph(llm) -> "langgraph.graph.CompiledGraph":
    ctx = ValidatorContext(llm_with_tools=llm.bind_tools(CODER_TOOLS))

    g = StateGraph(ValidatorState)
    g.add_node("generate", partial(validator_generate, ctx))
    g.add_node("tools", ToolNode(CODER_TOOLS))
    g.add_node("compile", validator_compile)
    g.add_node("finalize", validator_finalize)

    g.add_edge(START, "generate")
    g.add_conditional_edges(
        "generate",
        validator_route_after_generate,
        {"tools": "tools", "compile": "compile", "__end__": END},
    )
    g.add_edge("tools", "generate")
    g.add_conditional_edges(
        "compile",
        validator_route_after_compile,
        {"generate": "generate", "__end__": "finalize"},
    )
    g.add_edge("finalize", END)

    return g.compile()
