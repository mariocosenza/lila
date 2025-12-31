from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, TypedDict
import re

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

    iteration_count: int

    compile_attempts: int
    compile_result: Dict[str, Any]
    compile_errors: List[str]

    validated_code: str
    validation_summary: str
    test_summary: str  # New field to store the text summary of tests


# UPDATED: Added TESTS field to the required output format
VALIDATOR_SYSTEM = SystemMessage(
    content=(
        "You are VALIDATOR. You are the final step in the pipeline.\n\n"
        "### CONTEXT: GRAMMAR SPECIFICATION\n"
        f"{GRAMMO_LARK_SPEC}\n\n"
        "### TASK\n"
        "Analyze the Grammo code and the provided Test Results.\n"
        "1. Fix small syntax errors if present.\n"
        "2. Summarize the test outcome (e.g. 'Passed 2/2' or 'Failed: Output mismatch').\n"
        "3. Produce the final report.\n\n"
        "### 1. CHECKLIST\n"
        "- **I/O Syntax:** Verify strict usage of `>> \"Prompt\" # (var);` and `<< \"Msg\" # (var);`.\n"
        "- **Structure:** Ensure exactly ONE `func main` exists.\n"
        "- **Variables:** Ensure explicit `var int: x;` or constant `var x = 10;` style.\n\n"
        "### 2. STRICT OUTPUT FORMAT\n"
        "You MUST format your output exactly as follows:\n"
        "SUMMARY: [Concise summary of what the program does]\n"
        "TESTS: [Summary of test results, e.g. 'Passed', 'Failed', 'Not Run']\n"
        "[...Raw Grammo Code Here...]\n\n"
        "⛔ **NEGATIVE CONSTRAINTS** ⛔\n"
        "- **DO NOT** repeat the Grammar Specification.\n"
        "- **DO NOT** use Markdown fences (```) if possible, but if you do, ensure code is inside.\n"
        "- **DO NOT** include any conversational text after the code."
    )
)

@dataclass(frozen=True)
class ValidatorContext:
    llm_with_tools: object


def _get_candidate_code(state: ValidatorState) -> str:
    return (state.get("assembled_code") or state.get("code") or "").strip()


def _parse_validator_output(text: str) -> tuple[str, str, str]:
    """
    Parses the output to extract Summary, Test Report, and Code.
    Returns: (summary, test_summary, code)
    """
    summary = ""
    test_summary = ""
    
    lines = text.splitlines()
    clean_lines = []
    
    # 1. Extract Headers
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SUMMARY:"):
            summary = stripped[len("SUMMARY:") :].strip()
        elif stripped.startswith("TESTS:"):
            test_summary = stripped[len("TESTS:") :].strip()
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
            return summary, test_summary, code

    # 3. Fallback: Return whatever is left after stripping headers
    # Strip leading empty lines
    code = text.strip()
    
    return summary, test_summary, code


def validator_generate(ctx: ValidatorContext, state: ValidatorState) -> Dict:
    code = _get_candidate_code(state)
    task = (state.get("task") or state.get("original_task") or "").strip()
    test_result = state.get("test_result") or {}
    
    current_iter = state.get("iteration_count", 0) + 1

    user_payload = (
        "Validate and minimally fix this Grammo program.\n"
        "Do not change logic unless fixing a bug.\n\n"
        f"TASK:\n{task}\n\n"
        f"TEST_RESULTS (from Tester):\n{test_result}\n\n"
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


def validator_route_after_generate(state: ValidatorState) -> Literal["tools", "compile", "__end__"]:
    if state.get("iteration_count", 0) > 10:
        return "__end__"

    msgs = state.get("messages", [])
    if not msgs:
        return "__end__"
    
    last = msgs[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    
    return "compile"


def validator_compile(state: ValidatorState) -> Dict:
    full_text = (state.get("validated_code") or "").strip()
    
    # Strip headers (SUMMARY / TESTS) before compiling
    _, _, code_only = _parse_validator_output(full_text)

    result = grammo_compile.invoke({"code": code_only})
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
                    "Compilation failed. Apply the smallest possible fix.\n"
                    f"Errors:\n{errors or '(no details)'}\n\n"
                    "Return strict format:\n"
                    "SUMMARY: ...\nTESTS: ...\n[Code]"
                )
            )
        ]

    return out


def validator_route_after_compile(state: ValidatorState) -> Literal["generate", "__end__"]:
    if state.get("iteration_count", 0) > 10:
        return "__end__"

    result = state.get("compile_result") or {}
    compiled = bool(result.get("compiled", False))
    attempts = int(state.get("compile_attempts", 0))
    
    if compiled or attempts >= 3:
        return "__end__"
        
    return "generate"


def validator_finalize(state: ValidatorState) -> Dict:
    full_text = (state.get("validated_code") or "").strip()
    
    # Parse all fields
    summary, test_summary, code = _parse_validator_output(full_text)

    return {
        "validation_summary": summary,
        "test_summary": test_summary, # Captured in state
        "code": code or _get_candidate_code(state),
        "assembled_code": code or state.get("assembled_code", ""),
    }


def build_validator_subgraph(llm):
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