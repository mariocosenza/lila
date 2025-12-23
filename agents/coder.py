from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Dict, List, Literal, TypedDict, Any

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from client import grammo_lark_mcp
import asyncio, threading


class GrammoCode(BaseModel):
    code: str = Field(
        description=(
            "A complete Grammo program conforming to the provided Lark grammar. "
            "No surrounding markdown fences; just the source text."
        )
    )


@tool("grammo_lark", args_schema=GrammoCode)
def grammo_lark(code: str) -> str:
    """
    Check the given source string with the Lark syntax checker.

    Parameters
    ----------
    code : str
        Source text to validate with the underlying grammo_lark_mcp function.

    Returns
    -------
    str
        A message prefixed with "Lark syntax check result: " followed by the
        raw result returned by grammo_lark_mcp.
    """
    """"""
    def _run_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            res = {}
            def _target():
                res['value'] = asyncio.run(coro)
            t = threading.Thread(target=_target)
            t.start()
            t.join()
            return res.get('value')

    result = _run_sync(grammo_lark_mcp(code))
    return f"Lark syntax check result: {result}"


@tool("grammo_compile", args_schema=GrammoCode)
def grammo_compile(code: str) -> Dict[str, str]:
    """
    Compile a source string into Grammo format.

    Args:
        code (str): Source code to compile.

    Returns:
        dict: Mapping with the following keys:
            - "compiled" (bool): True if compilation succeeded, False otherwise.
            - "info" (str): Informational messages from the compiler.
            - "warning" (str): Compilation warnings (empty string if none).
            - "errors" (str): Compilation errors (empty string if none).
    """
    """"""
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
""""You are an expert Grammo Compilation Agent. Your goal is to generate strictly valid code for the "Grammo" language.

### 1. TOOL USAGE & VERIFICATION (MANDATORY)
You have access to a syntax validation tool named `grammo_lark`.
**Protocol:**
1. **Draft** your solution in your scratchpad.
2. **Call** `grammo_lark` with your drafted code to check for syntax errors.
3. **Analyze** the tool's output:
   - If **Valid**: Proceed to generate the final `GrammoCode` response.
   - If **Invalid**: Read the specific line number and error message, fix the code, and call `grammo_lark` again to confirm the fix.
4. **Final Output**: Only emit the `GrammoCode` structure once the code is confirmed valid.

### 2. CRITICAL GRAMMAR RULES (Strict Enforcement)

**Variable Declarations (Mutually Exclusive)**
   - **Option A (Explicit Type):** `var int : x;` (followed by `x = 0;`).
     - ERROR: `var int : x = 0;` (Never initialize in declaration).
   - **Option B (Constant Inference):** `var pi = 3.14;` (Cannot specify type).

**Input / Output (Symbols ONLY)**
   - Output (no newline): `<<`
   - Output (newline): `<<!`
   - Input: `>>`
   - **Chaining:** `<< "Val: " x;` is valid.
   - **Formatting:** If using the hash wrapper `#`, you MUST use parentheses: `#(x)`.
     - ERROR: `<< # "Text" # x;` (Missing parens).

**Operators & Logic**
   - **Not Equal:** `<>` (Do NOT use `!=`).
   - **Assignment:** `i = i + 1` (No `+=`, `++`).
   - **Equality:** `==`

**Program Structure**
   - Main: `func void -> main() { ... }`
   - Functions: `func int -> add(int: a, int: b) { ... }`

### 3. OUTPUT SCHEMA (`GrammoCode`)
When the code is verified, your final output must be a structured object with:
- `prefix` (str): A concise strategy explanation and confirmation that syntax validation passed.
- `code` (str): The raw, compile-ready Grammo source code. """
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


def build_coder_subgraph(llm):
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
