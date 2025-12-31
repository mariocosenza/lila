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

from client import grammo_compiler_mcp, grammo_lark_mcp
import asyncio, threading
import re

# ==========================================
# 1. Grammar Specification (FULL)
# ==========================================
GRAMMO_LARK_SPEC = """\
// =============================
// Grammo - Lark Grammar
// =============================
//
// ### CANONICAL USAGE EXAMPLE ###
// func void -> main() {
//    var int: n, r;
//    >> "Inserisci n: " # (n);   // Input pattern: Prompt string + Hash-Var
//    r = fact(n);
//    <<! "fact(n)=" # (r);       // Output pattern: Label string + Hash-Var
//    return;
// }
// =============================

start: program
// -----------------------------
// Program structure
// -----------------------------
program: top_decl*
top_decl: func_def | var_decl
// -----------------------------
// Functions / Procedures
// -----------------------------
func_def: FUNC return_type ARROW ID LPAR param_list? RPAR block
return_type: type | VOID_TYPE
type: INT_TYPE | REAL_TYPE | BOOL_TYPE | STRING_TYPE
param_list: param (COMMA param)*
param: type COLON ID
// -----------------------------
// Variable declarations
// -----------------------------
var_decl: VAR type COLON id_list SEMI | VAR ID ASSIGN const_expr SEMI
id_list: ID (COMMA ID)*
const_expr: INT_CONST | REAL_CONST | STRING_CONST | TRUE | FALSE
// -----------------------------
// Blocks & Statements
// -----------------------------
block: LBRACE stmt* RBRACE
stmt: var_decl | assign_stmt SEMI | proc_call SEMI | output_stmt SEMI | outputln_stmt SEMI | input_stmt SEMI
    | return_stmt SEMI | if_stmt | while_stmt | for_stmt | block
// -----------------------------
// Basic statements
// -----------------------------
assign_stmt: ID ASSIGN expr
proc_call: ID LPAR arg_list? RPAR
arg_list: expr (COMMA expr)*
// I/O
output_stmt: OUT io_args
outputln_stmt: OUTLN io_args
input_stmt: IN io_args
io_args: (expr | HASH LPAR expr RPAR)*
// Return
return_stmt: RETURN expr?
// -----------------------------
// Control structures
// -----------------------------
if_stmt: IF LPAR expr RPAR block elif_list? else_block?
elif_list: elif_clause+
elif_clause: ELIF LPAR expr RPAR block
else_block: ELSE block
while_stmt: WHILE LPAR expr RPAR block
for_stmt: FOR LPAR for_init? SEMI expr? SEMI for_update? RPAR block
for_init: assign_stmt
for_update: assign_stmt
// -----------------------------
// Expressions (precedence levels)
// -----------------------------
?expr: or_expr
?or_expr: and_expr | or_expr OR and_expr
?and_expr: equality_expr | and_expr AND equality_expr
?equality_expr: rel_expr | equality_expr EQ rel_expr | equality_expr NE rel_expr
?rel_expr: add_expr | rel_expr LT add_expr | rel_expr LE add_expr | rel_expr GT add_expr | rel_expr GE add_expr
?add_expr: mul_expr | add_expr PLUS mul_expr | add_expr MINUS mul_expr
?mul_expr: unary_expr | mul_expr TIMES unary_expr | mul_expr DIV unary_expr
?unary_expr: NOT unary_expr | MINUS unary_expr | primary
?primary: ID LPAR arg_list? RPAR -> func_call
        | ID -> var
        | INT_CONST
        | REAL_CONST
        | STRING_CONST
        | TRUE
        | FALSE
        | LPAR expr RPAR
// -----------------------------
// Tokens (lexical specification)
// -----------------------------
// Keywords
FUNC: "func"
VAR: "var"
INT_TYPE: "int"
REAL_TYPE: "real"
BOOL_TYPE: "bool"
STRING_TYPE:"string"
VOID_TYPE: "void"
IF: "if"
ELIF: "elif"
ELSE: "else"
WHILE: "while"
FOR: "for"
RETURN: "return"
TRUE: "true"
FALSE: "false"
// I/O and special symbols
OUTLN: "<<!"
OUT: "<<"
IN: ">>"
ARROW: "->"
HASH: "#"
// Comparison and logical operators
EQ: "=="
NE: "<>"
LE: "<="
LT: "<"
GE: ">="
GT: ">"
AND: "&&"
OR: "||"
NOT: "!"
// Assignment & arithmetic
ASSIGN: "="
PLUS: "+"
MINUS: "-"
TIMES: "*"
DIV: "/"
// Delimiters
LBRACE: "{"
RBRACE: "}"
LPAR: "("
RPAR: ")"
COLON: ":"
SEMI: ";"
COMMA: ","
// Identifiers & literals
ID: /[a-zA-Z_][a-zA-Z0-9_]*/
// Use common terminals for numbers and strings
%import common.INT -> INT_CONST
%import common.FLOAT -> REAL_CONST
%import common.ESCAPED_STRING-> STRING_CONST
// Whitespace & comments
%import common.WS
%ignore WS
COMMENT: /\\/\\/[^\\n]*/
COMMENT_BLOCK: /\\/\\*(.|\\n)*?\\*\\//
%ignore COMMENT
%ignore COMMENT_BLOCK
"""

# ==========================================
# 2. Helper Functions
# ==========================================

def _sanitize_grammo_source(text: str) -> str:
    """Best-effort sanitizer to ensure only Grammo source is passed to the compiler.

    Common failure mode: the LLM prefixes the program with natural-language
    explanations or wraps code in markdown fences.
    """
    if text is None:
        return ""

    t = str(text).strip()
    if not t:
        return ""

    # Strip markdown code fences if present.
    if '```' in t:
        m = re.search(r"```(?:\w+)?\s*\n(.*?)\n```", t, flags=re.DOTALL)
        if m:
            t = m.group(1).strip()
        else:
            t = t.replace('```', '').strip()

    # Drop leading natural-language / meta lines until we hit 'func' or 'var'.
    lines = t.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        s = line.lstrip()
        if not s:
            continue
        if s.startswith(('func', 'var')):
            start_idx = i
            break

    if start_idx is not None and start_idx > 0:
        t = "\n".join(lines[start_idx:]).strip()

    return t


class GrammoCode(BaseModel):
    code: str = Field(
        description=(
            "A complete Grammo program conforming to the provided Lark grammar. "
            "No surrounding markdown fences; just the source text."
        )
    )

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

# ==========================================
# 3. Tools
# ==========================================

@tool("grammo_lark", args_schema=GrammoCode)
def grammo_lark(code: str) -> str:
    """
    Check the given source string with the Lark syntax checker.
    """
    result = _run_sync(grammo_lark_mcp(code))
    return f"Lark syntax check result: {result}"


def _run_sync_compile(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        res: Dict[str, Any] = {}

        def _target():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                res["value"] = new_loop.run_until_complete(coro)
            except Exception as e:
                res["error"] = e
            finally:
                try:
                    new_loop.close()
                except Exception:
                    pass

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join()

        if "error" in res:
            raise res["error"]
        return res.get("value")


@tool("grammo_compile", args_schema=GrammoCode)
def grammo_compile(code: str) -> Dict[str, str]:
    """
    Compile a source string into Grammo format.
    """
    result = _run_sync_compile(grammo_compiler_mcp(code))

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
        "You are the **Grammo Architect**, an expert coding agent specialized in the Grammo programming language.\n\n"
        "### 1. OPERATIONAL PROTOCOL\n"
        "1. **DRAFT:** Internally draft the solution.\n"
        "2. **CHECK:** You **MUST** call `grammo_lark` to validate syntax.\n"
        "3. **FINALIZE:** Output only the code.\n\n"
        "### 2. STRICT OUTPUT FORMAT (RAW CODE ONLY)\n"
        "**⛔ NEGATIVE CONSTRAINTS ⛔**\n"
        "- NO Markdown (```).\n"
        "- NO 'SUMMARY:' or 'SAFETY:' sections.\n"
        "- NO Conversational filler ('Here is the code...').\n"
        "- Your output must start directly with `func` or `var`.\n\n"
        "### 3. STRICT I/O RULES\n"
        "- **Input:** Use `>> \"Prompt string: \" # (var);`\n"
        "- **Output:** Use `<<! \"Label: \" # (var);`\n\n"
        "### 4. MODULARITY (NO 'MAIN')\n"
        "**DO NOT write a `func main`** unless explicitly asked for an entry point.\n\n"
        "### 5. GRAMMAR SPECIFICATION\n"
        f"{GRAMMO_LARK_SPEC}"
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
    max_iters = int(state.get("max_iters", 5))

    code = _sanitize_grammo_source(ai.content or "")

    return {
        "messages": [ai],
        "iterations": iters,
        "max_iters": max_iters,
        "code": code,
    }


def coder_route_after_generate(state: CoderState) -> Literal["tools", "compile", "__end__"]:
    iters = int(state.get("iterations", 0))
    max_iters = int(state.get("max_iters", 3))
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
    code = _sanitize_grammo_source(state.get("code") or "")
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
        # FIX: Ensure code is returned so state updates correctly in parent graph
        "code": code
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