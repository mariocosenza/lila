from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, TypedDict
from langgraph.graph import CompiledGraph
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from coder import grammo_lark, grammo_compile, TOOLS as CODER_TOOLS


class IntegratorState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    task: str
    plan: List[str]
    workspace: Dict[str, str]

    code: str
    assembled_code: str

    iterations: int
    max_iters: int

    compile_attempts: int
    compile_result: Dict[str, Any]
    compile_errors: List[str]


GRAMMO_LARK_SPEC = """\
// =============================
// Grammo - Lark Grammar
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
COMMENT: /\\/\\/[^\n]*/
COMMENT_BLOCK: /\\/\\*(.|\\n)*?\\*\\//
%ignore COMMENT
%ignore COMMENT_BLOCK
"""


INTEGRATOR_SYSTEM = SystemMessage(
    content=(
        "You are INTEGRATOR.\n"
        "You are only used after the planner.\n"
        "Task: put together existing generated code into ONE working Grammo program.\n"
        "Rules:\n"
        "- FOLLOW THE LARK SPEC EXACTLY.\n"
        "- Change the MINIMUM amount of code to make it work.\n"
        "- Keep identifiers and structure stable unless required.\n"
        "- Validate syntax using `grammo_lark` and fix until valid.\n"
        "- Then compilation will run; if compilation fails, fix with minimal edits (max 3 attempts).\n"
        "- Output ONLY the final Grammo source code (no markdown).\n\n"
        "LARK SPECIFICATION:\n"
        f"{GRAMMO_LARK_SPEC}"
    )
)


@dataclass(frozen=True)
class IntegratorContext:
    llm_with_tools: object


def ensure_system(messages: List[BaseMessage]) -> List[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [INTEGRATOR_SYSTEM, *messages]


def integrator_generate(ctx: IntegratorContext, state: IntegratorState) -> Dict:
    msgs = ensure_system(state.get("messages", []))
    ai: AIMessage = ctx.llm_with_tools.invoke(msgs)

    iters = int(state.get("iterations", 0)) + 1
    max_iters = int(state.get("max_iters", 10))
    code = (ai.content or "").strip()

    return {
        "messages": [ai],
        "iterations": iters,
        "max_iters": max_iters,
        "assembled_code": code,
        "code": code,
    }


def integrator_route_after_generate(state: IntegratorState) -> Literal["tools", "compile", "__end__"]:
    iters = int(state.get("iterations", 0))
    max_iters = int(state.get("max_iters", 10))
    if iters >= max_iters:
        return "__end__"

    msgs = state.get("messages", [])
    if not msgs:
        return "__end__"

    last = msgs[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "compile"


def integrator_compile(state: IntegratorState) -> Dict:
    code = (state.get("assembled_code") or state.get("code") or "").strip()
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
                    "Compilation failed. Apply the smallest patch to fix it.\n"
                    f"Errors:\n{errors or '(no details)'}\n\n"
                    "Return ONLY the corrected Grammo code."
                )
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


def build_integrator_subgraph(llm) -> CompiledGraph:
    ctx = IntegratorContext(llm_with_tools=llm.bind_tools(CODER_TOOLS))

    g = StateGraph(IntegratorState)
    g.add_node("generate", partial(integrator_generate, ctx))
    g.add_node("tools", ToolNode(CODER_TOOLS))
    g.add_node("compile", integrator_compile)

    g.add_edge(START, "generate")
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
