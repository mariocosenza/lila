from langchain_core.messages import SystemMessage

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
        "### 5. CRITICAL SEMANTIC CONSTRAINTS\n"
        "- **NO SHADOWING:** It is strictly FORBIDDEN to declare a local variable with the same name as a global variable or a function parameter. Check scopes carefully.\n"
        "- **RETURN PATHS:** Loops (`while`, `for`) are NOT considered guaranteed return paths. You MUST ensure there is a `return` statement reachable outside of any loop in non-void functions.\n"
        "- **TYPE COERCION:** `int` -> `real` is implicit. String concatenation uses `+`.\n\n"
        "### 6. ADVANCED FEATURES EXAMPLE\n"
        "```grammo\n"
        "func void -> main() {\n"
        "    var int: i = 5;\n"
        "    var real: r;\n"
        "    var string: s;\n"
        "    // Implicit coercion int -> real\n"
        "    r = i + 2.5;\n"
        "    // String concatenation\n"
        "    s = \"Hello\" + \" \" + \"World\";\n"
        "    <<! \"Result: \" # (s);\n"
        "}\n"
        "```\n\n"
        "### 7. GRAMMAR SPECIFICATION\n"
        f"{GRAMMO_LARK_SPEC}"
    )
)

def build_generator_compile_failure_message(errors: str) -> str:
    return (
        "Compilation failed. Fix the code and try again.\n"
        f"Errors:\n{errors or '(no details)'}"
    )
