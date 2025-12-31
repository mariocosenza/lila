from typing import Dict, Any
from lark import Lark, exceptions

GRAMMO_GRAMMAR = r"""// =============================
// Grammo - Lark Grammar
// =============================

start: program

// -----------------------------
// Program structure
// -----------------------------

program: top_decl*

top_decl: func_def
        | var_decl

// -----------------------------
// Functions / Procedures
// -----------------------------

func_def: FUNC return_type ARROW ID LPAR param_list? RPAR block

return_type: type
           | VOID_TYPE

type: INT_TYPE
    | REAL_TYPE
    | BOOL_TYPE
    | STRING_TYPE

param_list: param (COMMA param)*
param: type COLON ID

// -----------------------------
// Variable declarations
// -----------------------------

var_decl: VAR type COLON id_list SEMI
        | VAR ID ASSIGN const_expr SEMI

id_list: ID (COMMA ID)*

const_expr: INT_CONST
          | REAL_CONST
          | STRING_CONST
          | TRUE
          | FALSE

// -----------------------------
// Blocks & Statements
// -----------------------------

block: LBRACE stmt* RBRACE

stmt: var_decl
    | assign_stmt SEMI
    | proc_call SEMI
    | output_stmt SEMI
    | outputln_stmt SEMI
    | input_stmt SEMI
    | return_stmt SEMI
    | if_stmt
    | while_stmt
    | for_stmt
    | block

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

?or_expr: and_expr
        | or_expr OR and_expr

?and_expr: equality_expr
         | and_expr AND equality_expr

?equality_expr: rel_expr
              | equality_expr EQ rel_expr
              | equality_expr NE rel_expr

?rel_expr: add_expr
         | rel_expr LT add_expr
         | rel_expr LE add_expr
         | rel_expr GT add_expr
         | rel_expr GE add_expr

?add_expr: mul_expr
         | add_expr PLUS mul_expr
         | add_expr MINUS mul_expr

?mul_expr: unary_expr
         | mul_expr TIMES unary_expr
         | mul_expr DIV unary_expr

?unary_expr: NOT unary_expr
           | MINUS unary_expr
           | primary

?primary: ID LPAR arg_list? RPAR   -> func_call
        | ID                       -> var
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
FUNC:       "func"
VAR:        "var"
INT_TYPE:   "int"
REAL_TYPE:  "real"
BOOL_TYPE:  "bool"
STRING_TYPE:"string"
VOID_TYPE:  "void"
IF:         "if"
ELIF:       "elif"
ELSE:       "else"
WHILE:      "while"
FOR:        "for"
RETURN:     "return"
TRUE:       "true"
FALSE:      "false"

// I/O and special symbols
OUTLN:      "<<!"
OUT:        "<<"
IN:         ">>"
ARROW:      "->"
HASH:       "#"

// Comparison and logical operators
EQ:         "=="
NE:         "<>"
LE:         "<="
LT:         "<"
GE:         ">="
GT:         ">"

AND:        "&&"
OR:         "||"
NOT:        "!"

// Assignment & arithmetic
ASSIGN:     "="
PLUS:       "+"
MINUS:      "-"
TIMES:      "*"
DIV:        "/"

// Delimiters
LBRACE:     "{"
RBRACE:     "}"
LPAR:       "("
RPAR:       ")"
COLON:      ":"
SEMI:       ";"
COMMA:      ","


// Identifiers & literals
ID: /[a-zA-Z_][a-zA-Z0-9_]*/

// Use common terminals for numbers and strings
%import common.INT           -> INT_CONST
%import common.FLOAT         -> REAL_CONST
%import common.ESCAPED_STRING-> STRING_CONST

// Whitespace & comments
%import common.WS
%ignore WS

COMMENT: /\/\/[^\n]*/
COMMENT_BLOCK: /\/\*(.|\n)*?\*\//

%ignore COMMENT
%ignore COMMENT_BLOCK
"""

class GrammoParser:
    def __init__(self):
        self.parser = Lark(GRAMMO_GRAMMAR, start="start", parser="lalr")

    def validate(self, code: str) -> Dict[str, Any]:
        try:
            tree = self.parser.parse(code)
            return {
                "is_valid": True,
                "ast_preview": tree.pretty()[:2000],
                "message": "Code is syntactically correct.",
            }

        # Handle UnexpectedCharacters explicitly: it often has `.allowed` but not `.expected`
        except exceptions.UnexpectedCharacters as u:
            context = u.get_context(code) if hasattr(u, "get_context") else None
            allowed = getattr(u, "allowed", None)
            if allowed is not None:
                try:
                    allowed = sorted((allowed))
                except Exception:
                    pass

            return {
                "is_valid": False,
                "error_type": "Unexpected Characters",
                "line": getattr(u, "line", None),
                "column": getattr(u, "column", None),
                "pos_in_stream": getattr(u, "pos_in_stream", None),
                "char": getattr(u, "char", None),
                "allowed": allowed,
                "context": context,
                "message": (
                    f"Unexpected character at line {getattr(u, 'line', None)}, "
                    f"column {getattr(u, 'column', None)}."
                ),
            }

        # Handle UnexpectedToken explicitly (it *does* normally have `.expected`)
        except exceptions.UnexpectedToken as t:
            context = t.get_context(code) if hasattr(t, "get_context") else None
            expected = getattr(t, "expected", None)
            if expected is not None:
                try:
                    expected = sorted((expected))
                except Exception:
                    pass

            return {
                "is_valid": False,
                "error_type": "Unexpected Token",
                "line": getattr(t, "line", None),
                "column": getattr(t, "column", None),
                "token": str(getattr(t, "token", "")),
                "expected": expected,
                "context": context,
                "message": (
                    f"Unexpected token '{getattr(t, 'token', '')}' at line {getattr(t, 'line', None)}. "
                    f"Expected: {getattr(t, 'expected', None)}"
                ),
            }

        # Generic UnexpectedInput (covers other Lark error types safely)
        except exceptions.UnexpectedInput as u:
            context = u.get_context(code) if hasattr(u, "get_context") else None

            expected = getattr(u, "expected", None)
            if expected is not None:
                try:
                    expected = sorted((expected))
                except Exception:
                    pass

            allowed = getattr(u, "allowed", None)
            if allowed is not None:
                try:
                    allowed = sorted((allowed))
                except Exception:
                    pass

            return {
                "is_valid": False,
                "error_type": "Syntax Error",
                "line": getattr(u, "line", None),
                "column": getattr(u, "column", None),
                "expected": expected,
                "allowed": allowed,
                "context": context,
                "message": (
                    f"Unexpected input at line {getattr(u, 'line', None)}, "
                    f"column {getattr(u, 'column', None)}."
                ),
            }

        except Exception as e:
            return {
                "is_valid": False,
                "error_type": "General Error",
                "message": str(e),
            }
