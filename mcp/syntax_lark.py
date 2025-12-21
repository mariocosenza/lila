from typing import Dict, Any
from lark import Lark, exceptions

GRAMMO_GRAMMAR = r"""
// [Insert the full grammar string here - keeping it compact for this file]
start: program
program: top_decl*
top_decl: func_def | var_decl
func_def: FUNC return_type ARROW ID LPAR param_list? RPAR block
return_type: type | VOID_TYPE
type: INT_TYPE | REAL_TYPE | BOOL_TYPE | STRING_TYPE
param_list: param (COMMA param)*
param: type COLON ID
var_decl: VAR type COLON id_list SEMI | VAR ID ASSIGN const_expr SEMI
id_list: ID (COMMA ID)*
const_expr: INT_CONST | REAL_CONST | STRING_CONST | TRUE | FALSE
block: LBRACE stmt* RBRACE
stmt: var_decl | assign_stmt SEMI | proc_call SEMI | output_stmt SEMI 
    | outputln_stmt SEMI | input_stmt SEMI | return_stmt SEMI 
    | if_stmt | while_stmt | for_stmt | block
assign_stmt: ID ASSIGN expr
proc_call: ID LPAR arg_list? RPAR
arg_list: expr (COMMA expr)*
output_stmt: OUT io_args
outputln_stmt: OUTLN io_args
input_stmt: IN io_args
io_args: (expr | HASH LPAR expr RPAR)*
return_stmt: RETURN expr?
if_stmt: IF LPAR expr RPAR block elif_list? else_block?
elif_list: elif_clause+
elif_clause: ELIF LPAR expr RPAR block
else_block: ELSE block
while_stmt: WHILE LPAR expr RPAR block
for_stmt: FOR LPAR for_init? SEMI expr? SEMI for_update? RPAR block
for_init: assign_stmt
for_update: assign_stmt
?expr: or_expr
?or_expr: and_expr | or_expr OR and_expr
?and_expr: equality_expr | and_expr AND equality_expr
?equality_expr: rel_expr | equality_expr EQ rel_expr | equality_expr NE rel_expr
?rel_expr: add_expr | rel_expr LT add_expr | rel_expr LE add_expr | rel_expr GT add_expr | rel_expr GE add_expr
?add_expr: mul_expr | add_expr PLUS mul_expr | add_expr MINUS mul_expr
?mul_expr: unary_expr | mul_expr TIMES unary_expr | mul_expr DIV unary_expr
?unary_expr: NOT unary_expr | MINUS unary_expr | primary
?primary: ID LPAR arg_list? RPAR -> func_call 
        | ID -> var | INT_CONST | REAL_CONST | STRING_CONST | TRUE | FALSE | LPAR expr RPAR

// Tokens
FUNC: "func"
VAR: "var"
INT_TYPE: "int"
REAL_TYPE: "real"
BOOL_TYPE: "bool"
STRING_TYPE: "string"
VOID_TYPE: "void"
IF: "if"
ELIF: "elif"
ELSE: "else"
WHILE: "while"
FOR: "for"
RETURN: "return"
TRUE: "true"
FALSE: "false"
OUTLN: "<<!"
OUT: "<<"
IN: ">>"
ARROW: "->"
HASH: "#"
EQ: "=="
NE: "<>"
LE: "<="
LT: "<"
GE: ">="
GT: ">"
AND: "&&"
OR: "||"
NOT: "!"
ASSIGN: "="
PLUS: "+"
MINUS: "-"
TIMES: "*"
DIV: "/"
LBRACE: "{"
RBRACE: "}"
LPAR: "("
RPAR: ")"
COLON: ":"
SEMI: ";"
COMMA: ","
ID: /[a-zA-Z_][a-zA-Z0-9_]*/
%import common.INT -> INT_CONST
%import common.FLOAT -> REAL_CONST
%import common.ESCAPED_STRING -> STRING_CONST
%import common.WS
%ignore WS
COMMENT: /\/\/[^\n]*/
COMMENT_BLOCK: /\/\*(.|\n)*?\*\//
%ignore COMMENT
%ignore COMMENT_BLOCK
"""

class GrammoParser:
    def __init__(self):
        self.parser = Lark(GRAMMO_GRAMMAR, start='start', parser='lalr')

    def validate(self, code: str) -> Dict[str, Any]:
        try:
            tree = self.parser.parse(code)
            return {
                "is_valid": True,
                "ast_preview": tree.pretty()[:2000], 
                "message": "Code is syntactically correct."
            }
        
        except exceptions.UnexpectedInput as u:
            context = u.get_context(code)
            return {
                "is_valid": False,
                "error_type": "Syntax Error",
                "line": u.line,
                "column": u.column,
                "expected": sorted(list(u.expected)),
                "context": context,
                "message": f"Unexpected input at line {u.line}, column {u.column}. Expected one of: {u.expected}"
            }
            
        except exceptions.UnexpectedToken as t:
            return {
                "is_valid": False,
                "error_type": "Unexpected Token",
                "line": t.line,
                "column": t.column,
                "token": str(t.token),
                "expected": sorted(list(t.expected)),
                "message": f"Unexpected token '{t.token}' at line {t.line}. Expected: {t.expected}"
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error_type": "General Error",
                "message": str(e)
            }