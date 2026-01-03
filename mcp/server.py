import json
from typing import Annotated, Optional, List
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from grammo.src.grammo.service import compile_text, run_tests
from syntax_lark import GrammoParser

mcp = FastMCP("grammo-mcp", strict_input_validation=True)

parser_instance = GrammoParser()


class SyntaxCheckResult(BaseModel):
    is_valid: bool = Field(..., description="True if syntax is correct")
    error_type: Optional[str] = Field(None, description="Type of syntax error")
    line: Optional[int] = Field(None, description="Line number of error")
    column: Optional[int] = Field(None, description="Column number of error")
    message: Optional[str] = Field(None, description="Error message")
    context: Optional[str] = Field(None, description="Code context around error")
    expected: Optional[List[str]] = Field(None, description="Expected tokens")

class CompilationResult(BaseModel):
    compiled: bool = Field(..., description="True if compilation succeeded")
    info: str = Field("", description="Compiler info output (includes IR)")
    warning: str = Field("", description="Compiler warnings")
    errors: str = Field("", description="Compiler errors")

class TestResult(BaseModel):
    passed: bool = Field(..., description="True if tests passed")
    stdout: str = Field("", description="Standard output from test execution")
    stderr: str = Field("", description="Standard error from test execution")

# --- Tools ---

@mcp.tool(
    name="grammo_lark",
    description="Check the correctness of Grammo syntax requires the code as input",
    enabled=True,
)
def syntax_checker(code: Annotated[str, "The generated Grammo Code"]) -> SyntaxCheckResult:
    result = parser_instance.validate(code)
    # Ensure the result dict matches the model fields
    return SyntaxCheckResult(**result)

@mcp.tool(
    name="grammo_compiler",
    description="Compile the code and return the compiled status",
    enabled=True,
)
def compiler(code: Annotated[str, "The generated Grammo Code"]) -> CompilationResult:
    result = compile_text(code, opt_level=3)
    return CompilationResult(**result)

@mcp.tool(
    name="grammo_test",
    description="Test the code and return result of the tests",
    enabled=True,
)
def tester(
    code: Annotated[str, "The generated Grammo Code"],
    tests: Annotated[str, "The test in Grammo Code"],
) -> TestResult:
    result = run_tests(code, tests)
    return TestResult(**result)

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
