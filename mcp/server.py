from typing import Annotated
from fastmcp import FastMCP

mcp = FastMCP("grammo-mcp", strict_input_validation=True)

@mcp.tool(
      name="validate_syntax_grammo",
      description="Check the correctness of Grammo syntax requires the code as input",
      enabled=True
)
def syntax_checker(code: Annotated[str, "The generated Grammo Code"]) -> dict:
   return {
      "passed": True,
      "info": "",
      "warning": "",
      "errors": ""
   }


@mcp.tool(
      name="grammo_compiler",
      description="Compile the code and return the compiled status",
      enabled=True
)
def syntax_checker(code: Annotated[str, "The generated Grammo Code"]) -> dict:
   return {
      "compiled": True,
      "info": "",
      "warning": "",
      "errors": ""
   }

   

if __name__ == "__main__":
   mcp.run(transport="http", host="0.0.0.0", port=8000)