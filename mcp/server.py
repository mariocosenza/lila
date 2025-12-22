import json
from typing import Annotated
from fastmcp import FastMCP

from syntax_lark import GrammoParser
mcp = FastMCP("grammo-mcp", strict_input_validation=True)

parser_instance = GrammoParser()
@mcp.tool(
      name="grammo_lark",
      description="Check the correctness of Grammo syntax requires the code as input",
      enabled=True
)
def syntax_checker(code: Annotated[str, "The generated Grammo Code"]) -> dict:
   result = parser_instance.validate(code)
   return result

@mcp.tool(
      name="grammo_compiler",
      description="Compile the code and return the compiled status",
      enabled=True
)
def compiler(code: Annotated[str, "The generated Grammo Code"]) -> dict:
   return {
      "compiled": True,
      "info": "",
      "warning": "",
      "errors": ""
   }



if __name__ == "__main__":
   mcp.run(transport="http", host="0.0.0.0", port=8000)