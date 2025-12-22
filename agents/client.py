from fastmcp import Client

client = Client("http://0.0.0.0:8000/mcp")

async def grammo_lark_mcp(code: str):
    return await client.call_tool("grammo_lark", {"code": code})

