import asyncio

from fastmcp import Client

from agents.mcp_client import ensure_streamable_http_transport_patch

MCP_URL = "http://localhost:8000/mcp"


async def main():
    ensure_streamable_http_transport_patch()
    client = Client(MCP_URL)

    async with client:
        # Basic server interaction
        await client.ping()

        # List available operations
        await client.list_tools()
        await client.list_resources()
        await client.list_prompts()

        # Execute operations
        result = await client.call_tool("grammo_compiler", {"code": "code"})
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
