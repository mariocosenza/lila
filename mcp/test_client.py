import asyncio
from fastmcp import Client


client = Client("http://localhost:8000/mcp")


async def main():
    async with client:
        # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        
        # Execute operations
        result = await client.call_tool("grammo_compiler", {"code": "code"})
        print(result)

asyncio.run(main())