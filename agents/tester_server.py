from __future__ import annotations

import os
import uvicorn
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

# Import the new Ollama-based graph builder
from tester import build_tester_graph
from prompts.tester_prompts import build_tester_server_invoke_prompt

# --- Data Models ---
class A2ARequest(BaseModel):
    task: str = Field(default="", description="Optional testing goal")
    code: str = Field(..., description="Grammo source code to test")

class A2AResponse(BaseModel):
    tests: str
    result: Dict[str, Any]

# --- App Definition ---
def create_app() -> FastAPI:
    app = FastAPI(title="Tester A2A Server (Ollama Edition)")

    # Configuration via Environment Variables
    ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss-20b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"🔄 Building Tester Graph with Ollama model: {ollama_model} at {ollama_base_url}...")
    try:
        graph = build_tester_graph(
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url
        )
        print("✅ Tester Graph built successfully.")
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        graph = None

    @app.get("/health")
    def health_check():
        """Endpoint to check if the server is online."""
        if graph is None:
            raise HTTPException(status_code=503, detail="Graph not initialized")
        return {
            "status": "ok", 
            "backend": "ollama", 
            "model": ollama_model, 
            "url": ollama_base_url
        }

    @app.post("/invoke", response_model=A2AResponse)
    def invoke(req: A2ARequest) -> A2AResponse:
        if graph is None:
            raise HTTPException(status_code=500, detail="Tester graph is not initialized.")

        print(f"📩 Received request for task: {req.task[:50]}...")

        # Construct the initial message for the Tester Agent
        msg = HumanMessage(
            content=build_tester_server_invoke_prompt(req.task)
        )

        try:
            # Invoke the local graph
            final_state = graph.invoke({"messages": [msg], "code": req.code})
            
            # Extract results from the final state
            tests = (final_state.get("tests") or "").strip()
            result = final_state.get("test_result") or {}
            
            print("✅ Test execution completed.")
            return A2AResponse(tests=tests, result=result)
            
        except Exception as e:
            print(f"❌ Error during invocation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app

app = create_app()

# --- SERVER STARTUP ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8088))
    print(f"🚀 Starting Tester Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)