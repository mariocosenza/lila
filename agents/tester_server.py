from __future__ import annotations

import os
import uvicorn
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

# Make sure this import works and tester.py is in the same folder or pythonpath
from tester import build_tester_graph

# --- Data Models ---
class A2ARequest(BaseModel):
    task: str = Field(default="", description="Optional testing goal")
    code: str = Field(..., description="Grammo source code to test")

class A2AResponse(BaseModel):
    tests: str
    result: Dict[str, Any]

# --- App Definition ---
def create_app() -> FastAPI:
    app = FastAPI(title="Tester A2A Server")

    # Config Model (override via ENV if needed)
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-27b-it")
    
    print(f"🔄 Building Tester Graph with model: {gemini_model}...")
    try:
        graph = build_tester_graph(gemini_model=gemini_model)
        print("✅ Tester Graph built successfully.")
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        graph = None

    @app.get("/health")
    def health_check():
        """Endpoint to check if the server is online."""
        if graph is None:
            raise HTTPException(status_code=503, detail="Graph not initialized")
        return {"status": "ok", "model": gemini_model}

    @app.post("/a2a/invoke", response_model=A2AResponse)
    def invoke(req: A2ARequest) -> A2AResponse:
        if graph is None:
            raise HTTPException(status_code=500, detail="Tester graph is not initialized.")

        print(f"📩 Received request for task: {req.task[:50]}...")

        # Construct the initial message for the Tester Agent
        msg = HumanMessage(
            content=(
                (req.task.strip() + "\n\n") if req.task.strip() else ""
                "Generate tests for the given Grammo program and run them.\n"
                "Return a short summary after the tool call."
            )
        )

        try:
            # Invoke the local graph
            # We initialize state with the code provided in the request
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
    # This runs the server on port 8088, accessible from 0.0.0.0
    print("🚀 Starting Tester Server on port 8088...")
    uvicorn.run(app, host="0.0.0.0", port=8088)