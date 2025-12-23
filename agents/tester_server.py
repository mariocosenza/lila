from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from tester import build_tester_graph


class A2ARequest(BaseModel):
    task: str = Field(default="", description="Optional testing goal")
    code: str = Field(..., description="Grammo source code to test")


class A2AResponse(BaseModel):
    tests: str
    result: Dict[str, Any]


def create_app() -> FastAPI:
    app = FastAPI(title="Tester A2A Server")

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    graph = build_tester_graph(gemini_model=gemini_model)

    @app.post("/a2a/invoke", response_model=A2AResponse)
    def invoke(req: A2ARequest) -> A2AResponse:
        msg = HumanMessage(
            content=(
                (req.task.strip() + "\n\n") if req.task.strip() else ""
                "Generate tests for the given Grammo program and run them.\n"
                "Return a short summary after the tool call."
            )
        )

        final_state = graph.invoke({"messages": [msg], "code": req.code})
        tests = (final_state.get("tests") or "").strip()
        result = final_state.get("test_result") or {}

        return A2AResponse(tests=tests, result=result)

    return app


app = create_app()