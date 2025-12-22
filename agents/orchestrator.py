from __future__ import annotations

import json
from typing import Dict, Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.memory import MemorySaver

from multi_agent import AgentState, build_llm
from coder import build_coder_subgraph
from planner import build_planner_subgraph


def build_app():
    llm = build_llm()
    coder = build_coder_subgraph(llm)
    planner = build_planner_subgraph(llm, coder)

    g = StateGraph(AgentState)

    def router(state: AgentState) -> Dict:
        # derive current user text from last message (set by CLI)
        last = state["messages"][-1] if state.get("messages") else None
        task = (getattr(last, "content", "") or "").strip()

        # If awaiting approval: ALWAYS go to planner, do not re-route "sì" to coder
        if state.get("awaiting_approval", False):
            return {"route": "planner", "task": task}

        sys = SystemMessage(
            content=(
                "You are a router.\n"
                "If the task is simple, choose coder.\n"
                "If the task is complex (multi-step), choose planner.\n"
                "Return ONLY JSON: {\"route\":\"coder\"} or {\"route\":\"planner\"}."
            )
        )
        resp = llm.invoke([sys, SystemMessage(content=task)])
        raw = (resp.content or "").strip()

        route: Literal["coder", "planner"] = "planner"
        try:
            data = json.loads(raw)
            if data.get("route") in ("coder", "planner"):
                route = data["route"]
        except Exception:
            route = "coder" if len(task) < 140 else "planner"

        return {"route": route, "task": task}

    def pick_route(state: AgentState) -> str:
        return state.get("route", "coder")

    g.add_node("orchestrator", router)
    g.add_node("coder", coder)
    g.add_node("planner", planner)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", pick_route, {"coder": "coder", "planner": "planner"})
    g.add_edge("coder", END)
    g.add_edge("planner", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)
