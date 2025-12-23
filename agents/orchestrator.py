from __future__ import annotations

import json
import os
from functools import partial
from typing import Dict, Literal

import requests
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from coder import build_coder_subgraph
from integrator import build_integrator_subgraph
from multi_agent import AgentState, build_llm
from planner import build_planner_subgraph
from validator import build_validator_subgraph


def router_node(llm, state: AgentState) -> Dict:
    last = state["messages"][-1] if state.get("messages") else None
    task = (getattr(last, "content", "") or "").strip()

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

    return {"route": route, "task": task, "planner_used": (route == "planner")}


def pick_route(state: AgentState) -> str:
    route = state.get("route", "coder")
    if route in ("coder", "planner", "integrator"):
        return route
    return "coder"


def after_planner(state: AgentState) -> str:
    if state.get("route") == "integrator":
        return "integrator"
    if state.get("planner_used", False):
        return "tester"
    return "__end__"


def after_integrator(state: AgentState) -> str:
    if state.get("planner_used", False):
        return "tester"
    return "__end__"


def tester_a2a_node(state: AgentState) -> Dict:
    url = os.getenv("TESTER_A2A_URL", "http://127.0.0.1:8088/a2a/invoke")

    code = (state.get("code") or state.get("assembled_code") or "").strip()
    task = (state.get("task") or state.get("original_task") or "").strip()

    if not code:
        return {"tester_error": "No code found in state.", "tests": "", "test_result": {}}

    try:
        r = requests.post(url, json={"task": task, "code": code}, timeout=60)
        r.raise_for_status()
        data = r.json()
        return {"tests": (data.get("tests") or ""), "test_result": (data.get("result") or {})}
    except Exception as e:
        return {"tester_error": str(e), "tests": "", "test_result": {}}


def after_coder(_: AgentState) -> str:
    return "validator"


def after_tester(_: AgentState) -> str:
    return "validator"


def build_app():
    llm = build_llm()

    coder = build_coder_subgraph(llm)
    integrator = build_integrator_subgraph(llm)
    planner = build_planner_subgraph(llm, coder)
    validator = build_validator_subgraph(llm)

    g = StateGraph(AgentState)

    g.add_node("orchestrator", partial(router_node, llm))
    g.add_node("coder", coder)
    g.add_node("planner", planner)
    g.add_node("integrator", integrator)
    g.add_node("tester", tester_a2a_node)
    g.add_node("validator", validator)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", pick_route, {"coder": "coder", "planner": "planner"})

    g.add_conditional_edges("coder", after_coder, {"validator": "validator"})

    g.add_conditional_edges("planner", after_planner, {"integrator": "integrator", "tester": "tester", "__end__": "validator"})
    g.add_conditional_edges("integrator", after_integrator, {"tester": "tester", "__end__": "validator"})

    g.add_conditional_edges("tester", after_tester, {"validator": "validator"})

    g.add_edge("validator", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)