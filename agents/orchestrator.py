from __future__ import annotations

import json
import os
import logging
import requests
import re
from urllib.parse import urlparse
from functools import partial
from typing import Dict, Literal, Any

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# Import Subgraphs
from coder import build_coder_subgraph
from integrator import build_integrator_subgraph
from multi_agent import AgentState, build_llm
from planner import build_planner_subgraph
from validator import build_validator_subgraph

# Configure Logger
logger = logging.getLogger("orchestrator")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Configuration
TESTER_URL = os.getenv("TESTER_A2A_URL", "http://127.0.0.1:8088/a2a/invoke")


def check_tester_service(url: str) -> bool:
    """
    Checks if the external Tester service is reachable.
    """
    try:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        health_url = f"{base_url}/health"
        
        response = requests.get(health_url, timeout=1.0)
        is_up = response.status_code == 200
        
        if is_up:
            logger.info(f"✅ Tester Service is ONLINE at {health_url}")
        else:
            logger.warning(f"⚠️ Tester Service returned status {response.status_code} at {health_url}")
        return is_up

    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Tester Service is UNREACHABLE at {url}. Tests will be skipped.")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Tester Service check failed: {e}")
        return False


def router_node(llm, state: AgentState) -> Dict:
    """
    Entry point for the Orchestrator.
    Routes the user task and decides INTELLIGENTLY if testing is required.
    """
    last = state["messages"][-1] if state.get("messages") else None
    task = (getattr(last, "content", "") or "").strip()

    # --- 1. Handling Planner Loops ---
    if state.get("awaiting_approval", False):
        return {"route": "planner", "task": task, "planner_used": True}

    # --- 2. EXPLICIT USER INTENT CHECK ---
    # Check if the user explicitly ASKED for tests.
    lower_task = task.lower()
    explicit_test_keywords = ["test", "verifica", "check", "debug", "prova", "validare", "controlla", "testalo"]
    user_wants_tests = any(w in lower_task for w in explicit_test_keywords)

    if user_wants_tests:
        logger.info("⚡ Router: User explicitly requested tests. forcing run_tests=True.")

    # --- 3. LLM CLASSIFICATION ---
    sys_prompt = SystemMessage(
        content=(
            "You are the ROUTER for a Grammo coding assistant.\n"
            "Your job is to analyze the user request and output a JSON decision.\n\n"
            "### 1. ROUTE SELECTION\n"
            "- 'coder': For single-file tasks, specific functions, bug fixes, or refactoring.\n"
            "- 'planner': For multi-file architectures, complex systems, or vague requirements.\n"
            "- 'other': If the request is not about coding.\n\n"
            "### 2. TESTING DECISION ('run_tests')\n"
            "Determine if a separate TEST SUITE execution is strictly necessary.\n"
            "- **FALSE**: If the task is simple logic, math (e.g., '2+2', 'factorial'), standard algorithms, basic string manipulation, or pure documentation.\n"
            "  -> EVEN IF IT IS LOGICAL, if it is simple/standard, set FALSE unless the user asked to verify it.\n"
            "- **TRUE**: Only for complex custom business logic, unknown edge cases, or risky algorithms.\n\n"
            "**IMPORTANT:** If the user explicitly asks to 'test', 'check' or 'verify', 'run_tests' MUST be true.\n\n"
            "Return ONLY valid JSON: {\"route\": \"...\", \"run_tests\": true|false}"
        )
    )
    
    try:
        resp = llm.invoke([sys_prompt, SystemMessage(content=task)])
        raw = str(resp.content).strip()
        
        # Parse JSON
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if match:
                raw = match.group(1).strip()
        
        data = json.loads(raw)
        route = data.get("route", "coder")
        llm_run_tests = data.get("run_tests", False) # Default to False now for safety
        
    except Exception as e:
        logger.warning(f"Router LLM parsing failed: {e}. Fallback to safe defaults.")
        route = "planner" if len(task) > 300 else "coder"
        # Fallback: if short, don't test. If long/complex, test.
        llm_run_tests = True if len(task) > 200 else False

    if route == "other":
        return {
            "route": "other",
            "messages": [AIMessage(content="I am a Grammo coding assistant. Please provide a coding task.")]
        }

    # --- 4. MERGE LOGIC ---
    # The explicit user request ALWAYS overrides the LLM's "False"
    final_run_tests = True if user_wants_tests else llm_run_tests

    logger.info(f"🧠 Router Decision -> Route: {route} | Run Tests: {final_run_tests} (User Requested: {user_wants_tests})")

    return {
        "route": route, 
        "task": task, 
        "planner_used": (route == "planner"),
        "run_tests": final_run_tests
    }


def pick_route(state: AgentState) -> str:
    route = state.get("route", "coder")
    if route in ("coder", "planner", "other"):
        return route
    return "coder"


def after_planner(state: AgentState) -> str:
    if state.get("awaiting_approval", False):
        return "awaiting_approval"
    return "integrator"


def after_coder(state: AgentState) -> str:
    """
    Decides where to go after coding.
    CHECKS 'run_tests' FLAG to avoid testing trivial logical tasks.
    """
    code = state.get("code") or state.get("assembled_code")
    run_tests = state.get("run_tests", False) # Default to False if missing to be conservative
    
    # Check if code exists
    if code and len(str(code).strip()) > 5:
        if run_tests:
            logger.info("✅ Routing to TESTER (Required by complexity or user request).")
            return "tester"
        else:
            logger.info("⏩ Skipping TESTER (Simple task & no user request). Routing to VALIDATOR.")
            return "validator"
    
    logger.warning("⚠️ No valid code found. Going to VALIDATOR.")
    return "validator"


def tester_a2a_node(state: AgentState) -> Dict:
    """
    Calls the external Tester agent.
    """
    code = (state.get("assembled_code") or state.get("code") or "").strip()
    task = (state.get("original_task") or state.get("task") or "").strip()

    if not code:
        return {"tester_error": "No code found in state.", "tests": "", "test_result": {}}

    logger.info(f"🚀 Dispatching task to Tester Agent at {TESTER_URL}...")

    try:
        payload = {"task": task, "code": code}
        # High timeout because generation+execution takes time
        r = requests.post(TESTER_URL, json=payload, timeout=90)
        r.raise_for_status()
        
        data = r.json()
        logger.info("✅ Tester Agent returned results.")
        
        return {
            "tests": data.get("tests", ""), 
            "test_result": data.get("result", {})
        }
    except Exception as e:
        err = f"Tester Agent Error: {e}"
        logger.error(err)
        return {"tester_error": err}


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

    # --- Edges ---
    g.add_edge(START, "orchestrator")
    
    g.add_conditional_edges("orchestrator", pick_route, {
        "coder": "coder", 
        "planner": "planner",
        "other": END
    })

    # Coder Flow: Smart routing based on 'run_tests'
    g.add_conditional_edges("coder", after_coder, {
        "tester": "tester", 
        "validator": "validator"
    })

    g.add_conditional_edges("planner", after_planner, {
        "awaiting_approval": END,
        "integrator": "integrator"
    })


    g.add_edge("integrator", "tester")

    g.add_edge("tester", "validator")
    g.add_edge("validator", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)