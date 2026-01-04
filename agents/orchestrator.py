from __future__ import annotations

import json
import os
import logging
import requests
import re
from copy import deepcopy
from difflib import SequenceMatcher
from functools import partial
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- Internal Imports ---
# Assuming these modules exist in your project structure
from generator import build_generator_subgraph
from integrator import build_integrator_subgraph
from multi_agent import AgentState, build_llm
from planner import build_planner_subgraph
from debugger_evaluator import build_debugger_evaluator_subgraph
from prompts.orchestrator_prompts import ROUTER_INSTRUCTIONS

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("orchestrator")

# Service URLs
TESTER_URL = os.getenv("TESTER_URL", "http://127.0.0.1:8088/invoke")
TESTER_HEALTH_URL = os.getenv("TESTER_HEALTH", "http://127.0.0.1:8088/health")

TASK_RESET_DEFAULTS: dict[str, Any] = {
    "plan": [],
    "plan_step": 0,
    "awaiting_approval": False,
    "code": "",
    "assembled_code": "",
    "compile_attempts": 0,
    "compile_result": {},
    "compile_errors": [],
    "tests": "",
    "test_result": {},
    "validated_code": "",
    "validation_summary": "",
    "diagnostics": [],
    "workspace": {},
    "safety_notes": "",
    "run_tests": False,
}

TASK_ARTIFACT_KEYS = (
    "code",
    "assembled_code",
    "plan",
    "validated_code",
    "compile_result",
    "tests",
    "test_result",
)

REFINEMENT_KEYWORDS = (
    "refine",
    "refinement",
    "improve",
    "improvement",
    "fix",
    "fixing",
    "bug",
    "bugs",
    "debug",
    "update",
    "upgrade",
    "change",
    "adjust",
    "tweak",
    "optimize",
    "optimize",
    "extend",
    "extension",
    "enhance",
    "aggiungi",
    "aggiungere",
    "migliora",
    "migliorare",
    "correggi",
    "correggere",
    "sistema",
    "sistemare",
    "riusa",
    "riutilizza",
    "reuse",
    "continue",
    "continua",
    "mantieni",
    "espandi",
    "expand",
)


def _normalize_text(val: str) -> str:
    return (val or "").strip().lower()


def _looks_like_refinement(task: str) -> bool:
    text = _normalize_text(task)
    if not text:
        return False
    return any(keyword in text for keyword in REFINEMENT_KEYWORDS)


def _has_prior_artifacts(state: AgentState) -> bool:
    for key in TASK_ARTIFACT_KEYS:
        value = state.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
        if not isinstance(value, (dict, list)) and value:
            return True
    return False


def _similar_to_previous_task(task: str, state: AgentState, threshold: float = 0.78) -> bool:
    previous = _normalize_text(state.get("original_task") or state.get("task") or "")
    current = _normalize_text(task)
    if not previous or not current:
        return False
    return SequenceMatcher(None, current, previous).ratio() >= threshold


def _should_reset_task_state(task: str, state: AgentState) -> bool:
    if not task:
        return False
    if state.get("awaiting_approval"):
        return False
    if not _has_prior_artifacts(state):
        return False
    if _looks_like_refinement(task):
        return False
    if _similar_to_previous_task(task, state):
        return False
    return True


def _build_task_reset_patch() -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for key, default_value in TASK_RESET_DEFAULTS.items():
        if isinstance(default_value, (dict, list)):
            patch[key] = deepcopy(default_value)
        else:
            patch[key] = default_value
    return patch

# ==========================================
# 1. UTILITIES
# ==========================================

def check_tester_service() -> bool:
    """
    Checks if the Tester FastAPI service is reachable.
    """
    try:
        response = requests.get(TESTER_HEALTH_URL, timeout=2.0)
        is_up = response.status_code == 200
        if is_up:
            logger.info(f"✅ Tester Service ONLINE at {TESTER_HEALTH_URL}")
        else:
            logger.warning(f"⚠️ Tester Service returned status {response.status_code}")
        return is_up
    except Exception as e:
        logger.warning(f"⚠️ Tester Service unreachable: {e}")
        return False

# ==========================================
# 2. CORE NODES
# ==========================================

def router_node(llm, state: AgentState) -> dict:
    """
    Decides the execution path (Generator vs Planner) and whether to run tests.
    """
    
    # 1. Extract the latest task
    last_msg = state["messages"][-1] if state.get("messages") else None
    task = (getattr(last_msg, "content", "") or "").strip()

    reset_patch: dict[str, Any] = {}
    if _should_reset_task_state(task, state):
        logger.info("♻️ Router: detected new task intent. Clearing previous artifacts.")
        reset_patch = _build_task_reset_patch()

    # 2. Check for Planner Loop (Awaiting Approval)
    if state.get("awaiting_approval", False):
        logger.info("🔄 Router: Returning to Planner (Awaiting Approval)")
        return {
            **reset_patch,
            "route": "planner",
            "task": task,
            "planner_used": True,
        }

    # 3. Check for Explicit User Overrides (User explicitly asking for tests)
    lower_task = task.lower()
    explicit_test_keywords = ["test", "verify", "check", "validate", "debug", "prove"]
    user_explicitly_wants_tests = any(w in lower_task for w in explicit_test_keywords)

    # 4. Construct System Prompt

    try:
        # Invoke LLM using HumanMessage for the task
        resp = llm.invoke([
            SystemMessage(content=ROUTER_INSTRUCTIONS), 
            HumanMessage(content=task)
        ])
        
        # Clean and Parse JSON
        raw = str(resp.content).strip()
        if "```" in raw:
            match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if match: 
                raw = match.group(1).strip()
        
        data = json.loads(raw)
        route = data.get("route", "generator")
        llm_thinks_test_needed = data.get("run_tests", False)
        
        logger.info(f"🧭 Router Decision: Route='{route}' | LLM wants tests={llm_thinks_test_needed}")

    except Exception as e:
        logger.warning(f"⚠️ Router Fallback (Error: {e}). Defaulting to 'generator'.")
        route = "generator"
        llm_thinks_test_needed = False

    # 5. Final Decision Logic
    # User intent overrides LLM conservatism
    final_run_tests = True if user_explicitly_wants_tests else llm_thinks_test_needed

    return {
        **reset_patch,
        "route": route, 
        "task": task, 
        "planner_used": (route == "planner"),
        "run_tests": final_run_tests
    }

def tester_node(state: AgentState) -> dict:
    """
    Synchronous call to the Tester FastAPI server.
    Blocking call with NO TIMEOUT to allow local LLMs to finish.
    """
    code = (state.get("assembled_code") or state.get("code") or "").strip()
    task = (state.get("original_task") or state.get("task") or "").strip()

    if not code:
        logger.warning("⚠️ Tester called but no code found in state.")
        return {"tester_error": "No code found to test."}

    logger.info(f"🚀 Calling Tester Service at {TESTER_URL}...")
    
    try:
        # TIMEOUT set to None to allow infinite wait for local models
        resp = requests.post(
            TESTER_URL, 
            json={"task": task, "code": code}, 
            timeout=None 
        )
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("error"):
                logger.error(f"Tester logical error: {data['error']}")
                return {"tester_error": data["error"]}
            
            logger.info("✅ Tester results received.")
            return {
                "tests": data.get("tests", ""),
                "test_result": data.get("result", {})
            }
        else:
            err = f"HTTP Error {resp.status_code}: {resp.text}"
            logger.error(err)
            return {"tester_error": err}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Tester connection failed: {e}")
        return {"tester_error": str(e)}

# ==========================================
# 3. CONDITIONAL EDGES
# ==========================================

def pick_route(state: AgentState) -> str:
    return state.get("route", "generator")

def after_generator(state: AgentState) -> str:
    code = state.get("code") or state.get("assembled_code")
    run_tests = state.get("run_tests", False)
    
    # Check if we have valid code AND tests are required
    if code and len(str(code).strip()) > 5 and run_tests:
        logger.info("👉 Routing to Tester (Code exists & run_tests=True)")
        return "tester"
    
    logger.info("👉 Routing to Evaluator (Skipping Tester)")
    return "debugger_evaluator"

def after_planner(state: AgentState) -> str:
    if state.get("awaiting_approval"):
        return "awaiting_approval"
    return "integrator"

# ==========================================
# 4. APP BUILDER
# ==========================================

def build_app():
    llm = build_llm()
    
    # Build Subgraphs
    generator = build_generator_subgraph(llm)
    integrator = build_integrator_subgraph(llm)
    planner = build_planner_subgraph(llm, generator)
    debugger_evaluator = build_debugger_evaluator_subgraph(llm)

    g = StateGraph(AgentState)
    
    # Add Nodes
    g.add_node("orchestrator", partial(router_node, llm))
    g.add_node("generator", generator)
    g.add_node("planner", planner)
    g.add_node("integrator", integrator)
    g.add_node("tester", tester_node)
    g.add_node("debugger_evaluator", debugger_evaluator)

    # Add Edges
    g.add_edge(START, "orchestrator")
    
    # Orchestrator Decisions
    g.add_conditional_edges("orchestrator", pick_route, {
        "generator": "generator", 
        "planner": "planner", 
        "other": END
    })
    
    # Generator Post-Processing
    g.add_conditional_edges("generator", after_generator, {
        "tester": "tester", 
        "debugger_evaluator": "debugger_evaluator"
    })
    
    # Planner Flow
    g.add_conditional_edges("planner", after_planner, {
        "integrator": "integrator", 
        "awaiting_approval": END
    })
    
    # Integrator always goes to Tester (if part of planner flow, usually extensive)
    # Note: You might want to apply the same logic as 'after_generator' here if strict,
    # but usually integrated multi-file code *should* be tested.
    g.add_edge("integrator", "tester")
    
    # Tester always goes to Evaluator
    g.add_edge("tester", "debugger_evaluator")
    
    # Evaluator is the end
    g.add_edge("debugger_evaluator", END)

    return g.compile(checkpointer=MemorySaver())