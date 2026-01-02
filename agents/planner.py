from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Union

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from multi_agent import AgentState

# --- Helper Functions ---

def _clean_json_text(text: str) -> str:
    """Cleans code fences and other noise from JSON output."""
    text = text.strip()
    # Remove markdown code blocks if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text

def _try_json_parse_list(text: str) -> List[str]:
    """Attempt to parse JSON as list or dict with 'steps'."""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data if x]
        if isinstance(data, dict) and "steps" in data:
            return [str(x) for x in data["steps"] if x]
    except json.JSONDecodeError:
        pass
    return []


def _parse_numbered_list(text: str) -> List[str]:
    """Fallback: parse numbered/bulleted list from text."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove "1.", "-", "*", etc.
        line = re.sub(r"^[\d\.\-\*\•\)]+\s*", "", line)
        if len(line) > 5:  # Filter out noise
            lines.append(line)
    return lines[:10]  # Cap at 10 steps


def _parse_plan_json(text: Union[str, list]) -> List[str]:
    """Robustly parses the plan from LLM output."""
    # 1. Handle weird list outputs from some adapters
    if isinstance(text, list):
        raw = ""
        for item in text:
            if isinstance(item, str):
                raw += item
            elif isinstance(item, dict):
                raw += item.get("text", "")
        text = raw
    
    text = _clean_json_text(str(text))
    
    # 2. Try JSON parsing first
    plan = _try_json_parse_list(text)
    if plan:
        return plan

    # 3. Fallback: Line parsing (if model gave a numbered list instead of JSON)
    return _parse_numbered_list(text)

def _is_yes(s: str) -> bool:
    s = (s or "").strip().lower()
    yes = {"si", "sì", "ok", "va bene", "procedi", "confermo", "yes", "y", "go", "sure"}
    return any(tok == s or tok in s for tok in yes)

def _is_no(s: str) -> bool:
    s = (s or "").strip().lower()
    no = {"no", "n", "non va bene", "cambia", "modifica", "nope", "not yet", "wrong"}
    return any(tok == s or tok in s for tok in no)

def _config_with_stream(config: Optional[RunnableConfig], stream_tokens: bool) -> RunnableConfig:
    base = dict(config or {})
    conf = dict(base.get("configurable") or {})
    conf["stream_tokens"] = stream_tokens
    base["configurable"] = conf
    return base

# --- Planner Node Definitions ---

def _create_make_plan_node(llm):
    """Factory for make_plan node."""
    def make_plan(state: AgentState) -> Dict:
        original = (state.get("task") or "").strip()

        # Strong System Prompt with Examples (Few-Shot)
        system_prompt = (
            "You are the **Technical Lead**. Your job is to break down a user request into small, implementation-ready coding steps.\n\n"
            "### RULES:\n"
            "1. **DECOMPOSE**: Do not just repeat the task. Split it into 3-6 logical phases.\n"
            "2. **FORMAT**: Return ONLY a raw JSON list of strings.\n"
            "3. **STYLE**: Steps must be instructions for a developer (e.g., 'Create struct', 'Implement logic').\n\n"
            "### EXAMPLES:\n"
            "User: 'Create a Snake game'\n"
            "You: [\n"
            "  \"Define the Grid and Snake data structures\",\n"
            "  \"Implement the movement logic and input handling\",\n"
            "  \"Implement collision detection and score tracking\",\n"
            "  \"Create the main game loop and rendering\"\n"
            "]\n\n"
            "User: 'Write a factorial function'\n"
            "You: [\n"
            "  \"Define the factorial function with integer input\",\n"
            "  \"Handle edge cases (0 and negative numbers)\",\n"
            "  \"Implement recursive or iterative calculation\"\n"
            "]"
        )

        resp = llm.invoke([SystemMessage(content=system_prompt), SystemMessage(content=original)])
        plan = _parse_plan_json(resp.content)
        
        # Safety Check: If plan is still empty or identical to input (Lazy Model), force split
        if not plan or (len(plan) == 1 and len(plan[0]) > len(original) * 0.8):
            if "game" in original.lower():
                plan = [
                    f"Define data structures for {original}",
                    "Implement core game logic and rules",
                    "Implement user input and main loop"
                ]
            else:
                plan = [original]

        return {
            "original_task": original,
            "plan": plan,
            "plan_step": 0,
        }
    return make_plan


def _create_revise_plan_node(llm):
    """Factory for revise_plan node."""
    def revise_plan(state: AgentState) -> Dict:
        original = (state.get("original_task") or "").strip()
        feedback = (state.get("task") or "").strip()

        sys = SystemMessage(
            content=(
                "You are the Technical Lead. The user rejected the previous plan.\n"
                "Create a NEW plan considering the user's feedback.\n"
                "Return ONLY a JSON list of strings."
            )
        )
        prompt = f"Original Request: {original}\nFeedback: {feedback}"
        
        resp = llm.invoke([sys, SystemMessage(content=prompt)])
        plan = _parse_plan_json(resp.content)
        
        if not plan:
            plan = [original]

        return {"plan": plan, "plan_step": 0}
    return revise_plan


# --- Planner Subgraph ---

# --- Planner Node Helper Functions ---

def _add_planner_nodes(g: StateGraph, make_plan, revise_plan_node, generator_subgraph, ask_approval, handle_approval, set_next_subtask, advance) -> None:
    """Add all nodes to the planner graph."""
    g.add_node("entry", lambda state: {})
    g.add_node("make_plan", make_plan)
    g.add_node("ask_approval", ask_approval)
    g.add_node("handle_approval", handle_approval)
    g.add_node("revise_plan", revise_plan_node)
    g.add_node("set_next_subtask", set_next_subtask)
    g.add_node("advance", advance)
    
    # Generator Integration
    def generator_no_stream(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
        return generator_subgraph.invoke(state, config=_config_with_stream(config, False))
    g.add_node("generator_no_stream", generator_no_stream)


def _add_planner_edges(g: StateGraph) -> None:
    """Add all edges to the planner graph."""
    def entry_route(state: AgentState) -> str:
        return "handle_approval" if state.get("awaiting_approval", False) else "make_plan"
    
    def after_handle_route(state: AgentState) -> str:
        if state.get("awaiting_approval", False):
            return "stop"
        ans = (state.get("task") or "").strip()
        return "revise_plan" if _is_no(ans) else "execute"
    
    def should_continue(state: AgentState) -> str:
        plan = state.get("plan") or []
        i = int(state.get("plan_step") or 0)
        return "loop" if i < len(plan) else "end"
    
    g.add_edge(START, "entry")
    g.add_conditional_edges("entry", entry_route, {
        "make_plan": "make_plan",
        "handle_approval": "handle_approval",
    })

    g.add_edge("make_plan", "ask_approval")
    g.add_edge("ask_approval", END)

    g.add_conditional_edges("handle_approval", after_handle_route, {
        "stop": END,
        "revise_plan": "revise_plan",
        "execute": "set_next_subtask",
    })

    g.add_edge("revise_plan", "ask_approval")
    g.add_edge("set_next_subtask", "generator_no_stream")
    g.add_edge("generator_no_stream", "advance")
    g.add_conditional_edges("advance", should_continue, {
        "loop": "set_next_subtask",
        "end": END,
    })


def build_planner_subgraph(llm, generator_subgraph):
    g = StateGraph(AgentState)

    # --- Node Functions (using factories) ---
    make_plan = _create_make_plan_node(llm)
    revise_plan_node = _create_revise_plan_node(llm)

    def ask_approval(state: AgentState) -> Dict:
        plan = state.get("plan") or []
        lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(plan))
        msg = (
            "Here is the proposed plan:\n"
            f"{lines}\n\n"
            "Is this okay? Reply with **yes** to proceed, or **no** and tell me what to change."
        )
        return {"awaiting_approval": True, "messages": [AIMessage(content=msg)]}

    def handle_approval(state: AgentState) -> Dict:
        answer = (state.get("task") or "").strip()

        if _is_yes(answer):
            return {
                "awaiting_approval": False,
                "messages": [AIMessage(content="Great. Starting step-by-step implementation.")],
            }

        if _is_no(answer):
            return {
                "awaiting_approval": False,
                "messages": [AIMessage(content="Received. Modifying the plan based on your feedback.")],
            }

        return {
            "awaiting_approval": True,
            "messages": [AIMessage(content="Please reply with 'yes' to confirm or 'no' to modify.")],
        }

    # --- Execution Logic ---

    def set_next_subtask(state: AgentState) -> Dict:
        plan = state.get("plan") or []
        i = int(state.get("plan_step") or 0)
        subtask = plan[i] if 0 <= i < len(plan) else "FINISH"
        return {"task": subtask}

    def advance(state: AgentState) -> Dict:
        return {"plan_step": int(state.get("plan_step") or 0) + 1}

    # --- Graph Construction ---
    _add_planner_nodes(g, make_plan, revise_plan_node, generator_subgraph, ask_approval, handle_approval, set_next_subtask, advance)
    _add_planner_edges(g)

    return g.compile()