from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from multi_agent import AgentState


def _parse_plan_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        s = s.lstrip("-•").strip()
        if len(s) >= 3 and s[:2].isdigit() and s[2:3] in [".", ")"]:
            s = s[3:].strip()
        lines.append(s)
    return lines[:6]


def _is_yes(s: str) -> bool:
    s = (s or "").strip().lower()
    yes = {"si", "sì", "ok", "va bene", "procedi", "confermo", "yes", "y"}
    return any(tok == s or tok in s for tok in yes)


def _is_no(s: str) -> bool:
    s = (s or "").strip().lower()
    no = {"no", "n", "non va bene", "cambia", "modifica", "nope"}
    return any(tok == s or tok in s for tok in no)


def _config_with_stream(config: Optional[RunnableConfig], stream_tokens: bool) -> RunnableConfig:
    base = dict(config or {})
    conf = dict(base.get("configurable") or {})
    conf["stream_tokens"] = stream_tokens
    base["configurable"] = conf
    return base


def build_planner_subgraph(llm, coder_subgraph):
    g = StateGraph(AgentState)

    def entry(state: AgentState) -> Dict:
        return {}

    def entry_route(state: AgentState) -> str:
        return "handle_approval" if state.get("awaiting_approval", False) else "make_plan"

    def make_plan(state: AgentState) -> Dict:
        original = (state.get("task") or "").strip()

        sys = SystemMessage(
            content=(
                "You are PLANNER.\n"
                "Split the request into 2-5 concrete coding subtasks.\n"
                "Return one subtask per line. No commentary.\n"
            )
        )
        resp = llm.invoke([sys, SystemMessage(content=original)])
        plan = _parse_plan_lines(resp.content) or [original]

        return {
            "original_task": original,
            "plan": plan,
            "plan_step": 0,
        }

    def ask_approval(state: AgentState) -> Dict:
        plan = state.get("plan") or []
        lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(plan))
        msg = (
            "Ecco il piano proposto:\n"
            f"{lines}\n\n"
            "Va bene? Rispondi con **sì** per procedere, oppure **no** e dimmi cosa cambiare."
        )
        return {"awaiting_approval": True, "messages": [AIMessage(content=msg)]}

    def handle_approval(state: AgentState) -> Dict:
        answer = (state.get("task") or "").strip()

        if _is_yes(answer):
            return {
                "awaiting_approval": False,
                "messages": [AIMessage(content="Ok, procedo col piano.")],
            }

        if _is_no(answer):
            return {
                "awaiting_approval": False,
                "messages": [AIMessage(content="Ok, aggiorno il piano in base al feedback.")],
            }

        return {
            "awaiting_approval": True,
            "messages": [AIMessage(content="Non ho capito: confermi il piano? (sì/no + eventuali modifiche)")],
        }

    def after_handle_route(state: AgentState) -> str:
        if state.get("awaiting_approval", False):
            return "stop"

        ans = (state.get("task") or "").strip()
        if _is_no(ans):
            return "revise_plan"
        return "execute"

    def revise_plan(state: AgentState) -> Dict:
        original = (state.get("original_task") or "").strip()
        feedback = (state.get("task") or "").strip()

        sys = SystemMessage(
            content=(
                "You are PLANNER.\n"
                "Revise the plan based on user feedback.\n"
                "Return 2-5 concrete subtasks, one per line. No commentary.\n"
            )
        )
        prompt = f"Original request:\n{original}\n\nUser feedback:\n{feedback}\n"
        resp = llm.invoke([sys, SystemMessage(content=prompt)])
        plan = _parse_plan_lines(resp.content) or [original]

        return {"plan": plan, "plan_step": 0}

    # ---- Execution ----
    def set_next_subtask(state: AgentState) -> Dict:
        plan = state.get("plan") or []
        i = int(state.get("plan_step") or 0)
        subtask = plan[i] if i < len(plan) else ""
        return {"task": subtask}

    def advance(state: AgentState) -> Dict:
        return {"plan_step": int(state.get("plan_step") or 0) + 1}

    def should_continue(state: AgentState) -> str:
        plan = state.get("plan") or []
        i = int(state.get("plan_step") or 0)
        return "loop" if i < len(plan) else "integrate"

    def integrate_request(state: AgentState) -> Dict:
        return {
            "task": (
                "Rewrite main.py as the final integrated solution.\n"
                "Return ONLY the full final code.\n"
                "Avoid meta-comments (subtask/part/step). Keep comments minimal and only code-related."
            )
        }

    # Nodes
    g.add_node("entry", entry)
    g.add_node("make_plan", make_plan)
    g.add_node("ask_approval", ask_approval)

    g.add_node("handle_approval", handle_approval)
    g.add_node("revise_plan", revise_plan)

    g.add_node("set_next_subtask", set_next_subtask)

    # We call coder multiple times; we will control streaming via config in wrapper nodes:
    def coder_no_stream(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
        # no streaming on intermediate subtasks
        return coder_subgraph.invoke(state, config=_config_with_stream(config, False))

    def coder_stream(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict:
        # stream only the final integrated code (or direct coder route)
        return coder_subgraph.invoke(state, config=_config_with_stream(config, True))

    g.add_node("coder_no_stream", coder_no_stream)
    g.add_node("advance", advance)
    g.add_node("integrate_request", integrate_request)
    g.add_node("coder_stream", coder_stream)

    # Edges
    g.add_edge(START, "entry")
    g.add_conditional_edges("entry", entry_route, {
        "make_plan": "make_plan",
        "handle_approval": "handle_approval",
    })

    # First turn: make plan -> ask approval -> stop
    g.add_edge("make_plan", "ask_approval")
    g.add_edge("ask_approval", END)

    # Approval handling
    g.add_conditional_edges("handle_approval", after_handle_route, {
        "stop": END,
        "revise_plan": "revise_plan",
        "execute": "set_next_subtask",
    })

    # If NO: revise -> ask again -> stop
    g.add_edge("revise_plan", "ask_approval")
    g.add_edge("ask_approval", END)

    # Execute plan: subtask loop (no stream)
    g.add_edge("set_next_subtask", "coder_no_stream")
    g.add_edge("coder_no_stream", "advance")
    g.add_conditional_edges("advance", should_continue, {
        "loop": "set_next_subtask",
        "integrate": "integrate_request",
    })

    # Final integration: stream ON
    g.add_edge("integrate_request", "coder_stream")
    g.add_edge("coder_stream", END)

    return g.compile()
