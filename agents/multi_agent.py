import os
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    route: str  
    task: str
    original_task: str
    workspace: Dict[str, str]
    diagnostics: List[str]
    plan: List[str]
    plan_step: int
    awaiting_approval: bool


def build_llm() -> ChatOllama:
    model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        reasoning=True,
    )
