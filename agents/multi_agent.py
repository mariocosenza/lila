import os

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_ollama import ChatOllama


def last_text(result: dict) -> str:
    return result["messages"][-1].text


def build_llm() -> ChatOllama:
    model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        reasoning=True,  
    )


def build_agents(llm: ChatOllama):
    planner = create_agent(
        llm,
        tools=[],
        system_prompt=(
            "You are PLANNER. Produce a short, actionable plan with numbered steps. "
            "Keep it concise. If requirements are missing, list assumptions."
        ),
    )

    coder = create_agent(
        llm,
        tools=[],
        system_prompt=(
            "You are CODER. Write correct, minimal code for the plan. "
            "Prefer clean architecture and add brief comments. Output only what is needed."
        ),
    )

    tester = create_agent(
        llm,
        tools=[],
        system_prompt=(
            "You are TESTER. Create tests (unit/integration as appropriate) "
            "and edge cases. If code is missing, propose test strategy and skeleton tests."
        ),
    )

    validator = create_agent(
        llm,
        tools=[],
        system_prompt=(
            "You are VALIDATION. Review plan/code/tests for correctness, missing cases, "
            "security pitfalls, and inconsistencies. Return a short checklist + fixes."
        ),
    )

    @tool
    def plan(request: str) -> str:
        return last_text(planner.invoke({"messages": [{"role": "user", "content": request}]}))

    @tool
    def code(request: str) -> str:
        return last_text(coder.invoke({"messages": [{"role": "user", "content": request}]}))

    @tool
    def test(request: str) -> str:
        return last_text(tester.invoke({"messages": [{"role": "user", "content": request}]}))

    @tool
    def validate(request: str) -> str:
        return last_text(validator.invoke({"messages": [{"role": "user", "content": request}]}))

    orchestrator = create_agent(
        llm,
        tools=[plan, code, test, validate],
        system_prompt=(
            "You are ORCHESTRATOR. "
            "For each user request, you MAY call tools in this typical order: "
            "plan -> code -> test -> validate. "
            "Call only what is needed. Then output a final integrated answer."
        ),
    )

    return orchestrator

