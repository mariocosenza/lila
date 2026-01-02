from __future__ import annotations

import asyncio
import json
import logging
import threading
from langchain.tools import tool
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, TypedDict, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from mcp_client import grammo_test_mcp
from integrator import GRAMMO_LARK_SPEC


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TesterAgent")


class RunTestsInput(BaseModel):
    code: str = Field(description="Grammo program source code")
    tests: str = Field(description="Test plan or test cases")

def _run_coro_sync(coro):
    """Safely run an async coroutine from a synchronous context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        res: Dict[str, Any] = {}
        def _target():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                res["value"] = new_loop.run_until_complete(coro)
            except Exception as e:
                res["error"] = e
            finally:
                try:
                    new_loop.close()
                except Exception:
                    pass
        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join()
        if "error" in res:
            raise res["error"]
        return res.get("value")

@tool("run_grammo_tests", args_schema=RunTestsInput)
def run_grammo_tests(code: str, tests: str) -> Dict[str, Any]:
    """Run Grammo tests against provided source code."""
    logger.info(f"🔧 [TOOL EXEC] Running tests... (Code len: {len(code)}, Tests len: {len(tests)})")
    try:
        # Calls the client MCP logic
        result = _run_coro_sync(grammo_test_mcp(code, tests))
        
        if not isinstance(result, dict):
            logger.error("❌ [TOOL ERROR] Invalid result format")
            return {"passed": False, "stdout": "", "stderr": str(result)}
        
        passed = bool(result.get("passed", False))
        status_icon = "✅" if passed else "❌"
        logger.info(f"{status_icon} [TOOL RESULT] Passed: {passed} | Stderr len: {len(str(result.get('stderr', '')))}")
        
        return {
            "passed": passed,
            "stdout": str(result.get("stdout", "") or ""),
            "stderr": str(result.get("stderr", "") or ""),
        }
    except Exception as e:
        logger.exception("❌ [TOOL EXCEPTION]")
        return {"passed": False, "stdout": "", "stderr": f"Execution Error: {str(e)}"}

TOOLS = [run_grammo_tests]


class TesterState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    code: str
    original_code: str
    tests: str
    test_result: Dict[str, Any]
    test_attempts: int
    max_test_attempts: int

TESTER_SYSTEM_CONTENT = (
    "You are TESTER, an expert debugger for the Grammo language.\n"
    "Goal: Create robust tests, run them, and ensure the code passes.\n\n"
    "### WORKFLOW\n"
    "1. **Initial Run:** Generate comprehensive test cases and run them against the input code.\n"
    "2. **Debug Loop:** If tests fail, analyze the `stderr` and `stdout`.\n"
    "   - **Is the Code broken?** -> Patch the Grammo code.\n"
    "   - **Are the Tests broken?** (e.g. syntax errors, wrong logic) -> Refine the tests.\n"
    "   - **Both?** -> Fix both.\n"
    "3. **Call Tool:** You MUST call `run_grammo_tests` with the (potentially updated) `code` and `tests` strings.\n\n"
    "### RULES\n"
    "- **ALWAYS** provide the full code and full tests in the tool arguments.\n"
    "- **Grammo Spec:** Follow the grammar below exactly.\n"
    "- **I/O Format:** Inputs in tests should mimic `>> \"Prompt\" # (var);` behavior.\n\n"
    "LARK SPECIFICATION:\n"
    f"{GRAMMO_LARK_SPEC}"
)

@dataclass(frozen=True)
class TesterContext:
    llm_with_tools: object

def build_ollama_llm(model: str = "gpt-oss-20b", base_url: str = "http://localhost:11434") -> object:
    logger.info(f"🔌 Connecting to Ollama: {base_url} (Model: {model})")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0
    )

def _init_original_code(state: TesterState) -> Dict:
    """Snapshot the original code before testing begins."""
    if state.get("original_code"):
        return {}
    code = (state.get("code") or "").strip()
    logger.info(f"🎬 [INIT] Starting session. Code length: {len(code)}")
    return {
        "original_code": code, 
        "max_test_attempts": int(state.get("max_test_attempts", 3)), 
        "test_attempts": 0
    }

def _build_initial_test_prompt(code: str) -> str:
    return (
        "GRAMMO CODE:\n"
        f"{code}\n\n"
        "Generate comprehensive tests and call `run_grammo_tests`."
    )

def _build_debug_test_prompt(last_result: Dict[str, Any], current_tests: str, code: str) -> str:
    return (
        "⚠️ PREVIOUS TESTS FAILED.\n"
        "Review the output below. Determine if the error is in the **Code** (logic bug) or the **Test** (syntax/format error).\n\n"
        f"RESULT:\n{json.dumps(last_result, indent=2)}\n\n"
        "INSTRUCTIONS:\n"
        "1. If `stderr` shows parsing errors in the test file -> **FIX THE TESTS**.\n"
        "2. If `stdout` shows assertion failures -> **FIX THE CODE** (or adjust tests if expectations were wrong).\n"
        "3. Call `run_grammo_tests` with the corrected full code and full tests.\n\n"
        "CURRENT CODE:\n"
        f"{code}\n\n"
        "CURRENT TESTS:\n"
        f"{current_tests}"
    )

def tester_generate(ctx: TesterContext, state: TesterState) -> Dict:
    """Generates tests, or patches code and re-tests."""
    attempts = int(state.get("test_attempts", 0))
    logger.info(f"🧠 [STEP: GENERATE] Attempt {attempts + 1}...")

    code = (state.get("code") or "").strip()
    current_tests = (state.get("tests") or "").strip()
    last_result = state.get("test_result") or {}

    msgs = state.get("messages", [])
    
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [SystemMessage(content=TESTER_SYSTEM_CONTENT), *msgs]

    if attempts == 0:
        prompt_text = _build_initial_test_prompt(code)
    else:
        logger.info("   ↳ Debugging failure from previous run.")
        prompt_text = _build_debug_test_prompt(last_result, current_tests, code)

    # Invoke Ollama
    response = ctx.llm_with_tools.invoke([*msgs, HumanMessage(content=prompt_text)])
    
    # Log concise summary of response
    if response.tool_calls:
        logger.info(f"   ↳ Generated Tool Call: {response.tool_calls[0]['name']}")
    else:
        logger.info("   ↳ Generated text response (No tool call).")

    return {"messages": [response]}

def tester_route(state: TesterState) -> Literal["tools", "__end__"]:
    """Check if the model wants to call a tool."""
    msgs = state.get("messages", [])
    if not msgs:
        return "__end__"
    last = msgs[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    logger.info("⏹️ [DECISION] No tool call detected. Ending graph.")
    return "__end__"

def _extract_tool_result_from_messages(msgs: List[BaseMessage]) -> Dict[str, Any]:
    for m in reversed(msgs):
        if isinstance(m, ToolMessage) and m.name == "run_grammo_tests":
            content = m.content
            if isinstance(content, dict):
                return content
            try:
                return json.loads(content)
            except Exception:
                return {"stderr": str(content)}
    return {}

def _extract_ai_tool_args(msgs: List[BaseMessage]) -> Dict[str, str]:
    args: Dict[str, str] = {}
    if msgs:
        last_ai_msg = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
        if last_ai_msg and last_ai_msg.tool_calls:
            tool_args = last_ai_msg.tool_calls[0].get("args", {})
            if "tests" in tool_args:
                args["tests"] = tool_args["tests"]
            if "code" in tool_args:
                args["code"] = tool_args["code"]
    return args

def tester_collect(state: TesterState) -> Dict:
    """Analyze tool output and update state code/tests."""
    msgs = state.get("messages", [])
    
    result = _extract_tool_result_from_messages(msgs)
    attempts = int(state.get("test_attempts", 0)) + 1
    max_attempts = int(state.get("max_test_attempts", 5))
    passed = bool(result.get("passed", False))

    logger.info(f"📝 [STEP: COLLECT] Analysis Complete. Passed: {passed} (Attempt {attempts}/{max_attempts})")

    updates: Dict[str, Any] = {
        "test_result": result,
        "test_attempts": attempts,
    }

    args = _extract_ai_tool_args(msgs)
    if args:
        logger.info("   ↳ Updating state with new code/tests from tool args.")
    updates.update(args)

    if (not passed) and (attempts >= max_attempts):
        logger.warning("⚠️ [MAX ATTEMPTS] Reverting to original code.")
        updates["code"] = (state.get("original_code") or state.get("code") or "").strip()

    return updates

def tester_after_collect(state: TesterState) -> Literal["generate", "__end__"]:
    """Decide next step based on test results."""
    result = state.get("test_result") or {}
    passed = bool(result.get("passed", False))
    attempts = int(state.get("test_attempts", 0))
    max_attempts = int(state.get("max_test_attempts", 5))

    if passed:
        logger.info("🎉 [SUCCESS] Tests passed. Finishing.")
        return "__end__"
    if attempts >= max_attempts:
        logger.info("🛑 [FAILURE] Max attempts reached. Finishing.")
        return "__end__"
    
    logger.info("🔄 [RETRY] Tests failed. Looping back to generator.")
    return "generate"

def build_tester_graph(ollama_model: str = "gpt-oss-20b", ollama_base_url: str = "http://localhost:11434"):
    """Compiles the LangGraph state machine using Ollama."""
    
    llm = build_ollama_llm(model=ollama_model, base_url=ollama_base_url)
    
    ctx = TesterContext(llm_with_tools=llm.bind_tools(TOOLS))

    g = StateGraph(TesterState)
    g.add_node("init", _init_original_code)
    g.add_node("generate", partial(tester_generate, ctx))
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("collect", tester_collect)

    g.add_edge(START, "init")
    g.add_edge("init", "generate")
    g.add_conditional_edges("generate", tester_route, {"tools": "tools", "__end__": END})
    g.add_edge("tools", "collect")
    g.add_conditional_edges("collect", tester_after_collect, {"generate": "generate", "__end__": END})

    logger.info(f"🚀 Graph Compiled. Ready for {ollama_model}.")
    return g.compile()