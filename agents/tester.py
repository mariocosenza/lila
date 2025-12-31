from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# --- External Dependencies ---
# pip install tenacity langchain-google-genai langgraph langchain-core pydantic

# Tenacity for smart retries
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    RetryCallState
)
from tenacity.wait import wait_base
from google.api_core.exceptions import ResourceExhausted

from langchain.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- Internal Imports ---
# These must exist in your project structure
from client import grammo_test_mcp
from integrator import GRAMMO_LARK_SPEC

# --- Configuration ---
logger = logging.getLogger("tester_debugger")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


# ==========================================
# 1. RETRY & MODEL WRAPPER LOGIC
# ==========================================

def _is_retryable_error(exception: Exception) -> bool:
    """Check if the exception is a Google API quota error."""
    if isinstance(exception, ResourceExhausted):
        return True
    if hasattr(exception, "__cause__") and isinstance(exception.__cause__, ResourceExhausted):
        return True
    msg = str(exception)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return True
    return False


def _extract_retry_delay(exception: Exception, default_delay: float = None) -> Optional[float]:
    """Parse the error message to see if Google suggests a specific wait time."""
    messages_to_check = [str(exception)]
    if hasattr(exception, "__cause__") and exception.__cause__:
        messages_to_check.append(str(exception.__cause__))
    
    pattern = r"Please retry in ([0-9\.]+)s"
    for msg in messages_to_check:
        match = re.search(pattern, msg)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return default_delay


class wait_dynamic_gemini(wait_base):
    """Custom wait strategy that respects Google's 'retry in X seconds' headers."""
    def __init__(self, fallback_wait: wait_base):
        self.fallback = fallback_wait

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception()
        extracted_delay = _extract_retry_delay(exc)
        if extracted_delay is not None:
            return extracted_delay + 1.0  # Add 1s buffer
        return self.fallback(retry_state)


def _log_retry(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    wait = retry_state.next_action.sleep
    attempt = retry_state.attempt_number
    logger.debug(f"Caught {type(exception).__name__} (Attempt {attempt}). Retrying in {wait:.2f}s...")


class _GeminiSafeWrapper:
    """
    Wraps ChatGoogleGenerativeAI to:
    1. Handle 429/ResourceExhausted retries intelligently.
    2. Handle Gemma models (flatten SystemMessage, simulate tools).
    """
    def __init__(self, llm: Any):
        self._llm = llm

    def _sanitize_messages(self, inp: Any) -> Any:
        if not isinstance(inp, list):
            return inp
            
        valid_msgs = [m for m in inp if isinstance(getattr(m, "content", ""), str) and m.content.strip()]
        
        # Handle Gemma: Merge SystemMessage into first HumanMessage
        # Gemma API often rejects separate SystemMessages or complex tool bindings
        model_name = getattr(self._llm, "model", "").lower()
        if "gemma" not in model_name:
            return valid_msgs
        
        # Extract and merge system messages for Gemma
        system_content = []
        chat_msgs = []
        for m in valid_msgs:
            if isinstance(m, SystemMessage):
                system_content.append(m.content)
            else:
                chat_msgs.append(m)
        
        if not system_content:
            return valid_msgs
        
        merged_sys = "\n\n".join(system_content)
        if chat_msgs and isinstance(chat_msgs[0], HumanMessage):
            original = chat_msgs[0]
            new_first = HumanMessage(
                content=f"Instructions:\n{merged_sys}\n\nQuery:\n{original.content}",
                additional_kwargs=original.additional_kwargs,
                response_metadata=original.response_metadata
            )
            chat_msgs[0] = new_first
        else:
            chat_msgs.insert(0, HumanMessage(content=merged_sys))
        return chat_msgs

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_dynamic_gemini(fallback_wait=wait_exponential(multiplier=2, min=2, max=60)),
        stop=stop_after_attempt(12),
        before_sleep=_log_retry
    )
    def invoke(self, input: Any, **kwargs: Any) -> Any:
        return self._llm.invoke(self._sanitize_messages(input), **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_GeminiSafeWrapper":
        # For Gemma, SKIP native binding to avoid 400 Errors.
        # We will handle tool injection via prompt engineering in `tester_generate`.
        model_name = getattr(self._llm, "model", "").lower()
        if "gemma" in model_name:
            return self
        
        bound = self._llm.bind_tools(tools, **kwargs)
        return _GeminiSafeWrapper(bound)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


# ==========================================
# 2. TEST EXECUTION TOOLS
# ==========================================

class RunTestsInput(BaseModel):
    code: str = Field(description="Grammo program source code")
    tests: str = Field(description="Test plan or test cases")


def _run_coro_sync(coro):
    """
    Safely run an async coroutine from a synchronous context.
    Creates a new event loop in a daemon thread to avoid conflicts with existing loops.
    """
    try:
        # Check if we are already in a loop (likely yes if running in LangGraph async)
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running, run directly
        return asyncio.run(coro)
    else:
        # Loop is running. Spawn a thread with a new loop.
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
    try:
        # Calls the client MCP logic
        result = _run_coro_sync(grammo_test_mcp(code, tests))
        
        if not isinstance(result, dict):
            return {"passed": False, "stdout": "", "stderr": str(result)}
        
        return {
            "passed": bool(result.get("passed", False)),
            "stdout": str(result.get("stdout", "") or ""),
            "stderr": str(result.get("stderr", "") or ""),
        }
    except Exception as e:
        return {"passed": False, "stdout": "", "stderr": f"Execution Error: {str(e)}"}


TOOLS = [run_grammo_tests]


# ==========================================
# 3. LANGGRAPH STATE & NODES
# ==========================================

class TesterState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    code: str
    original_code: str
    tests: str
    test_result: Dict[str, Any]
    test_attempts: int
    max_test_attempts: int


TESTER_SYSTEM = SystemMessage(
    content=(
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
)


@dataclass(frozen=True)
class TesterContext:
    llm_with_tools: object


def build_gemini_llm(model: str = "gemini-2.0-flash-exp") -> object:
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("Missing dependency: langchain-google-genai. Install it and set GOOGLE_API_KEY.")
    
    # Initialize base model
    base_llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=0)
    # Wrap it for safety
    return _GeminiSafeWrapper(base_llm)


def _init_original_code(state: TesterState) -> Dict:
    """Snapshot the original code before testing begins."""
    if state.get("original_code"):
        return {}
    code = (state.get("code") or "").strip()
    return {
        "original_code": code, 
        "max_test_attempts": int(state.get("max_test_attempts", 3)), 
        "test_attempts": 0
    }


def _build_initial_test_prompt(code: str) -> str:
    """Build the initial prompt for test generation."""
    return (
        "GRAMMO CODE:\n"
        f"{code}\n\n"
        "Generate comprehensive tests and call `run_grammo_tests`."
    )


def _build_debug_test_prompt(last_result: Dict[str, Any], current_tests: str, code: str) -> str:
    """Build the debug prompt when tests fail."""
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


def _add_gemma_tool_instruction(prompt_text: str) -> str:
    """Append Gemma-specific tool calling instructions."""
    return prompt_text + (
        "\n\nIMPORTANT: You cannot use native tool calls.\n"
        "To run tests, you MUST output a JSON object in a markdown code block:\n"
        "```json\n"
        "{\n"
        '  "action": "run_grammo_tests",\n'
        '  "args": {\n'
        '    "code": "FULL_CODE_HERE",\n'
        '    "tests": "TEST_CASES_HERE"\n'
        "  }\n"
        "}\n"
        "```"
    )


def _inject_synthetic_tool_call(ai: AIMessage, content: str) -> None:
    """Parse JSON from AI response and inject synthetic tool call for Gemma."""
    if ai.tool_calls:
        return  # Already has tool calls
    
    # Regex to find JSON block
    match = re.search(r"```json(.*?)```", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            if data.get("action") == "run_grammo_tests" and "args" in data:
                ai.tool_calls = [{
                    "name": "run_grammo_tests",
                    "args": data["args"],
                    "id": str(uuid.uuid4())
                }]
        except Exception:
            pass  # If parsing fails, we treat it as text response


def tester_generate(ctx: TesterContext, state: TesterState) -> Dict:
    """Generates tests, or patches code and re-tests."""
    code = (state.get("code") or "").strip()
    current_tests = (state.get("tests") or "").strip()
    attempts = int(state.get("test_attempts", 0))
    last_result = state.get("test_result") or {}

    msgs = state.get("messages", [])
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [TESTER_SYSTEM, *msgs]

    # Detect if we are using Gemma
    model_name = getattr(ctx.llm_with_tools, "model", "").lower()
    is_gemma = "gemma" in model_name

    # Prompt construction
    if attempts == 0:
        prompt_text = _build_initial_test_prompt(code)
    else:
        prompt_text = _build_debug_test_prompt(last_result, current_tests, code)

    # Gemma Handling: Instruct it to use JSON for tools since native is disabled
    if is_gemma:
        prompt_text = _add_gemma_tool_instruction(prompt_text)

    ai: AIMessage = ctx.llm_with_tools.invoke([*msgs, HumanMessage(content=prompt_text)])

    # Gemma Post-Processing: Parse JSON and inject synthetic ToolCalls
    if is_gemma:
        _inject_synthetic_tool_call(ai, ai.content)

    return {"messages": [ai]}


def tester_route(state: TesterState) -> Literal["tools", "__end__"]:
    """Check if the model wants to call a tool."""
    msgs = state.get("messages", [])
    if not msgs:
        return "__end__"
    last = msgs[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "__end__"


def _parse_tool_message_result(tool_message: ToolMessage) -> Dict[str, Any]:
    """Extract and parse result from ToolMessage."""
    result: Dict[str, Any] = {}
    if isinstance(tool_message.content, str):
        try:
            result = json.loads(tool_message.content) if tool_message.content.strip().startswith("{") else {"stderr": tool_message.content}
        except Exception:
            result = {"stderr": tool_message.content}
    elif isinstance(tool_message.content, dict):
        result = tool_message.content
    return result


def _extract_tool_result_from_messages(msgs: List[BaseMessage]) -> Dict[str, Any]:
    """Find the last ToolMessage result for grammo tests."""
    for m in reversed(msgs):
        if isinstance(m, ToolMessage) and m.name == "run_grammo_tests":
            return _parse_tool_message_result(m)
    return {}


def _extract_ai_tool_args(msgs: List[BaseMessage]) -> Dict[str, str]:
    """Extract code and tests from last AI message's tool call args."""
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

    updates: Dict[str, Any] = {
        "test_result": result,
        "test_attempts": attempts,
    }

    # Retrieve the code and tests that were actually used in this attempt
    args = _extract_ai_tool_args(msgs)
    updates.update(args)

    # FAIL-SAFE: If we exhausted attempts and are still failing, restore original code.
    if (not passed) and (attempts >= max_attempts):
        logger.warning("Max attempts reached. Reverting to original code.")
        updates["code"] = (state.get("original_code") or state.get("code") or "").strip()

    return updates


def tester_after_collect(state: TesterState) -> Literal["generate", "__end__"]:
    """Decide next step based on test results."""
    result = state.get("test_result") or {}
    passed = bool(result.get("passed", False))
    attempts = int(state.get("test_attempts", 0))
    max_attempts = int(state.get("max_test_attempts", 5))

    if passed:
        return "__end__"
    if attempts >= max_attempts:
        return "__end__"
    return "generate"


def build_tester_graph(gemini_model: str = "gemma-3-27b-it"):
    """Compiles the LangGraph state machine."""
    llm = build_gemini_llm(model=gemini_model)
    
    # Bind tools (Note: internal logic skips binding for Gemma)
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

    return g.compile()