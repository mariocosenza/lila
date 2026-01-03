import os
import time
import logging
import re
from typing import Annotated, TypedDict, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,  
    RetryCallState
)
from tenacity.wait import wait_base

from google.api_core.exceptions import ResourceExhausted

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages

logger = logging.getLogger("gemini_retry")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class AgentState(TypedDict, total=False):
    """
    Shared state for all agents (Orchestrator, Generator, Planner, Tester, DebuggerEvaluator).
    """
    # Message History
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Task Routing & Planning
    route: str
    task: str
    original_task: str
    workspace: dict[str, str]
    diagnostics: list[str]
    plan: list[str]
    plan_step: int
    awaiting_approval: bool
    run_tests: bool  # Flag to force testing
    
    # --- Code Artifacts ---
    code: str           # Direct output from Generator
    assembled_code: str # Output from Integrator
    
    # --- Compilation State ---
    compile_attempts: int
    compile_result: dict[str, Any]
    compile_errors: list[str]
    
    # --- Testing State ---
    tests: str
    test_result: dict[str, Any]
    
    # --- Validation & Finalization ---
    validated_code: str
    validation_summary: str
    safety_notes: str
    
    # --- Loop Control ---
    iterations: int
    max_iters: int
    global_iterations: int
    max_global_iters: int


def _is_retryable_error(exception: Exception) -> bool:
    """
    Check if the exception is a ResourceExhausted error, 
    even if wrapped inside a LangChain exception.
    """
    # 1. Direct Google Exception
    if isinstance(exception, ResourceExhausted):
        return True
        
    # 2. Wrapped Exception (e.g. ChatGoogleGenerativeAIError)
    # Check the cause
    if hasattr(exception, "__cause__") and isinstance(exception.__cause__, ResourceExhausted):
        return True
        
    # 3. String fallback (if imports fail or wrapper is obscure)
    # The error message usually contains "RESOURCE_EXHAUSTED" or code 429
    msg = str(exception)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return True
        
    return False


def _extract_retry_delay(exception: Exception, default_delay: float = None) -> float | None:
    """
    Attempts to extract the requested retry delay from an exception 
    (or its underlying cause).
    """
    if not exception:
        return default_delay

    # Scan both the wrapper message and the original cause message
    messages_to_check = [str(exception)]
    if hasattr(exception, "__cause__") and exception.__cause__:
        messages_to_check.append(str(exception.__cause__))

    # Pattern for "Please retry in 22.1920s"
    pattern = r"Please retry in ([0-9\.]+)s"
    
    for msg in messages_to_check:
        match = re.search(pattern, msg)
        if match:
            try:
                val = float(match.group(1))
                return val
            except ValueError:
                pass

    return default_delay


class wait_dynamic_gemini(wait_base):
    """
    Custom tenacity wait strategy.
    1. Checks if the exception tells us exactly how long to wait.
    2. If yes, wait that duration + 1s buffer.
    3. If no, fall back to standard exponential backoff.
    """
    def __init__(self, fallback_wait: wait_base):
        self.fallback = fallback_wait

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception()
        
        # Try to find specific delay from Google
        extracted_delay = _extract_retry_delay(exc)
        
        if extracted_delay is not None:
            # Add a buffer (1s) to be safe
            actual_wait = extracted_delay + 1.0
            logger.debug(f"API requested wait of {extracted_delay}s. Sleeping for {actual_wait:.2f}s.")
            return actual_wait
            
        # Fallback to exponential
        return self.fallback(retry_state)


def _log_retry(retry_state: RetryCallState):
    """
    Callback for tenacity to log the exception and the wait time before the next attempt.
    """
    exception = retry_state.outcome.exception()
    wait = retry_state.next_action.sleep
    attempt = retry_state.attempt_number
    
    logger.debug(
        f"Caught {type(exception).__name__} (Attempt {attempt}). "
        f"Retrying in {wait:.2f}s..."
    )


class _GeminiSafeWrapper:
    """
    Wrap a LangChain chat model to ensure:
      1. Gemini never receives an empty `contents` payload.
      2. 429 errors (even wrapped ones) are retried with smart backoff.
      3. Gemma models (no system prompt support) have SystemMessages merged.
      4. Gemma models (no tool support) gracefully skip tool binding.
    """

    def __init__(self, llm: Any):
        self._llm = llm

    def _merge_system_for_gemma(self, valid_msgs: list[BaseMessage]) -> list[BaseMessage]:
        """Merge SystemMessage into first HumanMessage for Gemma models."""
        system_content = []
        chat_msgs = []
        
        for m in valid_msgs:
            if isinstance(m, SystemMessage):
                system_content.append(m.content)
            else:
                chat_msgs.append(m)
        
        if not system_content:
            return valid_msgs
        
        merged_system_text = "\n\n".join(system_content)
        
        if chat_msgs and isinstance(chat_msgs[0], HumanMessage):
            original_first = chat_msgs[0]
            new_first = HumanMessage(
                content=f"Instructions:\n{merged_system_text}\n\nQuery:\n{original_first.content}",
                additional_kwargs=original_first.additional_kwargs,
                response_metadata=original_first.response_metadata
            )
            chat_msgs[0] = new_first
        else:
            chat_msgs.insert(0, HumanMessage(content=merged_system_text))
        
        return chat_msgs

    def _sanitize_messages(self, inp: Any) -> Any:
        # PromptValue-like objects
        if hasattr(inp, "to_messages") and callable(getattr(inp, "to_messages")):
            try:
                inp = inp.to_messages()
            except Exception:
                return inp

        if not isinstance(inp, list):
            return inp

        # 1. Filter empty messages
        valid_msgs: list[BaseMessage] = []
        for m in inp:
            content = getattr(m, "content", None)
            if isinstance(content, str) and content.strip() == "":
                continue
            valid_msgs.append(m)

        # 2. Handle Gemma Specifics
        model_name = getattr(self._llm, "model", "").lower()
        if "gemma" in model_name:
            valid_msgs = self._merge_system_for_gemma(valid_msgs)

        fallback = os.getenv("GEMINI_FALLBACK_PROMPT", "Continue.")

        # 3. Final safety checks
        if not valid_msgs:
            return [HumanMessage(content=fallback)]

        has_non_system = any(not isinstance(m, SystemMessage) for m in valid_msgs)
        if not has_non_system:
            return valid_msgs + [HumanMessage(content=fallback)]

        return valid_msgs

    # ---- Retry Logic ----
    # We now use `retry_if_exception` with our custom checker to catch wrapped errors.

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_dynamic_gemini(
            fallback_wait=wait_exponential(multiplier=2, min=2, max=60)
        ),
        stop=stop_after_attempt(12), # Increased attempt count for longer TPM waits
        before_sleep=_log_retry
    )
    def _execute_with_retry(self, method_name: str, *args, **kwargs):
        method = getattr(self._llm, method_name)
        return method(*args, **kwargs)

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_dynamic_gemini(
            fallback_wait=wait_exponential(multiplier=2, min=2, max=60)
        ),
        stop=stop_after_attempt(12),
        before_sleep=_log_retry
    )
    async def _aexecute_with_retry(self, method_name: str, *args, **kwargs):
        method = getattr(self._llm, method_name)
        return await method(*args, **kwargs)

    # ---- Runnable / ChatModel interface methods (delegated) ----

    def invoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> Any:
        return self._execute_with_retry(
            "invoke", 
            self._sanitize_messages(input), 
            config=config, 
            **kwargs
        )

    async def ainvoke(self, input: Any, config: dict | None = None, **kwargs: Any) -> Any:
        return await self._aexecute_with_retry(
            "ainvoke", 
            self._sanitize_messages(input), 
            config=config, 
            **kwargs
        )

    def batch(self, inputs: list[Any], config: dict | None = None, **kwargs: Any) -> Any:
        sanitized = [self._sanitize_messages(i) for i in inputs]
        return self._execute_with_retry(
            "batch", 
            sanitized, 
            config=config, 
            **kwargs
        )

    async def abatch(self, inputs: list[Any], config: dict | None = None, **kwargs: Any) -> Any:
        sanitized = [self._sanitize_messages(i) for i in inputs]
        return await self._aexecute_with_retry(
            "abatch", 
            sanitized, 
            config=config, 
            **kwargs
        )

    def _get_retry_sleep_time(self, e: Exception) -> float:
        """Calculate sleep time for retry, preferring API-suggested delay."""
        delay = _extract_retry_delay(e)
        if delay:
            return delay + 1.0
        return min(60, 2 * (2 ** (getattr(self, "_stream_attempt", 1) - 1)))

    def stream(self, input: Any, config: dict | None = None, **kwargs: Any):
        attempt = 0
        max_attempts = 12
        
        while True:
            try:
                yield from self._llm.stream(self._sanitize_messages(input), config=config, **kwargs)
                break
            except Exception as e:
                if _is_retryable_error(e):
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    
                    sleep_time = self._get_retry_sleep_time(e)
                    logger.debug(f"Caught ResourceExhausted (Stream Attempt {attempt}). Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    raise

    async def astream(self, input: Any, config: dict | None = None, **kwargs: Any):
        attempt = 0
        max_attempts = 12
        import asyncio

        while True:
            try:
                async for chunk in self._llm.astream(self._sanitize_messages(input), config=config, **kwargs):
                    yield chunk
                break
            except Exception as e:
                if _is_retryable_error(e):
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    
                    sleep_time = self._get_retry_sleep_time(e)
                    logger.debug(f"Caught ResourceExhausted (Async Stream Attempt {attempt}). Retrying in {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_GeminiSafeWrapper":
        model_name = getattr(self._llm, "model", "").lower()
        if "gemma" in model_name:
            logger.warning(f"Skipping native tool binding for Gemma model '{model_name}'.")
            return self

        bound = self._llm.bind_tools(tools, **kwargs)
        return _GeminiSafeWrapper(bound)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "_GeminiSafeWrapper":
        wso = self._llm.with_structured_output(*args, **kwargs)
        return _GeminiSafeWrapper(wso)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def build_llm(*, use_gemini: bool = None):
    """
    Build and return the chat model.
    """
    # If not explicitly set, check environment variable
    if use_gemini is None:
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        use_gemini = not use_local

    if use_gemini:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "Gemini support requires `langchain-google-genai`.\n"
                "Install with: pip install -U langchain-google-genai"
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Missing Gemini API key. Set GOOGLE_API_KEY (preferred) or GEMINI_API_KEY."
            )

        model_name = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")

        gemini_llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            max_retries=0 
        )

        return _GeminiSafeWrapper(gemini_llm)

    model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        reasoning=True,
    )