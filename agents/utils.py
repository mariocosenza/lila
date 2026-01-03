import asyncio
import threading
import re
import json
from typing import Any, Optional
from tenacity import retry_if_exception
from google.api_core.exceptions import ResourceExhausted

# ==========================================
# 1. String & Code Sanitization
# ==========================================

def sanitize_grammo_source(text: str) -> str:
    """Best-effort sanitizer to ensure only Grammo source is returned.

    Removes markdown fences and drops leading natural-language/meta lines until
    the first plausible top-level declaration (`func` or `var`).
    """
    if text is None:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    # Strip markdown code fences if present.
    if '```' in s:
        m = re.search(r"```(?:\w+)?\s*\n(.*?)\n```", s, flags=re.DOTALL)
        if m:
            s = m.group(1).strip()
        else:
            s = s.replace('```', '').strip()

    # Drop leading natural-language / meta lines until we hit 'func' or 'var'.
    lines = s.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        line_s = line.lstrip()
        if not line_s:
            continue
        if line_s.startswith(('func', 'var')):
            start_idx = i
            break

    if start_idx is not None and start_idx > 0:
        s = "\n".join(lines[start_idx:]).strip()

    return s


def clean_json_text(text: str) -> str:
    """Cleans code fences and other noise from JSON output."""
    text = str(text).strip()
    # Remove markdown code blocks if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    return text


# ==========================================
# 2. Async Helper
# ==========================================

def _target_sync(coro, res: dict[str, Any]):
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

def run_async_in_sync(coro):
    """Safely run an async coroutine from a synchronous context."""
    try:
        # If we are in a loop, we can't use asyncio.run() directly if it's already running
        # But usually this is called from a thread or a sync context.
        # If there is a running loop, we must use a thread.
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(coro)
    else:
        # Running loop detected, spawn a thread
        res: dict[str, Any] = {}
        t = threading.Thread(target=_target_sync, args=(coro, res), daemon=True)
        t.start()
        t.join()
        if "error" in res:
            raise res["error"]
        return res.get("value")


# ==========================================
# 3. Retry Logic
# ==========================================

def is_retryable_error(exception: Exception) -> bool:
    """
    Check if the exception is a ResourceExhausted error, 
    even if wrapped inside a LangChain exception.
    """
    # 1. Direct Google Exception
    if isinstance(exception, ResourceExhausted):
        return True
        
    # 2. Wrapped Exception (e.g. ChatGoogleGenerativeAIError)
    if hasattr(exception, "__cause__") and isinstance(exception.__cause__, ResourceExhausted):
        return True
        
    # 3. String fallback
    msg = str(exception)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return True
        
    return False

def extract_retry_delay(exception: Exception, default_delay: float = None) -> float | None:
    """
    Attempts to extract the requested retry delay from an exception.
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
