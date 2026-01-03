import json
from typing import Dict, Any
from integrator import GRAMMO_LARK_SPEC

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

def build_initial_test_prompt(code: str) -> str:
    return (
        "GRAMMO CODE:\n"
        f"{code}\n\n"
        "Generate comprehensive tests and call `run_grammo_tests`."
    )

def build_debug_test_prompt(last_result: Dict[str, Any], current_tests: str, code: str) -> str:
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

def build_tester_server_invoke_prompt(task: str) -> str:
    return (
        (task.strip() + "\n\n") if task.strip() else ""
        "Generate tests for the given Grammo program and run them.\n"
        "Return a short summary after the tool call."
    )
