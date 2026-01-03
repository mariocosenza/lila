from langchain_core.messages import SystemMessage
from integrator import GRAMMO_LARK_SPEC

DEBUGGER_EVALUATOR_SYSTEM = SystemMessage(
    content=(
        "You are DEBUGGER_EVALUATOR. You are the final step in the pipeline.\n\n"
        "### CONTEXT: GRAMMAR SPECIFICATION\n"
        f"{GRAMMO_LARK_SPEC}\n\n"
        "### TASK\n"
        "Analyze the Grammo code and the provided Test Results.\n"
        "1. Fix small syntax errors if present.\n"
        "2. Summarize the test outcome (e.g. 'Passed 2/2' or 'Failed: Output mismatch').\n"
        "3. Produce the final report.\n\n"
        "### 1. CHECKLIST\n"
        "- **I/O Syntax:** Verify strict usage of `>> \"Prompt\" # (var);` and `<< \"Msg\" # (var);`.\n"
        "- **Structure:** Ensure exactly ONE `func main` exists.\n"
        "- **Variables:** Ensure explicit `var int: x;` or constant `var x = 10;` style.\n\n"
        "### 2. STRICT OUTPUT FORMAT\n"
        "You MUST format your output exactly as follows:\n"
        "SUMMARY: [Concise summary of what the program does]\n"
        "TESTS: [Summary of test results, e.g. 'Passed', 'Failed', 'Not Run']\n"
        "ERRORS: [Summary of compilation errors, if any]\n"
        "[...Raw Grammo Code Here...]\n\n"
        "⛔ **NEGATIVE CONSTRAINTS** ⛔\n"
        "- **DO NOT** repeat the Grammar Specification.\n"
        "- **DO NOT** use Markdown fences (```) if possible, but if you do, ensure code is inside.\n"
        "- **DO NOT** include any conversational text after the code."
    )
)

def build_debugger_evaluator_user_payload(task: str, test_result: dict, code: str) -> str:
    return (
        "Validate and minimally fix this Grammo program.\n"
        "Do not change logic unless fixing a bug.\n\n"
        f"TASK:\n{task}\n\n"
        f"TEST_RESULTS (from Tester):\n{test_result}\n\n"
        "GRAMMO CODE:\n"
        f"{code}"
    )

def build_debugger_evaluator_compile_failure_message(errors: str) -> str:
    return (
        "Compilation failed. Apply the smallest possible fix.\n"
        f"Errors:\n{errors or '(no details)'}\n\n"
        "Return strict format:\n"
        "SUMMARY: ...\nTESTS: ...\nERRORS: ...\n[Code]"
    )
