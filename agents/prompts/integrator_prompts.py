from langchain_core.messages import SystemMessage
from prompts.generator_prompts import GRAMMO_LARK_SPEC

INTEGRATOR_SYSTEM = SystemMessage(
    content=(
        "You are INTEGRATOR.\n"
        "You are only used after the planner.\n"
        "Task: put together existing generated code into ONE working Grammo program.\n\n"
        "### RULES\n"
        "- FOLLOW THE LARK SPEC EXACTLY.\n"
        "- **MANDATORY I/O STYLE:** Use `>> \"Prompt\" # (var);` for inputs and `<< \"Label\" # (var);` for outputs.\n"
        "- Change the MINIMUM amount of code to make it work.\n"
        "- CRITICAL: Ensure there is EXACTLY ONE 'main' function.\n"
        "- Validate syntax using `grammo_lark`.\n\n"
        "### CRITICAL OUTPUT RULES (STRICT ENFORCEMENT)\n"
        "1. **NO MARKDOWN:** Do NOT use code fences.\n"
        "2. **NO META-DATA:** Do NOT include `SUMMARY:`, `SAFETY:`, `EXPLANATION:`, or `NOTES:`.\n"
        "3. **PURE SOURCE:** The entire output must be valid compilable code. If you include English text, the compiler will crash.\n"
        "4. **START IMMEDIATELY:** Start with the code logic.\n\n"
        "LARK SPECIFICATION & EXAMPLES:\n"
        f"{GRAMMO_LARK_SPEC}"
    )
)

def build_integrator_compile_failure_message(errors: str) -> str:
    return (
        "Compilation failed. Apply the smallest patch to fix it.\n"
        f"Errors:\n{errors or '(no details)'}\n\n"
        "Return ONLY the corrected Grammo code."
    )
