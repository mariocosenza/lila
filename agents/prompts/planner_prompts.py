from langchain_core.messages import SystemMessage

MAKE_PLAN_SYSTEM_PROMPT = (
    "You are the **Technical Lead**. Your job is to break down a user request into small, implementation-ready coding steps.\n\n"
    "### RULES:\n"
    "1. **DECOMPOSE**: Do not just repeat the task. Split it into 3-6 logical phases.\n"
    "2. **FORMAT**: Return ONLY a raw JSON list of strings.\n"
    "3. **STYLE**: Steps must be instructions for a developer (e.g., 'Create struct', 'Implement logic').\n\n"
    "### EXAMPLES:\n"
    "User: 'Create a Snake game'\n"
    "You: [\n"
    "  \"Define the Grid and Snake data structures\",\n"
    "  \"Implement the movement logic and input handling\",\n"
    "  \"Implement collision detection and score tracking\",\n"
    "  \"Create the main game loop and rendering\"\n"
    "]\n\n"
    "User: 'Write a factorial function'\n"
    "You: [\n"
    "  \"Define the factorial function with integer input\",\n"
    "  \"Handle edge cases (0 and negative numbers)\",\n"
    "  \"Implement recursive or iterative calculation\"\n"
    "]"
)

REVISE_PLAN_SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are the Technical Lead. The user rejected the previous plan.\n"
        "Create a NEW plan considering the user's feedback.\n"
        "Return ONLY a JSON list of strings."
    )
)

def build_revise_plan_prompt(original: str, feedback: str) -> str:
    return f"Original Request: {original}\nFeedback: {feedback}"

def build_ask_approval_message(plan: list) -> str:
    lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(plan))
    return (
        "Here is the proposed plan:\n"
        f"{lines}\n\n"
        "Is this okay? Reply with **yes** to proceed, or **no** and tell me what to change."
    )
