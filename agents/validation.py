class ValidationState(dict):
    messages: str
    code: str
    compiled: bool
    errors: list[str]
    llm_calls: int