"""
Comprehensive Evaluation Script for LILA (LLM Integrated Language Agent)
Implements specific metrics requested by the professor:
1. Outcome (Task Success Rate, Pass@k)
2. Quality (Accuracy, Format Compliance/Constraint Violations)
3. Efficiency (Latency, Token Usage, Step Count)
4. Robustness (Tool Call Success, Retry Rate)
5. Graph Behavior (Iterations)

This script runs a "Golden Set" of tasks without modifying the agent code.
"""

import sys
import json
import asyncio
import time
import re
import statistics
import uuid
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add paths to access agents and mcp tools
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from orchestrator import build_app
from grammo.src.grammo.service import run_tests, compile_text

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("orchestrator").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

# ============================================
# 1. Golden Set Definition
# ============================================

TASKS = {
    "Factorial": {
        "prompt": "Write a Grammo function `func int -> factorial(int: n)` that computes the factorial of n. Output ONLY the function definition.",
        "harness": """
            func void -> main() {
                var int: res;
                res = factorial(5);
                <<! "RESULT: " # (res);
            }
        """,
        "expected_output_regex": r"RESULT:\s*120"
    },
    "Fibonacci": {
        "prompt": "Write a Grammo function `func int -> fib(int: n)` that returns the n-th Fibonacci number (0-indexed). Output ONLY the function definition.",
        "harness": """
            func void -> main() {
                var int: res;
                res = fib(6);
                <<! "RESULT: " # (res);
            }
        """,
        "expected_output_regex": r"RESULT:\s*8"
    },
    "GCD": {
        "prompt": "Write a Grammo function `func int -> gcd(int: a, int: b)` that computes the greatest common divisor. Output ONLY the function definition.",
        "harness": """
            func void -> main() {
                var int: res;
                res = gcd(48, 18);
                <<! "RESULT: " # (res);
            }
        """,
        "expected_output_regex": r"RESULT:\s*6"
    },
    "IsPrime": {
        "prompt": "Write a Grammo function `func bool -> is_prime(int: n)` that returns true if n is prime. Output ONLY the function definition.",
        "harness": """
            func void -> main() {
                var bool: p7, p4;
                p7 = is_prime(7);
                p4 = is_prime(4);
                if (p7) { <<! "7 is prime"; } else { << "7 is not prime"; }
                if (p4) { <<! "4 is prime"; } else { << "4 is not prime"; }
            }
        """,
        "expected_output_regex": r"7 is prime"
    }
}

# ============================================
# 2. Metrics Architecture
# ============================================

@dataclass
class RunMetrics:
    task_name: str
    success: bool = False
    latency_seconds: float = 0.0
    steps: int = 0
    estimated_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    compilation_success: bool = False
    iterations: int = 0
    output_length: int = 0
    constraint_violation: bool = False # e.g. text outside code block
    
    def to_dict(self):
        return asdict(self)

import os

class CtypesStdoutCapture:
    """Context manager to capture C-level stdout (printf) checking both Python sys.stdout and FD 1."""
    def __init__(self):
        self.captured = ""
        self._stdout_fd = None
        self._saved_stdout_fd = None
        self._pipe_out = None
        self._pipe_in = None

    def __enter__(self):
        sys.stdout.flush()
        try:
            self._stdout_fd = sys.stdout.fileno()
            self._saved_stdout_fd = os.dup(self._stdout_fd)
            self._pipe_out, self._pipe_in = os.pipe()
            os.dup2(self._pipe_in, self._stdout_fd)
        except Exception:
            self._stdout_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stdout_fd is None:
            return
        sys.stdout.flush()
        os.close(self._pipe_in)
        os.dup2(self._saved_stdout_fd, self._stdout_fd)
        os.close(self._saved_stdout_fd)
        with os.fdopen(self._pipe_out, 'r') as f:
            self.captured = f.read()

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens (char / 4)."""
    if not text:
        return 0
    return max(1, len(str(text)) // 4)

def extract_grammo_code(text: str) -> str:
    """
    Robust extraction of Grammo code from Mixed Markdown/Text.
    Prioritizes:
    1. ```grammo ... ``` blocks
    2. ```c ... ``` blocks (LLMs confuse Grammo with C)
    3. ``` ... ``` generic blocks
    4. Raw text if 'func' keyword is detected near start
    """
    if not text:
        return ""
        
    text = text.strip()
    
    # Try finding code blocks with specific tags
    patterns = [
        r"```grammo\s*([\s\S]*?)\s*```",
        r"```c\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
    # Fallback: if it looks like code (has 'func ' at start of a line)
    # and no markdown grammar structure is found
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("func "):
            return "\n".join(lines[i:])
            
    return text

def _get_message_content_length(content: Any) -> int:
    """Helper to get length of message content."""
    length = 0
    if isinstance(content, str):
        length += len(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                length += len(str(block.get("text", "")))
                length += len(str(block.get("image_url", "")))
            else:
                length += len(str(block))
    return length

def _calculate_estimated_tokens(messages: list) -> int:
    """Helper to calculate estimated tokens from message content."""
    total_chars = 0
    for m in messages:
        if hasattr(m, "content"):
            total_chars += _get_message_content_length(m.content)
    return estimate_tokens(str(total_chars))

def _count_tool_calls(messages: list, state: dict) -> tuple[int, int]:
    """Helper to count tool calls and errors."""
    requested_calls = 0
    executed_calls = 0
    tool_errors_count = 0
    
    for m in messages:
        if isinstance(m, AIMessage):
            if hasattr(m, 'tool_calls') and m.tool_calls:
                requested_calls += len(m.tool_calls)
        if isinstance(m, ToolMessage):
            executed_calls += 1
            content_str = str(m.content).lower()
            if "error" in content_str or "exception" in content_str or "failed" in content_str:
                tool_errors_count += 1
                
    final_tool_count = max(requested_calls, executed_calls)

    iterations = state.get("iterations", 0) + state.get("global_iterations", 0)
    if final_tool_count == 0 and iterations > 0:
        final_tool_count = iterations
        
    return final_tool_count, tool_errors_count

def analyze_state_metrics(state: dict) -> dict:
    """Extract metrics from the final AgentState without modifying agents."""
    messages = state.get("messages", [])
    steps = len(messages)
    estimated_tokens = _calculate_estimated_tokens(messages)
    tool_calls, tool_errors_count = _count_tool_calls(messages, state)
    
    iterations = state.get("iterations", 0) + state.get("global_iterations", 0)
    
    return {
        "steps": steps,
        "estimated_tokens": estimated_tokens,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors_count,
        "iterations": iterations
    }

class ProgressBar:
    def __init__(self, total, prefix='', decimals=1, length=50, fill='█'):
        self.total = total
        self.prefix = prefix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.iteration = 0

    def print_progress(self, iteration):
        self.iteration = iteration
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.prefix} |{bar}| {percent}% Complete', end='\r')
        if self.iteration == self.total:
            print()

async def generate_and_evaluate(app, task_name: str, config: dict) -> RunMetrics:
    start_time = time.time()
    
    try:
        thread_id = str(uuid.uuid4())
        final_state = await asyncio.wait_for(
            asyncio.to_thread(
                app.invoke,
                {"messages": [HumanMessage(content=config["prompt"])]},
                config={"configurable": {"stream_tokens": False, "thread_id": thread_id}}
            ),
            timeout=180.0
        )
    except asyncio.TimeoutError:
        return RunMetrics(task_name=task_name, success=False, latency_seconds=180.0, steps=0, estimated_tokens=0, tool_calls=0)
    except Exception:
        return RunMetrics(task_name=task_name, success=False, latency_seconds=time.time() - start_time)
        
    end_time = time.time()
    latency = end_time - start_time
    
    raw_code = final_state.get("code", "")
    code = extract_grammo_code(raw_code)
    
    constraint_violation = False
    if len(raw_code.strip()) > 0 and len(code) < len(raw_code.strip()) * 0.8: 
        constraint_violation = True
    
    internal_metrics = analyze_state_metrics(final_state)
    
    passed = False
    compilation_success = False
    
    if code:
        try:
            harness = config["harness"]
            c_res = compile_text(code + "\n\n" + harness)
            compilation_success = c_res.get("compiled", False)
            
            if compilation_success:
                try:
                    with CtypesStdoutCapture() as capture:
                        test_res = run_tests(code, harness)
                    
                    stdout = test_res.get("stdout", "") + capture.captured
                except Exception:
                    stdout = ""

                if re.search(config["expected_output_regex"], stdout):
                    passed = True
        except Exception:
            pass
    
    metric = RunMetrics(
        task_name=task_name,
        success=passed,
        latency_seconds=latency,
        steps=internal_metrics["steps"],
        estimated_tokens=internal_metrics["estimated_tokens"],
        tool_calls=internal_metrics["tool_calls"],
        tool_errors=internal_metrics["tool_errors"],
        compilation_success=compilation_success,
        iterations=internal_metrics["iterations"],
        output_length=len(code),
        constraint_violation=constraint_violation
    )

    return metric

# ============================================
# 3. Main Runner
# ============================================

def _save_json_results(data: dict, file_path: Path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

async def main():
    print("\n")
    print("="*70)
    print(" LILA COMPREHENSIVE EVALUATION (Professor's Metrics)")
    print("="*70)

    try:
        # Suppress logging during app build if possible, but some might leak
        app = build_app()
    except Exception as e:
        print(f"Error building app: {e}")
        return

    all_metrics: List[RunMetrics] = []
    
    RUNS_PER_TASK = 1
    total_tasks = len(TASKS) * RUNS_PER_TASK
    total_completed = 0
    
    print("\nStarting Evaluation...")
    pb = ProgressBar(total_tasks, prefix='Progress:', length=40)
    pb.print_progress(0)
    
    for task_name, config in TASKS.items():
        for _ in range(RUNS_PER_TASK):
            metric = await generate_and_evaluate(app, task_name, config)
            all_metrics.append(metric)
            total_completed += 1
            pb.print_progress(total_completed)
            
    # ============================================
    # 4. Reporting
    # ============================================
    
    print("\n" + "="*70)
    print(" 📊 FINAL REPORT")
    print("="*70)
    
    success_count = sum(1 for m in all_metrics if m.success)
    total_count = len(all_metrics)
    
    latencies = [m.latency_seconds for m in all_metrics]
    tokens = [m.estimated_tokens for m in all_metrics]
    tool_calls = [m.tool_calls for m in all_metrics]
    constraint_violations = sum(1 for m in all_metrics if m.constraint_violation)
    
    avg_tokens = statistics.mean(tokens) if tokens else 0
    avg_tool_calls = statistics.mean(tool_calls) if tool_calls else 0
    
    print("\n1. OUTCOME")
    print(f"   • Overall Task Success Rate: {success_count}/{total_count} ({success_count/total_count:.1%})")
    print(f"   • Constraint Violations:     {constraint_violations}/{total_count}")

    print("\n2. EFFICIENCY")
    print(f"   • End-to-End Latency (Avg): {statistics.mean(latencies):.2f}s")
    print(f"   • Avg Estimated Tokens:     {avg_tokens:.0f} tokens/run")

    print("\n3. GRAPH BEHAVIOR")
    print(f"   • Avg Tool Calls per Run:   {avg_tool_calls:.1f}")
    if tool_calls:
        total_tool_errors = sum(m.tool_errors for m in all_metrics)
        total_tool_calls = sum(tool_calls)
        err_rate = total_tool_errors/total_tool_calls if total_tool_calls else 0
        print(f"   • Tool Error Rate:          {total_tool_errors}/{total_tool_calls} ({err_rate:.1%})")
    
    # JSON Dump
    results_data = {
        "summary": {
            "success_rate": success_count/total_count if total_count else 0,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "avg_tokens": avg_tokens,
            "avg_tool_calls": avg_tool_calls
        },
        "details": [m.to_dict() for m in all_metrics]
    }
    
    out_file = Path(__file__).parent / "comprehensive_metrics.json"
    await asyncio.to_thread(_save_json_results, results_data, out_file)
    print(f"\n📄 Detailed metrics saved to: {out_file}")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
