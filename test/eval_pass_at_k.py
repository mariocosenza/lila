"""
Pass@K Evaluation Script
Calcola la metrica pass@k (k=5,10,20) per tre compiti:
1. Calcolatrice (add, sub, mul, div)
2. Fattoriale
3. Fibonacci

Pass@K = 1 - (C(n-c, k) / C(n, k))
Dove n=numero campioni, c=campioni corretti
"""

from __future__ import annotations

import sys
import json
import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Any
from math import comb
from dataclasses import dataclass

# Aggiungi agents al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))

from langchain_core.messages import HumanMessage
from orchestrator import build_app
from generator import grammo_compile

# ============================================
# Task Definitions
# ============================================

TASKS = {
    "calcolatrice": {
        "prompt": """Generate a Grammo program that implements a simple calculator.
The program should:
1. Read two numbers from input
2. Read an operator (+, -, *, /)
3. Perform the operation
4. Output the result

Example I/O:
Input: 10, 5, +
Output: Result: 15

Make sure the program compiles and runs correctly.""",
        "name": "calculator"
    },
    "fattoriale": {
        "prompt": """Generate a Grammo program that computes the factorial of a number.
The program should:
1. Read a positive integer n
2. Calculate n! = n × (n-1) × (n-2) × ... × 1
3. Output the result

Example I/O:
Input: 5
Output: Factorial: 120

Make sure the program compiles and runs correctly.""",
        "name": "factorial"
    },
    "fibonacci": {
        "prompt": """Generate a Grammo program that computes the Nth Fibonacci number.
The program should:
1. Read a positive integer n
2. Calculate the n-th Fibonacci number (0-indexed)
3. Output the result

Example I/O:
Input: 6
Output: Fibonacci: 8

Make sure the program compiles and runs correctly.""",
        "name": "fibonacci"
    }
}

@dataclass
class PassKResult:
    task_name: str
    samples_generated: int
    samples_compiled: int
    pass_at_5: float
    pass_at_10: float
    pass_at_20: float
    
    def __repr__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"Task: {self.task_name}\n"
            f"Samples Generated: {self.samples_generated}\n"
            f"Samples Compiled: {self.samples_compiled}\n"
            f"─────────────────────────────────────────────────────────────\n"
            f"Pass@5  = {self.pass_at_5:.2%}\n"
            f"Pass@10 = {self.pass_at_10:.2%}\n"
            f"Pass@20 = {self.pass_at_20:.2%}\n"
            f"{'='*60}"
        )


def calculate_pass_at_k(num_correct: int, num_samples: int, k: int) -> float:
    """
    Calcola pass@k usando la formula:
    pass@k = 1 - C(n-c, k) / C(n, k)
    
    Args:
        num_correct: numero di campioni corretti
        num_samples: numero totale di campioni
        k: numero di tentativi
    
    Returns:
        pass@k come valore tra 0 e 1
    """
    if num_samples < k:
        # Se abbiamo meno campioni di k, usiamo solo quelli disponibili
        return 1.0 if num_correct > 0 else 0.0
    
    if num_correct == 0:
        return 0.0
    
    if num_correct == num_samples:
        return 1.0
    
    # Formula: 1 - C(n-c, k) / C(n, k)
    try:
        return 1.0 - (comb(num_samples - num_correct, k) / comb(num_samples, k))
    except:
        # Fallback se la combinazione non è calcolabile
        return 1.0 if num_correct > 0 else 0.0


def check_compilation(code: str) -> bool:
    """
    Verifica se il codice Grammo compila usando grammo_compile.
    """
    try:
        result = grammo_compile.invoke({"code": code})
        return bool(result.get("compiled", False))
    except Exception as e:
        print(f"  ⚠️  Compilation check failed: {e}")
        return False


async def generate_samples(app, task_prompt: str, num_samples: int) -> List[str]:
    """
    Genera num_samples campioni usando l'agent.
    """
    samples = []
    
    for i in range(num_samples):
        try:
            print(f"  Generating sample {i+1}/{num_samples}...", end=" ")
            
            thread_id = str(uuid.uuid4())
            result = await asyncio.to_thread(
                app.invoke,
                {"messages": [HumanMessage(content=task_prompt)]},
                config={"configurable": {"stream_tokens": False, "thread_id": thread_id}}
            )
            
            # Estrai il codice generato
            code = result.get("code") or result.get("assembled_code", "")
            
            if code and len(code.strip()) > 10:
                samples.append(code)
                print(f"✅ Generated ({len(code)} chars)")
            else:
                print("❌ Empty or invalid code")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return samples


async def evaluate_task(
    app,
    task_name: str,
    task_prompt: str,
    num_samples: int = 20
) -> PassKResult:
    """
    Valuta un task generando campioni e verificando la compilazione.
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")
    print(f"Generating {num_samples} samples...")
    
    # Genera campioni
    samples = await generate_samples(app, task_prompt, num_samples)
    num_generated = len(samples)
    
    print(f"\n✅ Generated {num_generated} samples")
    print("Checking compilation...")
    
    # Verifica compilazione
    num_compiled = 0
    for i, code in enumerate(samples):
        compiled = check_compilation(code)
        if compiled:
            num_compiled += 1
            print(f"  Sample {i+1}: ✅ Compiled")
        else:
            print(f"  Sample {i+1}: ❌ Failed")
    
    # Calcola pass@k
    pass_at_5 = calculate_pass_at_k(num_compiled, num_generated, 5)
    pass_at_10 = calculate_pass_at_k(num_compiled, num_generated, 10)
    pass_at_20 = calculate_pass_at_k(num_compiled, num_generated, 20)
    
    result = PassKResult(
        task_name=task_name,
        samples_generated=num_generated,
        samples_compiled=num_compiled,
        pass_at_5=pass_at_5,
        pass_at_10=pass_at_10,
        pass_at_20=pass_at_20
    )
    
    return result


async def main():
    """
    Main entry point per valutare tutti i task.
    """
    print("\n" + "="*60)
    print("PASS@K EVALUATION")
    print("Valutazione su: Calcolatrice, Fattoriale, Fibonacci")
    print("="*60)
    
    # Build app
    print("\nBuilding orchestrator app...")
    try:
        app = build_app()
        print("✅ App ready")
    except Exception as e:
        print(f"❌ Failed to build app: {e}")
        return
    
    # Parametri
    num_samples = 20  # Generiamo 20 campioni per calcolare pass@5, @10, @20
    
    # Valuta tutti i task
    results: List[PassKResult] = []
    
    for task_key, task_config in TASKS.items():
        result = await evaluate_task(
            app,
            task_config["name"],
            task_config["prompt"],
            num_samples=num_samples
        )
        results.append(result)
    
    # Stampa risultati finali
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for result in results:
        print(result)
    
    # Stampa summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        print(f"{result.task_name:15} | Pass@5: {result.pass_at_5:6.2%} | Pass@10: {result.pass_at_10:6.2%} | Pass@20: {result.pass_at_20:6.2%}")
    
    # Calcola medie
    avg_pass_at_5 = sum(r.pass_at_5 for r in results) / len(results)
    avg_pass_at_10 = sum(r.pass_at_10 for r in results) / len(results)
    avg_pass_at_20 = sum(r.pass_at_20 for r in results) / len(results)
    
    print(f"{'─'*60}")
    print(f"{'Average':15} | Pass@5: {avg_pass_at_5:6.2%} | Pass@10: {avg_pass_at_10:6.2%} | Pass@20: {avg_pass_at_20:6.2%}")
    print(f"{'='*60}")
    
    # Salva risultati in JSON
    output_file = Path(__file__).parent / "pass_k_results.json"
    results_dict = {
        "tasks": [
            {
                "name": r.task_name,
                "samples_generated": r.samples_generated,
                "samples_compiled": r.samples_compiled,
                "pass_at_5": r.pass_at_5,
                "pass_at_10": r.pass_at_10,
                "pass_at_20": r.pass_at_20,
            }
            for r in results
        ],
        "averages": {
            "pass_at_5": avg_pass_at_5,
            "pass_at_10": avg_pass_at_10,
            "pass_at_20": avg_pass_at_20,
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
