import sys
import os
import uuid
from langchain_core.messages import HumanMessage

current_dir = os.path.dirname(os.path.abspath(__file__))
agents_dir = os.path.join(current_dir, '..', 'agents')
sys.path.append(agents_dir)

try:
    from orchestrator import build_app
    from agent_client import run_turn
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

def main():
    print("Starting Grammo code generation...")
    
    app = build_app()
    thread_id = str(uuid.uuid4())
    
    prompt = """Generate a Grammo code.
Show a menu for choosing an arithmetic operation.
• Handle user input (integers or doubles).
• Calculate and return the result and handle a loop to continue to the next operation or close the program.
• N.B. use at least two functions. In general, try to use all the features of the implemented language."""

    messages = [HumanMessage(content=prompt)]
    
    print("Sending request to the agent system...")
    
    max_turns = 5
    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")
        
        final_state, messages = run_turn(app, thread_id, messages, trace_level="basic")
        
        extracted_code = final_state.get("assembled_code")
        if not extracted_code:
            extracted_code = final_state.get("code")

        if extracted_code:
            output_filename = "grammo_calculator_generated.txt"
            output_path = os.path.join(current_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(extracted_code)
                
            print(f"\nSuccess! Code saved to: {output_path}")
            print("-" * 30)
            print(extracted_code[:500] + ("..." if len(extracted_code) > 500 else ""))
            print("-" * 30)
            break
        
        print("\nAgent has paused (likely waiting for plan confirmation).")
        user_response = input("Press [Enter] to send 'yes' (confirm plan), or type a message: ").strip()
        if not user_response:
            user_response = "yes"
            
        messages.append(HumanMessage(content=user_response))

    else:
        print("\nError: Agent did not generate code within the maximum number of turns.")
        print("Final state keys:", final_state.keys())

if __name__ == "__main__":
    main()
