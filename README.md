# 🌸 LILA (LLM Integrated Language Agent)

![License](https://img.shields.io/github/license/mariocosenza/lila?style=for-the-badge&color=blueviolet)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜🔗%20LangChain-0.1.0-green?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-Alpha-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Development-success?style=for-the-badge)

**LILA** is an advanced, autonomous multi-agent coding system designed to master **Grammo**, a custom programming language that compiles to LLVM IR. 

Built on top of **LangGraph** and **Model Context Protocol (MCP)**, LILA orchestrates a team of specialized AI agents that plan, write, compile, test, and debug code iteratively until the user's requirements are met.

---

## 🚀 Features

- **🤖 Multi-Agent Architecture**: A coordinated swarm of agents (Orchestrator, Planner, Generator, Integrator, Tester, Debugger) working together.
- **📝 Grammo Language Support**: Native understanding of the Grammo syntax (Lark-based) and compilation pipeline (LLVM IR).
- **🔄 Self-Healing Workflow**: Agents automatically analyze compiler errors and test failures to fix their own code.
- **🔌 Model Context Protocol (MCP)**: Standardized server-client architecture for tool exposure and execution.
- **🧠 Flexible Brain**: Supports **Google Gemini Pro** for high-reasoning tasks and **Ollama** for local, privacy-focused execution.
- **🧪 Automated Testing**: Integrated test runner that executes generated code against test cases and reports results back to the agents.

---

## 🏗️ Architecture

LILA operates as a state machine graph where nodes represent agents or tools:

1.  **Orchestrator**: Analyzes the request and routes it (Generator vs Planner).
2.  **Planner**: Decomposes complex tasks into step-by-step subtasks.
3.  **Generator**: Writes Grammo code and fixes syntax errors.
4.  **Integrator**: Assembles code fragments into a complete program.
5.  **Tester**: Runs the compiled program against defined test cases.
6.  **Debugger/Evaluator**: Analyzes failures and requests fixes or finalizes the output.

---

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/mariocosenza/lila.git
    cd lila
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have `langchain`, `langgraph`, `lark`, `llvmlite`, etc. installed)*

---

## ⚙️ Configuration

Create a `.env` file or set environment variables:

```env
# Required for Gemini (Default)
GOOGLE_API_KEY=your_gemini_api_key

# Optional: Use Local LLM (Ollama)
USE_LOCAL_LLM=false
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
```

---

## 🏃 Usage

### 1. Start All Services (Interactive Mode)
This script starts the MCP Server, the Tester Server, and launches the interactive Agent Client in the current terminal.

```powershell
.\start_all_services.ps1
```

### 2. Start Services Only (Headless)
If you want to run the client separately or just keep the backend running:

```powershell
# Default (Gemini)
.\start_services_no_client.ps1

# Use Local LLM
.\start_services_no_client.ps1 -Local
```

### 3. Agent Client
Once services are running, you can interact with LILA:

```bash
python agents/agent_client.py
```

---

## 📝 The Grammo Language

Grammo is a C-like toy language designed for this project. It supports:
- **Types**: `int`, `real`, `bool`, `string`, `void`
- **Control Flow**: `if/elif/else`, `while`, `for`
- **I/O**: `>>` (input) and `<<!` (output with newline)
- **Functions**: `func type -> name(args) { ... }`

**Example:**
```grammo
func void -> main() {
   var int: n, r;
   >> "Enter number: " # (n);
   r = factorial(n);
   <<! "Result: " # (r);
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <sub>Built with ❤️ by Mario Cosenza</sub>
</div>
