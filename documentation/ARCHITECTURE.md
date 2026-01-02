# 🏗️ Documentazione dell'Architettura LILA

## Panoramica

LILA (LLM Integrated Language Agent) è progettato come un **Sistema Multi-Agente (MAS)** orchestrato da un grafo a macchina a stati. Il sistema sfrutta **LangGraph** per definire il flusso di controllo e i dati tra agenti specializzati.

La filosofia di base è il **raffinamento iterativo**: il codice non viene semplicemente generato, ma pianificato, scritto, compilato, testato e debuggato in un ciclo continuo finché non soddisfa i requisiti.

## 🧩 Componenti Principali

### 1. Il Grafo (Macchina a Stati)
L'intera applicazione è un grafo diretto dove:
- **Nodi** sono agenti o passaggi di elaborazione.
- **Archi** definiscono la logica di transizione basata sullo stato corrente (es. "Se la compilazione fallisce, torna al Generator").

### 2. Stato Condiviso (`AgentState`)
Tutti gli agenti condividono una struttura di memoria comune (`AgentState`), garantendo che il contesto sia preservato attraverso le transizioni.

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]  # Cronologia della conversazione
    task: str                    # Descrizione del task corrente
    code: str                    # Codice Grammo generato
    compile_result: Dict         # Output dal compilatore
    test_result: Dict            # Output dal runner dei test
    run_tests: bool              # Flag per forzare il testing
    # ... e altro
```

## 🤖 Agenti e Ruoli

### 1. Orchestrator (Router)
- **Ruolo**: Il punto di ingresso e decisore.
- **Logica**: Analizza la richiesta dell'utente per decidere la strategia migliore.
  - **Percorso Generator**: Per task semplici, su singolo file.
  - **Percorso Planner**: Per task complessi, multi-step.
- **Intelligenza**: Determina se il testing è richiesto (`run_tests`) basandosi sulla complessità del task o su richiesta esplicita dell'utente.

### 2. Planner (Architetto)
- **Ruolo**: Decompone problemi complessi.
- **Logica**:
  1.  **Crea Piano**: Suddivide il task in una lista JSON di sotto-task.
  2.  **Chiedi Approvazione**: Presenta il piano all'utente.
  3.  **Gestisci Approvazione**: Attende la conferma o il feedback dell'utente.
  4.  **Esegui**: Invia i sotto-task uno alla volta al Generator.

### 3. Generator (Coder)
- **Ruolo**: Lo specialista nella sintassi Grammo.
- **Logica**:
  - Scrive codice basandosi sul task corrente.
  - **Auto-Correzione**: Se il codice generato è vuoto o invalido, riprova.
  - **Controllo Sintassi**: Usa il tool `grammo_lark` per verificare la sintassi prima di finalizzare.
  - **Ciclo di Compilazione**: Tenta di compilare il codice. Se la compilazione fallisce, riceve l'errore e prova a correggere il codice (fino a 5 tentativi).

### 4. Integrator
- **Ruolo**: L'assemblatore.
- **Logica**: Usato nel flusso del Planner per combinare frammenti di codice da più sotto-task in un unico programma coerente.

### 5. Tester
- **Ruolo**: L'ingegnere Quality Assurance (QA).
- **Logica**:
  - Invia il codice compilato e la descrizione del task a un **Tester Server** esterno.
  - Il server genera casi di test (input/output attesi) ed esegue il binario compilato.
  - Restituisce un report dettagliato (Pass/Fail, log).

### 6. Debugger / Evaluator
- **Ruolo**: Il validatore finale e correttore.
- **Logica**:
  - Analizza i Risultati dei Test.
  - Se i test falliscono, tenta di applicare patch al codice.
  - Produce il sommario finale di output per l'utente.

## 🔄 Flussi di Lavoro

### Flusso Standard (Task Semplice)
`Orchestrator` -> `Generator` -> (Compilazione OK) -> `Validator` -> `Fine`

### Flusso Test-Driven (Task Complesso)
`Orchestrator` -> `Generator` -> (Compilazione OK) -> `Tester` -> `Debugger` -> (Correzioni) -> `Fine`

### Flusso Pianificato (Architettura Complessa)
`Orchestrator` -> `Planner` -> `Approvazione Utente` -> `Generator` (Ciclo per ogni step) -> `Integrator` -> `Tester` -> `Debugger` -> `Fine`

## 🔌 Model Context Protocol (MCP)

LILA usa MCP per esporre strumenti all'LLM.
- **Server**: `mcp/server.py` ospita strumenti come `grammo_lark` (checker sintattico) e `grammo_compile` (compilatore).
- **Client**: Gli agenti si connettono a questo server per eseguire questi strumenti in sicurezza.

## 🧠 Integrazione LLM

LILA supporta un approccio a doppio motore:
- **Google Gemini Pro**: Usato per ragionamento di alto livello, pianificazione e debugging complesso.
- **Ollama (Locale)**: Può essere attivato per esecuzione focalizzata sulla privacy o offline.
