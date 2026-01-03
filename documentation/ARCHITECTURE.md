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

## 🧠 Integrazione LLM e Performance

LILA supporta un approccio flessibile ai modelli linguistici, bilanciando potenza cloud e privacy locale.

### 1. Google Gemini (Cloud)
- **Modelli**: Gemini 2.5, Gemini 3 Pro, Gemini 3 Flash.
- **Punti di Forza**:
  - **Tool Calling Nativo**: Supporto eccellente per l'uso di strumenti MCP.
  - **System Prompts**: Supporto nativo per istruzioni di sistema complesse.
  - **Performance**: Risultati superiori nella generazione di sintassi Grammo corretta.
  - **Finestra di Contesto**: La finestra di contesto estesa di Gemini 3 permette di mantenere in memoria l'intera specifica del linguaggio Grammo e la cronologia di debugging senza perdita di informazioni.

### 2. Ollama (Locale)
- **Modelli Testati**: `gemma3:27b`, `gpt-oss-20b`.
- **Adattamenti Architetturali**:
  - **Gemma 3 27b**: Sebbene potente, questo modello può avere limitazioni nel supporto nativo per i *System Prompts* separati in alcune implementazioni di Ollama. L'architettura di LILA gestisce questo unendo le istruzioni di sistema nel primo messaggio utente (`_merge_system_for_gemma` in `multi_agent.py`).
  - **Tool Calling**: Per modelli come `gpt-oss-20b` che potrebbero non avere un supporto tool nativo robusto, LILA implementa strategie di fallback o parsing dell'output strutturato, permettendo l'uso di strumenti anche senza API dedicate.

### 3. Benchmark e Limitazioni (Pass@k)
I test condotti (vedi `test/pass_k_results.json`) evidenziano le sfide di generare codice in un linguaggio custom (`Grammo`) senza fine-tuning specifico.

- **Fibonacci**: Alto tasso di successo (Pass@20: 100%). Il pattern algoritmico è comune e il modello riesce ad adattarlo alla sintassi.
- **Calcolatrice / Fattoriale**: Tassi di successo inferiori (0% in alcuni run zero-shot). Questo indica che per logiche che richiedono I/O specifico o strutture di controllo annidate, i modelli generalisti faticano a rispettare rigorosamente la grammatica Grammo senza iterazioni di correzione (Debugger).

**Confronto Modelli**:
- **Gemini 3 Pro/Flash**: Mostrano una capacità di auto-correzione nettamente superiore nel ciclo Generator -> Compiler -> Debugger. Gemini 3 Flash offre un eccellente compromesso velocità/costo per iterazioni rapide.
- **Gemma 3 27b**: Tende a "dimenticare" le regole sintattiche specifiche (come l'I/O `>>` e `<<`) nei contesti lunghi, richiedendo più iterazioni del Debugger. Tuttavia, per task a singolo file, offre prestazioni sorprendenti per un modello locale.
