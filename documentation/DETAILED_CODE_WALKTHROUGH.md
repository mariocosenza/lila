# 📘 Analisi Dettagliata del Codice e dell'Architettura di LILA

Questo documento fornisce una descrizione tecnica approfondita di ogni componente del sistema LILA (Language Interface for Logic Agents), spiegando le scelte architetturali, le librerie utilizzate e i protocolli di comunicazione.

---

## 1. Panoramica Architetturale

LILA è un sistema multi-agente progettato per generare, compilare, testare e correggere codice in un linguaggio proprietario chiamato **Grammo**. L'architettura è modulare e distribuita, basata su tre pilastri principali:

1.  **Orchestrator & Agents (Client)**: Il cervello del sistema, implementato con **LangGraph**. Gestisce il flusso di lavoro tra Planner, Generator, Tester e Debugger.
2.  **MCP Server (Model Context Protocol)**: Un server che espone gli strumenti "fisici" (compilatore, analizzatore sintattico, runner di test) agli agenti AI in modo standardizzato.
3.  **Tester Server (Agent-to-Agent Service)**: Un microservizio dedicato che ospita l'agente di testing (basato su modelli locali come Gemma 3 o GPT-OSS) per disaccoppiare il carico di lavoro e permettere l'uso di modelli specializzati.

### Diagramma di Flusso Logico

```mermaid
graph TD
    User[Utente] -->|Input| Client[Agent Client (UI)]
    Client -->|Start| Planner
    Planner -->|Plan| Generator
    Generator -->|Grammo Code| CompilerNode
    CompilerNode -->|MCP Call| MCPServer[MCP Server (Port 8000)]
    MCPServer -->|Result| CompilerNode
    
    CompilerNode -->|Success| TesterNode
    CompilerNode -->|Error| Debugger
    
    TesterNode -->|HTTP POST| TesterServer[Tester Server (Port 8088)]
    TesterServer -->|Ollama| LocalLLM[Gemma 3 / Ollama]
    TesterServer -->|MCP Call| MCPServer
    
    TesterServer -->|Result| TesterNode
    TesterNode -->|Success| End
    TesterNode -->|Failure| Debugger
    
    Debugger -->|Feedback| Generator
```

---

## 2. Struttura del Progetto e File

### Cartella `agents/` (Logica Agenti)
Questa cartella contiene l'implementazione degli agenti intelligenti e dell'orchestrazione.

*   **`agent_client.py`**: 
    *   **Ruolo**: Punto di ingresso (Entry Point) per l'utente.
    *   **Tecnologia**: Usa la libreria `rich` per creare un'interfaccia a riga di comando (CLI) moderna e interattiva.
    *   **Funzionamento**: Inizializza il grafo di LangGraph (`multi_agent.py`), gestisce la selezione del modello (Gemini 3, Gemma 3, ecc.) e visualizza lo streaming degli eventi del grafo.
    
*   **`multi_agent.py`**:
    *   **Ruolo**: Definisce la `StateGraph` (macchina a stati) di LangGraph.
    *   **Dettaglio**: Configura i nodi (`planner`, `generator`, `compiler`, `tester`, `debugger`) e gli archi condizionali (es. se la compilazione fallisce -> vai al debugger). Gestisce la memoria condivisa (`AgentState`).

*   **`planner.py`**:
    *   **Ruolo**: Agente pianificatore.
    *   **Logica**: Analizza la richiesta utente e produce un piano in linguaggio naturale. Non scrive codice, ma prepara il terreno per il Generator.

*   **`generator.py`**:
    *   **Ruolo**: Agente generatore.
    *   **Logica**: Prende il piano e scrive il codice Grammo. È istruito specificamente sulla sintassi di Grammo (I/O con `>>` e `<<`, tipi espliciti).

*   **`tester.py`** e **`tester_server.py`**:
    *   **Ruolo**: Agente di testing.
    *   **Architettura A2A (Agent-to-Agent)**: Il `tester.py` definisce la logica dell'agente. Il `tester_server.py` avvolge questa logica in un server **FastAPI**.
    *   **Perché?**: Questo permette di eseguire l'agente di testing su una macchina diversa o un processo isolato, usando modelli locali (Ollama) senza appesantire il processo principale.
    *   **Protocollo**: Comunica via HTTP REST (`POST /invoke`).

*   **`debugger_evaluator.py`**:
    *   **Ruolo**: Agente di correzione.
    *   **Logica**: Analizza l'output di errore (dal compilatore o dai test), confronta il codice attuale con l'errore e suggerisce o applica correzioni.

*   **`mcp_client.py`**:
    *   **Ruolo**: Client per il protocollo MCP.
    *   **Tecnologia**: Usa `fastmcp.Client` per connettersi al server MCP. Gestisce la serializzazione/deserializzazione dei messaggi JSON-RPC su HTTP.

### Cartella `mcp/` (Server degli Strumenti)
Implementa il protocollo MCP per esporre le funzionalità del compilatore.

*   **`server.py`**:
    *   **Ruolo**: Server MCP.
    *   **Tecnologia**: `FastMCP`. Espone tre tool principali:
        1.  `grammo_lark`: Validazione sintattica veloce.
        2.  `grammo_compiler`: Compilazione completa in LLVM IR.
        3.  `grammo_test`: Esecuzione di test case.
    *   **Protocollo**: In ascolto su HTTP (porta 8000), funge da bridge tra gli agenti AI e il codice Python del compilatore.

*   **`grammo/`**:
    *   Contiene l'implementazione del linguaggio Grammo.
    *   **`lex_syntax/grammo.lark`**: La grammatica formale definita con la libreria **Lark**. Definisce token e regole di produzione.
    *   **`semantic/`**: Analisi semantica (controllo tipi, scope delle variabili).
    *   **`codegen/`**: Generazione di codice **LLVM IR** usando la libreria `llvmlite`.

---

## 3. Librerie Chiave e Scelte Tecniche

### 1. LangGraph & LangChain
*   **Scelta**: LangGraph è stato scelto per la sua capacità di gestire flussi ciclici (loop di correzione) e stato persistente, essenziale per il pattern "Generator -> Error -> Debugger -> Generator".
*   **Utilizzo**: Definisce il grafo degli agenti in `multi_agent.py`.

### 2. FastMCP (Model Context Protocol)
*   **Scelta**: MCP è uno standard emergente per connettere LLM a strumenti esterni.
*   **Vantaggio**: Disaccoppia l'implementazione del tool (il compilatore) dall'agente. Se domani cambiamo l'agente da Gemini a Claude, il server MCP rimane identico.
*   **Implementazione**: `mcp/server.py` usa `FastMCP` per creare rapidamente server compatibili.

### 3. FastAPI
*   **Scelta**: Usato in `tester_server.py`. È lo standard de facto per API Python moderne, performante e con validazione automatica (Pydantic).
*   **Utilizzo**: Espone l'agente Tester come un microservizio REST.

### 4. Lark
*   **Scelta**: Parser library per Python. Scelta per la sua facilità d'uso e potenza nel definire grammatiche EBNF.
*   **Utilizzo**: Parsing del codice Grammo in un albero sintattico (AST).

### 5. LLVM (via llvmlite)
*   **Scelta**: Backend di compilazione industriale.
*   **Utilizzo**: Grammo non è interpretato, ma compilato in IR (Intermediate Representation) ed eseguito JIT (Just-In-Time), garantendo performance reali.

### 6. Rich
*   **Scelta**: Libreria per interfacce terminale.
*   **Utilizzo**: Rende l'output dell'agente leggibile, con colori, tabelle e pannelli, migliorando la UX dello sviluppatore.

---

## 4. Protocolli di Comunicazione

### MCP (Model Context Protocol)
*   **Tipo**: JSON-RPC (Remote Procedure Call) su HTTP (in questa implementazione specifica, spesso MCP usa stdio, ma qui è adattato su HTTP/SSE).
*   **Flusso**:
    1.  Client invia: `{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "grammo_compiler", "arguments": {"code": "..."}}}`
    2.  Server esegue la funzione Python `compiler()`.
    3.  Server risponde: `{"jsonrpc": "2.0", "result": {"content": [{"type": "text", "text": "Compilation successful..."}]}}`

### HTTP REST (Agent-to-Agent)
*   **Endpoint**: `POST http://localhost:8088/invoke`
*   **Payload**: JSON definito da modelli Pydantic (`A2ARequest`).
*   **Scopo**: Permette all'agente principale (Orchestrator) di delegare il task di testing a un sottosistema autonomo.

---

## 5. Walkthrough di una Richiesta Utente

1.  **Input**: L'utente digita "Crea una calcolatrice".
2.  **Planner**: Riceve l'input. Consulta il prompt di sistema e genera un piano: "1. Definire funzioni per somma, sottrazione... 2. Creare un loop nel main...".
3.  **Generator**: Riceve il piano. Usa il modello LLM (es. Gemini 3) per tradurre il piano in codice Grammo, rispettando la sintassi `func int -> ...`.
4.  **Compiler Node**:
    *   Chiama `mcp_client.grammo_compiler(code)`.
    *   Il client contatta `localhost:8000`.
    *   Il server MCP parsa il codice con Lark. Se OK, genera LLVM IR.
    *   Ritorna "Success" o errori di sintassi.
5.  **Decisione**:
    *   Se **Errore**: Il grafo passa il controllo al **Debugger**. Il Debugger legge l'errore, modifica il codice e lo rimanda al Compiler.
    *   Se **Successo**: Il grafo passa al **Tester**.
6.  **Tester Node**:
    *   Invia il codice a `localhost:8088` (Tester Server).
    *   Il Tester Server usa Ollama (Gemma 3) per generare casi di test (es. "input: 5, 3 -> output atteso: 8").
    *   Esegue i test chiamando nuovamente il server MCP (`grammo_test`).
    *   Ritorna il report dei test.
7.  **Output**: L'utente vede il risultato finale sulla CLI.

---

## 6. Dettagli sui Modelli AI

### Gemini 3 Pro / Flash
*   Usati per la logica principale (Planner, Generator, Debugger).
*   Scelti per la loro capacità di seguire istruzioni complesse e gestire finestre di contesto ampie (necessarie per mantenere in memoria la specifica del linguaggio Grammo).

### Gemma 3 27b (Locale)
*   Usato per il Tester.
*   Scelto per dimostrare la capacità ibrida Cloud/Locale. Essendo un modello locale, riduce i costi API per operazioni ripetitive come la generazione di test case.
*   **Nota Tecnica**: Poiché Gemma 3 ha peculiarità nel gestire i prompt di sistema, LILA implementa una logica di "patching" che fonde le istruzioni di sistema nel primo messaggio utente.
*   **Configurazione di Test**: I test di validazione del sistema sono stati eseguiti utilizzando **Gemma 3** come modello locale per il Tester Server, con l'agente di test configurato per operare in modalità autonoma (disaccoppiato dal flusso principale se necessario).

---

## 7. Riferimenti e Crediti

Il linguaggio **Grammo** e la sua documentazione originale sono stati creati da **Salvatore Di Martino**. Per dettagli specifici sulla sintassi e le specifiche originali, fare riferimento a:

*   **Repository Grammo**: [https://github.com/saldm04/Grammo](https://github.com/saldm04/Grammo)
*   **Documentazione Ufficiale**: [https://github.com/saldm04/Grammo/tree/main/Documents](https://github.com/saldm04/Grammo/tree/main/Documents)
*   **Profilo Autore**: [https://github.com/saldm04](https://github.com/saldm04)
