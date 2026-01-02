# 📘 Guida Utente LILA

## 🚀 Per Iniziare

### Prerequisiti
- **Python 3.10+**
- **Git**
- **PowerShell** (per gli script di avvio, anche se l'esecuzione manuale funziona anche su Bash)

### Installazione

1.  **Clona il repository**
    ```bash
    git clone https://github.com/mariocosenza/lila.git
    cd lila
    ```

2.  **Configura l'Ambiente Virtuale**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/Mac
    ```

3.  **Installa le Dipendenze**
    ```bash
    pip install -r requirements.txt
    ```

### Configurazione dell'Ambiente

LILA richiede un backend LLM. Puoi usare **Google Gemini** (Cloud) o **Ollama** (Locale).

Crea un file `.env` nella directory principale:

**Opzione A: Google Gemini (Consigliato per le migliori prestazioni)**
```env
GOOGLE_API_KEY=la_tua_chiave_api_reale_qui
```

**Opzione B: LLM Locale (Ollama)**
Assicurati di avere [Ollama](https://ollama.com/) installato e in esecuzione.
```env
USE_LOCAL_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b  # O llama3, mistral, ecc.
```

---

## 🎮 Eseguire LILA

LILA consiste di tre componenti principali che devono essere eseguiti simultaneamente:
1.  **Server MCP**: Gestisce l'esecuzione degli strumenti (compilazione, controllo sintassi).
2.  **Server Tester**: Gestisce la generazione e l'esecuzione dei test.
3.  **Client Agente**: L'interfaccia chat interattiva.

Forniamo script PowerShell per gestire tutto facilmente.

### Modalità 1: Sessione Interattiva Completa
Questo avvia tutto in una volta. Il client si avvierà nella finestra del terminale corrente.

```powershell
.\start_all_services.ps1
```

### Modalità 2: Headless / Solo Server
Usa questo se vuoi eseguire il client separatamente o stai facendo debugging del backend.

```powershell
# Esegui con Gemini
.\start_services_no_client.ps1

# Esegui con LLM Locale
.\start_services_no_client.ps1 -Local
```

Poi, in un terminale separato:
```bash
python agents/agent_client.py
```

---

## 💡 Come Usare l'Agente

Una volta che il client è in esecuzione, vedrai un prompt. Puoi chiedere a LILA di scrivere codice per te.

**Esempi:**

> "Scrivi un programma che calcola la sequenza di Fibonacci fino a N."

> "Crea una calcolatrice che supporti addizione, sottrazione, moltiplicazione e divisione."

> "Scrivi una funzione per controllare se un numero è primo, e testala."

### Consigli per i Migliori Risultati
- **Sii Specifico**: "Scrivi una funzione fattoriale" va bene. "Scrivi una funzione fattoriale che prenda input dall'utente e stampi il risultato" è meglio.
- **Richiedi Test**: Se vuoi assicurarti che il codice funzioni, aggiungi "e testalo" o "verificalo" al tuo prompt. Questo forza l'Orchestrator a instradare il task attraverso l'agente Tester.
- **Feedback**: Se l'agente chiede chiarimenti (in modalità Planner), rispondi con "sì" per procedere o fornisci feedback per rivedere il piano.

---

## 📊 Visualizzare l'Architettura

Puoi generare una rappresentazione visiva del grafo degli agenti corrente eseguendo:

```bash
python generate_graph.py
```

Questo creerà un file diagramma Mermaid (`documentation/lila_graph_mermaid.mmd`) e proverà a generare un'immagine PNG.
