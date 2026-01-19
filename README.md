# 🌸 LILA (LLM Integrated Language Agent)

![License](https://img.shields.io/github/license/mariocosenza/lila?style=for-the-badge&color=blueviolet)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜🔗%20LangChain-1.2.0-green?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.5-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Development-success?style=for-the-badge)

**LILA** è un sistema multi-agente autonomo avanzato progettato per padroneggiare **Grammo**, un linguaggio di programmazione custom che compila in LLVM IR.

Basato su **LangGraph** e **Model Context Protocol (MCP)**, LILA orchestra un team di agenti AI specializzati che pianificano, scrivono, compilano, testano e correggono codice in modo iterativo fino a soddisfare i requisiti dell’utente.

---

## 🚀 Caratteristiche principali

- **🤖 Architettura Multi-Agente**: Orchestrator, Planner, Generator, Integrator, Tester, Debugger lavorano insieme.
- **📝 Supporto nativo a Grammo**: Parsing, analisi e generazione codice per Grammo (basato su Lark, output LLVM IR).
- **🔄 Workflow auto-riparante**: Gli agenti analizzano errori di compilazione e test, correggendo autonomamente il codice.
- **🔌 Model Context Protocol (MCP)**: Architettura server-client standardizzata per l’esposizione e l’esecuzione di tool.
- **🧠 Modelli flessibili**: Supporto a **Google Gemini Pro** (cloud) e **Ollama** (locale, privacy).
- **🧪 Testing automatico**: Runner integrato che esegue i test e riporta i risultati agli agenti.

---

## 🏗️ Architettura

LILA funziona come una macchina a stati dove i nodi rappresentano agenti o strumenti:

1.  **Orchestrator**: Analizza la richiesta e la smista (Generator o Planner).
2.  **Planner**: Scompone task complessi in sottotask sequenziali.
3.  **Generator**: Scrive codice Grammo e corregge errori di sintassi.
4.  **Integrator**: Assembla i frammenti in un programma completo.
5.  **Tester**: Esegue il programma compilato sui test case.
6.  **Debugger/Evaluator**: Analizza i fallimenti e richiede correzioni o finalizza l’output.

---

## 🛠️ Requisiti e installazione

### Ambiente consigliato

Si consiglia l’uso di **conda** (o Miniconda/Anaconda) per la gestione dell’ambiente Python e delle dipendenze.

1. **Installa [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o Anaconda** se non già presente.

2. **Clona il repository**
   ```bash
   git clone https://github.com/mariocosenza/lila.git
   cd lila
   ```

3. **Crea l’ambiente conda tramite environment.yml**
   ```bash
   conda env create -f environment.yml
   conda activate lila
   ```
   Questo installerà tutte le dipendenze necessarie (Python, langchain, langgraph, lark, llvmlite, ecc.).

> ⚠️ **Non usare requirements.txt**: tutte le dipendenze sono definite in `environment.yml`.

---

## ⚙️ Configurazione

Configura le variabili d’ambiente (puoi usare un file `.env`):

```env
# Necessario per Gemini (default)
GOOGLE_API_KEY=la_tua_chiave_gemini

# Opzionale: per usare LLM locale (Ollama)
USE_LOCAL_LLM=false
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
```

---

## 🏃‍♂️ Avvio e utilizzo

### 1. Avvio completo (modalità interattiva)
Questo script avvia MCP Server, Tester Server e il client interattivo nella stessa finestra:

```powershell
.\start_all_services.ps1
```

### 2. Solo servizi backend (senza client)
Per avviare solo i servizi e collegare il client separatamente:

```powershell
# Default (Gemini)
.\start_services_no_client.ps1

# Per usare LLM locale
.\start_services_no_client.ps1 -Local
```

### 3. Avvio del client
Quando i servizi sono attivi puoi interagire con LILA tramite:

```bash
python agents/agent_client.py
```

### 4. Valutazione automatica
LILA include una suite di valutazione per misurare performance, accuratezza ed efficienza:

```bash
# Pass@k benchmark (verifica compilazione)
python test/eval_pass_at_k.py

# Valutazione funzionale completa (metriche docente)
python test/eval_comprehensive.py
```

---

## 📝 Il linguaggio Grammo

Grammo è un linguaggio didattico stile C progettato per questo progetto. Supporta:
- **Tipi**: `int`, `real`, `bool`, `string`, `void`
- **Controllo di flusso**: `if/elif/else`, `while`, `for`
- **I/O**: `>>` (input), `<<!` (output con newline)
- **Funzioni**: `func tipo -> nome(args) { ... }`

**Esempio:**
```grammo
func void -> main() {
     var int: n, r;
     >> "Inserisci numero: " # (n);
     r = fattoriale(n);
     <<! "Risultato: " # (r);
}
```

### Crediti & Documentazione
Il linguaggio Grammo è stato creato da **Salvatore Di Martino**.
Per la documentazione ufficiale e la teoria, vedi:
- **Repository**: [saldm04/Grammo](https://github.com/saldm04/Grammo)
- **Documentazione**: [Grammo Documents](https://github.com/saldm04/Grammo/tree/main/Documents)

---

## 🤝 Contribuire

Contributi e segnalazioni sono benvenuti! Apri una Pull Request.

1. Fai il fork del progetto
2. Crea una branch (`git checkout -b feature/NomeFunzionalità`)
3. Fai commit delle modifiche (`git commit -m 'Aggiunta funzionalità'`)
4. Push sulla branch (`git push origin feature/NomeFunzionalità`)
5. Apri una Pull Request

---

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi il file `LICENSE`.

---

<div align="center">
    <sub>Realizzato con ❤️ da Mario Cosenza</sub>
</div>
