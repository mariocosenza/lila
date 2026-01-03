# 📝 Specifica del Linguaggio Grammo

**Grammo** è un linguaggio di programmazione giocattolo personalizzato, tipizzato staticamente, progettato per LILA. Presenta una sintassi simile al C con I/O semplificato e tipizzazione esplicita. Compila direttamente in **LLVM IR**.

## 🔤 Panoramica della Sintassi

### Struttura di Base
Un programma Grammo consiste in dichiarazioni di variabili globali e definizioni di funzioni. Non c'è una funzione `main` obbligatoria a meno che non sia richiesto un punto di ingresso per l'esecuzione.

```grammo
func void -> main() {
    // Il codice va qui
}
```

### Tipi di Dati
- `int`: Numeri interi (es. `42`)
- `real`: Numeri in virgola mobile (es. `3.14`)
- `bool`: Valori booleani (`true`, `false`)
- `string`: Stringhe di testo (`"Ciao"`)
- `void`: Per funzioni che non restituiscono un valore.

### Variabili
Le variabili devono essere dichiarate con `var`.

**Dichiarazione Esplicita:**
```grammo
var int: x, y;
var string: nome;
```

**Inizializzazione:**
```grammo
var x = 10;      // Tipo inferito come int
var pi = 3.14;   // Tipo inferito come real
```

### Input / Output (I/O)
Grammo usa operatori unici per l'I/O, progettati per essere facilmente parsati.

**Input (`>>`)**
Formato: `>> "Stringa di Prompt" # (variabile);`
```grammo
var int: eta;
>> "Inserisci la tua età: " # (eta);
```

**Output (`<<` e `<<!`)**
- `<<`: Stampa senza nuova riga.
- `<<!`: Stampa con nuova riga.
Formato: `<< "Etichetta" # (variabile);`
```grammo
<<! "La tua età è: " # (eta);
```

### Controllo di Flusso

**If / Elif / Else**
```grammo
if (x > 0) {
    <<! "Positivo" # (x);
} elif (x == 0) {
    <<! "Zero" # (x);
} else {
    <<! "Negativo" # (x);
}
```

**Ciclo While**
```grammo
while (x < 10) {
    x = x + 1;
}
```

**Ciclo For**
```grammo
for (var int: i = 0; i < 10; i = i + 1) {
    <<! "Conteggio: " # (i);
}
```

### Funzioni
Le funzioni sono definite con la parola chiave `func` e una freccia `->` che punta al nome.

```grammo
func int -> somma(int: a, int: b) {
    return a + b;
}

func void -> main() {
    var int: res;
    res = somma(5, 3);
}
```

## ⚠️ Vincoli e Best Practices per LLM

Poiché Grammo è un linguaggio custom non presente nel training set della maggior parte degli LLM, i modelli tendono a commettere errori ricorrenti. L'architettura di LILA mitiga questi problemi tramite il **Debugger**, ma è utile conoscere le criticità:

1.  **Allucinazione di Sintassi C/C++**: I modelli spesso usano `printf` o `std::cout` invece di `<<`.
2.  **I/O Sbagliato**: La sintassi `>> "Prompt" # (var);` è unica e spesso viene sbagliata (es. dimenticando il `#` o le parentesi).
3.  **Dichiarazioni**: Grammo richiede tipi espliciti (`var int: x;`). I modelli abituati a Python o JS potrebbero omettere il tipo.

### Performance dei Modelli (Pass@k)
Dai benchmark effettuati:
- **Algoritmi Standard (es. Fibonacci)**: I modelli performano bene (Pass@20: 100%) perché la logica algoritmica è trasferibile.
- **Logica Custom (es. Calcolatrice)**: I modelli faticano maggiormente (Pass@5: 0% in alcuni casi) a combinare logica complessa con la sintassi I/O rigorosa di Grammo senza feedback di compilazione. L'uso di modelli avanzati (Gemini 3 Pro) o il ciclo di auto-correzione è essenziale.

1.  **Punto e virgola**: Ogni istruzione deve terminare con un punto e virgola `;`.
2.  **Parentesi graffe**: I blocchi `{ ... }` sono obbligatori per le strutture di controllo.
3.  **Pattern I/O**: La sintassi di I/O è rigorosa. Devi usare il separatore `#` tra la stringa letterale e la lista di variabili tra parentesi.
4.  **Ricorsione**: Supportata (es. per fattoriale o fibonacci).

## 🔍 Esempio: Fattoriale

```grammo
func int -> fattoriale(int: n) {
    if (n <= 1) {
        return 1;
    }
    return n * fattoriale(n - 1);
}

func void -> main() {
    var int: n, r;
    >> "Inserisci numero: " # (n);
    r = fattoriale(n);
    <<! "Il fattoriale è: " # (r);
}
```

## 🧠 Specifiche Semantiche Avanzate

Questa sezione approfondisce le regole semantiche implementate nel compilatore, basate sulle specifiche originali di Grammo.

### 1. Sistema di Tipi e Coercizione
Grammo è fortemente tipizzato ma supporta alcune promozioni implicite per facilitare la scrittura del codice:
*   **Promozione `int` -> `real`**: Un valore intero può essere usato dove è atteso un reale (es. `3 + 4.5` diventa `3.0 + 4.5`). Il contrario non è permesso senza cast esplicito (non ancora supportato).
*   **Operazioni Binarie**:
    *   `+`, `-`, `*`, `/`: Supportano `int` e `real`.
    *   `+` su stringhe: Esegue la concatenazione.
    *   `&&`, `||`, `!`: Richiedono rigorosamente operandi `bool`.

### 2. Regole di Scoping e Shadowing
*   **Scoping Statico (Lexical)**: Le variabili sono visibili nel blocco in cui sono dichiarate e nei blocchi annidati.
*   **Shadowing Proibito**: A differenza di C o Python, **non è permesso** dichiarare una variabile locale con lo stesso nome di una variabile globale o di un parametro della funzione. Questo riduce ambiguità e bug.
    *   *Errore*: `SemanticError: Variable 'x' already declared in outer scope`.

### 3. Analisi del Flusso di Controllo (Return Analysis)
Per le funzioni non-void (es. `func int -> ...`), il compilatore esegue un'analisi statica per garantire che **tutti i percorsi di esecuzione** restituiscano un valore.
*   Un blocco `if` garantisce il ritorno solo se:
    1.  Il ramo `then` ha un `return`.
    2.  Esiste un ramo `else` e anch'esso ha un `return`.
    3.  Tutti gli eventuali rami `elif` hanno un `return`.
*   I cicli `while` e `for` **non** sono considerati percorsi di ritorno garantiti (poiché il ciclo potrebbe non essere mai eseguito).

### 4. Riferimenti Ufficiali
Per i dettagli completi sull'implementazione teorica del linguaggio, consultare la documentazione originale nel repository di Salvatore Di Martino:
*   [Analisi Lessicale e Sintattica](https://github.com/saldm04/Grammo/blob/main/Documents/Analisi_lessicale_sintattica.pdf)
*   [Analisi Semantica](https://github.com/saldm04/Grammo/blob/main/Documents/Analisi_semantica.pdf)
*   [Generazione Codice LLVM](https://github.com/saldm04/Grammo/blob/main/Documents/Generazione_del_codice_LLVM.pdf)

