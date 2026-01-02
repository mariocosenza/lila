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

## ⚠️ Vincoli e Best Practices

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
