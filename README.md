# HARTH_project
Esame IA2 02/02/26 su HARTH dataset
# Human Activity Recognition (HAR) via LightGBM
Questo progetto implementa una pipeline di **classificazione multiclasse supervisionata** per il riconoscimento automatico delle attività fisiche (Human Activity Recognition) basata su dati accelerometrici reali.

## Descrizione del Progetto
Il task consiste nel classificare il tipo di attività svolta da un soggetto partendo dai segnali catturati da sensori di accelerazione indossabili (posizionati su schiena e coscia). I dati provengono dal dataset HARTH, raccolto in un contesto free-living, il che comporta una sfida maggiore rispetto ai dati di laboratorio a causa del rumore e della variabilità dei movimenti naturali.

## Pipeline Tecnica

### 1. Preprocessing e Segmentazione
I dati grezzi sono stati ripuliti, normalizzati e trasformati in formato tabellare seguendo questi criteri:
Sliding Windows: Il segnale continuo è stato suddiviso in finestre temporali di 2 secondi (100 campioni) con un overlap del 50%.
Filtro Transizioni: Sono state mantenute solo le "finestre pure", ovvero segmenti in cui l'attività è costante per tutta la durata della finestra, scartando i momenti di transizione.

### 2. Feature Engineering
Per ogni finestra e per ognuno dei 6 assi (3 per la schiena, 3 per la coscia), sono state estratte le seguenti feature:
* Statistiche di base: Media, deviazione standard, minimo e massimo.
* Analisi Energetica: Energia del segnale ($\text{mean}(x^2)$), identificata in fase di EDA come ottimo discriminante per classi ad alta intensità.
* Movimento Complessivo: Calcolo della Signal Magnitude Area (SMA) per catturare l'intensità globale del movimento su schiena e coscia.

### 3. Modellazione e Valutazione
Il dataset tabellare risultante è stato fornito a un modello LightGBM (Gradient Boosting Decision Tree). La valutazione è stata effettuata tramite:
* GroupShuffleSplit: Per garantire che i dati dello stesso soggetto non appaiano contemporaneamente nel training e nel test set (evitando data leakage).
* Metriche: F1-Score (per gestire lo sbilanciamento delle classi), Confusion Matrix e analisi della Feature Importance.

---

## Librerie Utilizzate
Il progetto è sviluppato in Python utilizzando il seguente stack tecnologico:
* `numpy` (np)
* `pandas` (pd)
* `matplotlib.pyplot` (plt)
* `lightgbm` (lgb)
* `scikit-learn` (sklearn)
    * `sklearn.model_selection.GroupShuffleSplit`
    * `sklearn.metrics.classification_report`, `confusion_matrix`
    * `sklearn.metrics.ConfusionMatrixDisplay`, `precision_recall_fscore_support`

---

## Come eseguire il codice per riprodurre i risultati

Per replicare l'analisi e l'addestramento del modello, segui questi passaggi:

1.  **Configurazione Ambiente**:
   Python  deve essereninstallato. Installare le dipendenze necessarie tramite pip:
    ```bash
    pip install numpy pandas matplotlib scikit-learn lightgbm
    ```

2.  **Preparazione Dati**:
    * Scaricare il dataset HARTH e inserire i file `.csv` nella cartella `data/`.
    * Eseguire lo script di preprocessing per generare le finestre temporali e calcolare le feature (SMA, Energia, ecc.).

3.  **Training del Modello**:
    * Lanciare il notebook o lo script principale di training.
    * Il sistema utilizzerà il `GroupShuffleSplit` per dividere i soggetti e addestrerà il classificatore `LGBMClassifier`.

4.  **Generazione Risultati**:
    * Al termine dell'esecuzione, verranno visualizzate a schermo la Confusion Matrix e il Classification Report con i valori di F1-score per ogni attività.
    * Verrà generato un grafico a barre relativo alla Feature Importance per mostrare quali sensori hanno influenzato maggiormente la classificazione.
