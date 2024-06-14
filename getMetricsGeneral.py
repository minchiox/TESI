import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carica il dataset JSON
def load_json_dataset(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Errore: Il file JSON {json_path} non è stato trovato.")
        return None
    except json.JSONDecodeError as e:
        print(f"Errore: Il file JSON non è formattato correttamente. Dettagli: {e}")
        return None

# Carica il dataset CSV
def load_csv_dataset(csv_path):
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Errore: Il file CSV {csv_path} non è stato trovato.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Errore: Il file CSV {csv_path} è vuoto.")
        return None
    except pd.errors.ParserError as e:
        print(f"Errore: Problema durante il parsing del file CSV {csv_path}. Dettagli: {e}")
        return None

# Pulire i dati per assicurarsi che tutte le etichette siano stringhe valide
def clean_labels(labels):
    valid_labels = {'FAST', 'SLOW'}
    return labels.apply(lambda x: x if x in valid_labels else None).dropna()

# Calcola le metriche di valutazione
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_fast = precision_score(true_labels, predicted_labels, pos_label='FAST', zero_division=0)
    recall_fast = recall_score(true_labels, predicted_labels, pos_label='FAST', zero_division=0)
    f1_fast = f1_score(true_labels, predicted_labels, pos_label='FAST', zero_division=0)
    precision_slow = precision_score(true_labels, predicted_labels, pos_label='SLOW', zero_division=0)
    recall_slow = recall_score(true_labels, predicted_labels, pos_label='SLOW', zero_division=0)
    f1_slow = f1_score(true_labels, predicted_labels, pos_label='SLOW', zero_division=0)
    
    return accuracy, (precision_fast, recall_fast, f1_fast), (precision_slow, recall_slow, f1_slow)


# Funzione principale
def main():
    # Lista dei nomi dei dataset
    datasets = ['Eclipse', 'Mozilla','KDE','LiveCode','Novell','OpenXchange','W3C']

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # Percorsi dei dataset
        json_path = f'outputBALANCED/{dataset}/2_resultset.json'
        csv_paths = [
            f'outputBALANCED/{dataset}/3_responses_llama2:70b.csv',
            f'outputBALANCED/{dataset}/3_responses_llama3.csv',
            f'outputBALANCED/{dataset}/3_responses_mistral.csv'
        ]

        # Carica il dataset JSON
        json_df = load_json_dataset(json_path)
        if json_df is None:
            continue

        # Controlla se la colonna 'class' esiste
        if 'class' not in json_df.columns:
            print("Errore: La colonna 'class' non è presente nel dataset JSON.")
            print("Colonne disponibili:", json_df.columns)
            continue

        true_labels = clean_labels(json_df['class'])

        # Calcola e stampa le metriche per ogni dataset CSV
        for csv_path in csv_paths:
            csv_df = load_csv_dataset(csv_path)
            if csv_df is None:
                continue

            # Controlla se la colonna 'label' esiste
            if 'label' not in csv_df.columns:
                print(f"Errore: La colonna 'label' non è presente nel dataset CSV {csv_path}.")
                print("Colonne disponibili:", csv_df.columns)
                continue

            predicted_labels = clean_labels(csv_df['label'])

            # Controlla che le etichette pulite abbiano la stessa lunghezza
            if len(true_labels) != len(predicted_labels):
                print(f"Errore: Lunghezza diversa tra true_labels e predicted_labels per il dataset {csv_path}.")
                print(f"Lunghezza true_labels: {len(true_labels)}, Lunghezza predicted_labels: {len(predicted_labels)}")
                continue

            accuracy, metrics_fast, metrics_slow = calculate_metrics(true_labels, predicted_labels)

            print(f"Metrics for {csv_path}:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Metrics for 'FAST':")
            print(f"  Precision: {metrics_fast[0]:.4f}")
            print(f"  Recall: {metrics_fast[1]:.4f}")
            print(f"  F1 Score: {metrics_fast[2]:.4f}")
            print("Metrics for 'SLOW':")
            print(f"  Precision: {metrics_slow[0]:.4f}")
            print(f"  Recall: {metrics_slow[1]:.4f}")
            print(f"  F1 Score: {metrics_slow[2]:.4f}")
            print()

if __name__ == "__main__":
    main()
