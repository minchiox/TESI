import json
import csv
import requests
import os
import string

# Carica i dati dal file JSON
def load_data_from_json(file_path, dataset):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("load data from: {}".format(dataset))
            print(f"Totale dati caricati: {len(data)}")
        return data
    except FileNotFoundError:#prima gestione degli errori
        print(f"Errore: Il file {file_path} non è stato trovato.")
        return None
    except json.JSONDecodeError as e:#prima gestione degli errori
        print(f"Errore: Il file JSON non è formattato correttamente. Dettagli: {e}")
        return None
    except UnicodeDecodeError as e:#seconda gestione degli errori
        print(f"Errore di codifica durante la lettura del file. Dettagli: {e}")
        return None

# Invia un prompt all'LLM e ottieni la risposta
def generate_response(model, prompt, stream=False):
    # Definisci i dati da inviare come payload JSON
    payload = {
        "model": model, #modello intercambiabile
        "prompt": prompt, #prompt intercambiabile
        "stream": stream
    }

    # URL DELL'API NEL CONTAINER DOCKER
    url = "http://localhost:11434/api/generate"

    try:
        # Effettua la richiesta POST con i dati JSON
        response = requests.post(url, json=payload)

        # Controlla lo stato della risposta
        if response.status_code == 200:
            # Se la richiesta ha avuto successo, restituisci il contenuto della risposta
            return response.json()
        else:
            # Altrimenti, restituisci un messaggio di errore
            print("errore nella richiesta")
            return {"error": f"Errore nella richiesta: {response.text}"}#prima gestione degli errori
    except Exception as e:
        print("errore nella richiesta exception")
        # Gestisci eventuali eccezioni durante la richiesta
        return {"error": f"Errore durante la richiesta: {str(e)}"}

# Salva le risposte in un file CSV
# Nel file salviamo id del bug, giorni di risoluzione, la label derivata "FAST"/"SLOW", e la risposta originale dell'LLM
def save_responses_to_csv(responses, output_csv_path):
    print("saving response to csv")
    fieldnames = ['id', 'days_resolution', 'label', 'response']
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for response in responses:
            writer.writerow(response)


# FILTRAGGIO INSERITO PER CONVERTIRE LA RISPOSTA IN LABEL, IN QUANTO
# NON SEMPRE LLM RESTITUISCE SOLO LA LABEL
def extract_response_category(response_text):
    # Rimuovi la punteggiatura dalle parole e gli apici singoli
    # Controllo aggiunto per problemi sulle LABEL - 'A' A, A. - 
    words = [''.join(char for char in word if char not in string.punctuation) for word in response_text.split()]
    # Se 'A' è presente nella risposta, restituisci 'FAST'
    if "A" in words:
        return 'FAST'
    # Se 'B' è presente nella risposta, restituisci 'SLOW'
    elif "B" in words:
        return 'SLOW'


def main():
    # Liste dei dataset e dei modelli
    datasets = ["Eclipse","KDE","W3C", "LiveCode", "Novell", "OpenXchange", "W3C","Mozilla"]
    models = ["mistral", "llama2:70b", "llama3"]

    # Prompt pattern
    # Input Semantics, Output Customization, Persona
    prompt_pattern = (
        "Start thinking as if you were an expert software developer, "
        "A = 'FAST' and B = 'SLOW' you can prompt in output only A or B, "
        "predicting the resolution time of a bug by saying those that can be "
        "resolved in LESS than 50 DAYS in the 'A' label and those that can be resolved "
        "in MORE than 50 DAYS in the 'B' label.\n"
        "BUG: comes from the {dataset} BTS (Bug Tracking System) Dataset, the estimated "
        "resolution time is {days_resolution} days and about this bug we know that: {comments}\n"
        "Output: A or B, don't write other words!"
    )

    # Itera attraverso ogni dataset
    for dataset in datasets:
        # Carica i dati dal file JSON
        #outputFULL per i dataset completi
        #outputBALANCED per i dataset bilanciati
        #output per i dataset sbilanciati
        json_file_path = os.path.join('outputFULL', dataset, '2_resultset.json')
        data = load_data_from_json(json_file_path, dataset)
        if data is None:
            continue

        # Itera attraverso ogni modello
        for model in models:
            # Lista per salvare le risposte
            responses = []
            output_csv_path = os.path.join('outputFULL', dataset, f'3_responses_{model}.csv')

            # Itera attraverso i dati e invia i prompt personalizzati
            for i, item in enumerate(data, start=1):
                days_resolution = item['days_resolution']
                comments = item['comments']
                prompt = prompt_pattern.format(
                    dataset=dataset,
                    days_resolution=days_resolution,
                    comments=comments
                )
                response = generate_response(model, prompt)
                # Controlla che effettivamente ci sia una risposta
                response_text = response['response'] if 'response' in response else response['error']
                # Estrai dalla risposta solo la label 'SLOW' or 'FAST'
                response_category = extract_response_category(response_text)
                # Aggiungiamo i dati nel nuovo dataset
                responses.append({
                    'id': item['bug_id'],
                    'days_resolution': item['days_resolution'],
                    'label': response_category,
                    'response': response_text
                })

                # Stampa il numero di ciclo, l'id del bug e la label, utilizzato in esecuzione per comprendere lo stato delle run
                print(f"Ciclo: {i}, Dataset: {dataset}, Modello: {model}, ID Bug: {item['bug_id']}, Label: {response_category}")
                # Stampa la risposta da attivare in fase di debug
                #print(f"Risposta: {response_text}")

            # Salva le risposte in un file CSV
            save_responses_to_csv(responses, output_csv_path)


if __name__ == "__main__":
    main()
