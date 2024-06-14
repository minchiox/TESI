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
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non è stato trovato.")
        return None
    except json.JSONDecodeError as e:
        print(f"Errore: Il file JSON non è formattato correttamente. Dettagli: {e}")
        return None
    except UnicodeDecodeError as e:
        print(f"Errore di codifica durante la lettura del file. Dettagli: {e}")
        return None

# Invia un prompt a LLaMA2 e ottieni la risposta
def generate_response(model, prompt, stream=False):
    # Definisci i dati da inviare come payload JSON
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

    # Definisci l'URL della tua API
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
            return {"error": f"Errore nella richiesta: {response.text}"}
    except Exception as e:
        print("errore nella richiesta exception")
        # Gestisci eventuali eccezioni durante la richiesta
        return {"error": f"Errore durante la richiesta: {str(e)}"}

# Salva le risposte in un file CSV
def save_responses_to_csv(responses, output_csv_path):
    print("saving response to csv")
    fieldnames = ['id', 'days_resolution', 'label', 'response']
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for response in responses:
            writer.writerow(response)

# Legge il nome del dataset da un file di testo
def read_dataset_name(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    return stripped_line
        print("Errore: Nessun nome di dataset trovato nel file.")
        return None
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non è stato trovato.")
        return None
    except UnicodeDecodeError as e:
        print(f"Errore di codifica durante la lettura del file. Dettagli: {e}")
        return None



def extract_response_category(response_text):
    # Rimuovi la punteggiatura dalle parole e gli apici singoli
    words = [''.join(char for char in word if char not in string.punctuation) for word in response_text.split()]
    # Se 'FAST' è presente nella risposta, restituisci 'FAST'
    if "A" in words:
        return 'FAST'
    # Se 'SLOW' è presente nella risposta, restituisci 'SLOW'
    elif "B" in words:
        return 'SLOW'
    # Se non restituisce A o B settiamo di default a 'SLOW'
    else:
        return 'SLOW'


def main():
    # Percorso del file di testo contenente il nome del dataset
    dataset_name_file = 'datasetList.txt'
    
    # Leggere il nome del dataset dal file di testo
    dataset = read_dataset_name(dataset_name_file)
    if dataset is None:
        return


    model = 'mistral'
    #model = 'llama2:70b'
    #model = 'llama3'

    # Costruire il percorso completo del file JSON
    json_file_path = os.path.join('output', dataset, '2_resultset.json')
    output_csv_path = os.path.join('output', dataset, f'3_responses_{model}.csv')
    
    # Prompt pattern
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

    # Carica i dati dal file JSON
    data = load_data_from_json(json_file_path, dataset)
    if data is None:
        return

    # Lista per salvare le risposte
    responses = []
    
    #label per SLOW or FAST
    response_category = ""

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
        #controlliamo che effettivamente ci sia una risposta
        response_text = response['response'] if 'response' in response else response['error']
        #estraiamo dalla risposta solo la label 'SLOW' or 'FAST'
        response_category = extract_response_category(response_text)
        responses.append({
            'id': item['bug_id'],
            'days_resolution': item['days_resolution'],
            'label': response_category,
            'response': response_text
        })
        
        # Stampa il numero di ciclo, l'id del bug e la label
        print(f"Ciclo: {i}, ID Bug: {item['bug_id']}, Label: {response_category}")
        #Stampa la risposta da attivare in fase di debug
        #print(f"Risposta: {response_text}")
        
    # Salva le risposte in un file CSV
    save_responses_to_csv(responses, output_csv_path)

if __name__ == "__main__":
    main()
