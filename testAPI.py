import requests

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
            return {"error": f"Errore nella richiesta: {response.text}"}
    except Exception as e:
        # Gestisci eventuali eccezioni durante la richiesta
        return {"error": f"Errore durante la richiesta: {str(e)}"}

#Invio di un prompt al modello "llama3"
model = "llama2:70b"
prompt = "Why is the sky blue?"
response = generate_response(model, prompt)
print("Risposta dal modello llama3:", response)
