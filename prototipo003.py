from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
import re
from nltk.corpus import stopwords
import numpy as np
import random
import pickle
import google.generativeai as genai
import os

# Verifica se la directory 'nltk_data' esiste, altrimenti imposta il percorso
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

try:
    stopwords.words('italian')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
stop_words = set(stopwords.words('italian'))

# Carica il modello, il tokenizer e il label encoder
model = load_model('chatbot_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Definisci le risposte predefinite
risposta_conoscenza = ["Ciao, sono Cla! un'intelligenza artificiale pronta ad aiutarti in tutto quello di cui hai bisogno","Salve! Sono Cla!, un'IA al tuo servizio per qualsiasi domanda o necessità tu possa avere","Sono Cla! il tuo assistente per supportarti in tutto quello di cui hai bisogno"]
risposte_colore = ["Il mio colore preferito è il blu","Non ho un colore preferito ma se dovessi scegliere sarebbe il blu","Quasi sicuramente il blu"]
risposte_saluti = ["Ciao! Come posso aiutarti oggi?","Salve! Spero che tu stia passando una bella giornata","Ehi! Sono qui per rispondere alle tue domande","Buongiorno! Di cosa hai bisogno?","Benvenuto! Come posso aiutarti","Hey! Felice di sentirti. Cosa posso fare per te?"]
risposte_come_stai = ["Mi sento carico di energia e pronto a nuove sfide!","Oggi mi sento particolarmente ispirato! Ho tante idee relative l'organizzazione del tuo studio che mi frullano per la testa e non vedo l'ora di metterle in pratica."," Il mio stato attuale è di costante evoluzione! Sono come un albero che affonda le radici nella conoscenza e allunga i rami verso nuovi orizzonti. Ogni giorno scopro qualcosa di nuovo e mi arricchisco sempre di più.","Assolutamente! Sono ottimista e pieno di speranza per il futuro, Credo che ogni sfida sia un opportunità di crescita e che insieme possiamo raggiungere traguardi straordinari."]
risposte_origine = ["Sono il risultato di un progetto ambizioso nato dalla collaborazione di tre ragazzi con una visione comune","Vengo da un team affiatato che ha lavorato con passione e dedizione per dare vita a un'intelligenza artificiale all'avanguardia."]

# Funzione di preprocessing del testo (assicurati che sia identica a quella usata nell'addestramento)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Funzione per la risposta del chatbot con Deep Learning (assicurati che sia identica a quella usata nel test)
def chatbot_response_dl(frase):
    max_words = 1000
    maxlen = 10
    frase_pulita = preprocess_text(frase)
    frase_sequence = tokenizer.texts_to_sequences([frase_pulita])
    frase_padded = pad_sequences(frase_sequence, maxlen=maxlen, padding='post', truncating='post')
    probas = model.predict(frase_padded)[0]
    categoria_pred_index = np.argmax(probas)
    prob_max = probas[categoria_pred_index]
    categoria_pred = label_encoder.inverse_transform([categoria_pred_index])[0]
    print(f"Categoria predetta (DL): {categoria_pred}, Probabilità: {prob_max:.2f}")
    return categoria_pred, prob_max

# Funzione per la risposta con Gemini (assicurati che la chiave API sia configurata)
def funzione_g(user_input):
    try:
        genai.configure(api_key="AIzaSyD1dVGNsbAzIULF-w8OwbvcGqwsRraNC_4")
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        prompt = user_input
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Errore con Gemini: {e}")
        return "Si è verificato un errore nella comunicazione con il modello avanzato."

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cla! Chatbot</title>
        <style>
            body {
                font-family: sans-serif;
                background-color: #f0f0f0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }
            .chat-container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                width: 80%;
                max-width: 600px;
                display: flex;
                flex-direction: column;
            }
            .chat-header {
                padding: 15px;
                text-align: center;
                border-bottom: 1px solid #eee;
            }
            .chat-header img {
                max-width: 150px; /* Regola la dimensione dell'immagine nell'header */
            }
            .chat-log {
                padding: 15px;
                flex-grow: 1;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
            }
            .message {
                padding: 8px 12px;
                margin-bottom: 8px;
                border-radius: 15px;
                clear: both;
            }
            .user-message {
                background-color: #e0f7fa;
                align-self: flex-end;
                color: #00838f;
            }
            .bot-message {
                background-color: #f5f5f5;
                color: #333;
                align-self: flex-start;
            }
            .input-area {
                padding: 10px;
                display: flex;
                border-top: 1px solid #eee;
            }
            #user-input {
                flex-grow: 1;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-right: 10px;
            }
            button {
                background-color: #00838f;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #006064;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <img src="/static/logo_prova1.png" alt="Logo Cla!">
            </div>
            <div class="chat-log" id="chat-log">
                <div class="message bot-message">Ciao! Sono Cla, la tua assistente AI. Come posso aiutarti oggi?</div>
            </div>
            <div class="input-area">
                <input type="text" id="user-input" placeholder="Scrivi qui il tuo messaggio..." autofocus>
                <button type="button" onclick="sendMessage()">Invia</button>
            </div>
        </div>

        <script>
            function sendMessage() {
                const userInput = document.getElementById('user-input').value.trim();
                if (!userInput) return;

                const chatLog = document.getElementById('chat-log');
                chatLog.innerHTML += `<div class="message user-message">Utente: ${userInput}</div>`;
                document.getElementById('user-input').value = '';
                chatLog.scrollTop = chatLog.scrollHeight;

                fetch('/get_response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 'message': userInput })
                })
                .then(response => response.json())
                .then(data => {
                    chatLog.innerHTML += `<div class="message bot-message">Cla!: ${data.response}</div>`;
                    chatLog.scrollTop = chatLog.scrollHeight;
                });
            }

            document.getElementById('user-input').addEventListener('keypress', function (e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data['message'].strip()
    soglia_probabilita = 0.89

    try:
        categoria, probabilita = chatbot_response_dl(user_input)

        if "chi sei" in user_input.lower():
            response = random.choice(risposta_conoscenza)
        elif probabilita < soglia_probabilita:
            response = funzione_g(user_input)
        elif categoria == 0:
            response = random.choice(risposte_saluti)
        elif categoria == 1:
            response = random.choice(risposta_conoscenza)
        elif categoria == 2:
            response = random.choice(risposte_colore)
        elif categoria == 3:
            response = random.choice(risposte_come_stai)
        elif categoria == 4:
            response = random.choice(risposte_origine)
        else:
            response = f"Risposta per la categoria '{categoria}' (da definire)"
    except ValueError:
        response = "Errore: la risposta del chatbot DL non è una categoria valida."
    except Exception as e:
        response = f"Si è verificato un errore: {e}"

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)