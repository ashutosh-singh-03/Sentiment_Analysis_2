import os
import pickle
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sklearn

nltk.download("stopwords")

# Initialize Flask app
app = Flask(__name__)

# Load all models
models = {
    'lstm': load_model("models/lstm_three_class_model.h5"),
    'rnn': load_model("models/rnn_three_class_model.h5"),
    'gru': load_model("models/gru_three_class_model.h5")
}

# Default model
current_model = 'lstm'

# Load tokenizer and label encoder
with open("models/tokenizer_three_class.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("models/label_encoder_three_class.pkl", "rb") as f:
    label_encoder = pickle.load(f)
    
# Text Preprocessing Function
def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_sentiment(text, model_name='lstm'):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = models[model_name].predict(padded_sequence)[0]
    predicted_class = np.argmax(prediction)
    sentiment = label_encoder.inverse_transform([predicted_class])[0]
    confidence_percentage = float(np.max(prediction)) * 100
    return sentiment, f"{confidence_percentage:.1f}%"

@app.route("/", methods=['GET','POST'])
def home():
    result = None
    confidence = None
    selected_model = 'lstm'  # Default model
    
    if request.method == 'POST':
        text = request.form.get("text")
        selected_model = request.form.get("model", "lstm")
        
        if text:
            result, confidence = predict_sentiment(text, selected_model)
    
    # Pass model descriptions to the template
    model_descriptions = {
        'lstm': "LSTM (Long Short-Term Memory) - Best for capturing long-range dependencies",
        'rnn': "RNN (Recurrent Neural Network) - Simpler model, faster predictions",
        'gru': "GRU (Gated Recurrent Unit) - Balance between LSTM and RNN"
    }
    
    return render_template("index.html", 
                          result=result, 
                          confidence=confidence, 
                          selected_model=selected_model,
                          model_descriptions=model_descriptions)

if __name__ == "__main__":
    app.run(debug=True)
