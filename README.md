# Multi-Model Sentiment Analysis App

A Flask web application that analyzes the sentiment of text using multiple neural network models (LSTM, RNN, GRU).

## Features

- Real-time sentiment analysis (Positive/Negative/Neutral)
- Three different neural network models to choose from:
  - **LSTM** (Long Short-Term Memory): Best for capturing long-range dependencies
  - **RNN** (Recurrent Neural Network): Simpler model, faster predictions
  - **GRU** (Gated Recurrent Unit): Balance between LSTM and RNN
- Model selection with informative tooltips
- Confidence percentage for predictions
- Clean, responsive web interface
- Text preprocessing with NLTK

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sentiment_Analysis_2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Open your browser** and go to `http://127.0.0.1:5000`

## Usage

1. Enter any text in the input field
2. Select your preferred model (LSTM, RNN, or GRU)
3. Click "Analyse"
4. View the sentiment result (Positive, Negative, or Neutral) and confidence percentage

## Project Structure

```
├── main.py                         # Flask application
├── requirements.txt                # Python dependencies
├── models/
│   ├── lstm_three_class_model.h5   # Trained LSTM model
│   ├── rnn_three_class_model.h5    # Trained RNN model
│   ├── gru_three_class_model.h5    # Trained GRU model
│   ├── tokenizer_three_class.pkl   # Text tokenizer
│   └── label_encoder_three_class.pkl # Label encoder
└── templates/
    └── index.html                  # Web interface
```

## Technologies Used

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, Bootstrap, Font Awesome
- **NLP**: NLTK, NumPy
- **Models**: LSTM, RNN, GRU Neural Networks

## Model Comparison

| Model | Strengths | Best For |
|-------|-----------|----------|
| LSTM  | Captures long-term dependencies, handles vanishing gradient problem | Complex text with important context across many words |
| RNN   | Simpler architecture, faster predictions, lower memory usage | Shorter texts where recent context is most important |
| GRU   | Fewer parameters than LSTM, often performs similarly to LSTM | Good balance between performance and computational efficiency |

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Flask
- NLTK
- NumPy
- scikit-learn
