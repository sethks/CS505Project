from flask import Flask, request, render_template
import pickle
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__)

# Load your trained models and vectorizer here
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        message = request.form['message']
        prediction, confidence = predict_message(message)

    return render_template('index.html', prediction=prediction, confidence=confidence)



def predict_message(message):
    # Preprocess the message
    processed_message = preprocess_text(message)

    # Vectorize the message
    vectorized_message = vectorizer.transform([processed_message])

    # Make a prediction
    prediction_nb = nb_model.predict(vectorized_message)
    prediction_lr = lr_model.predict(vectorized_message)

    # Calculate confidence
    probabilities = nb_model.predict_proba(vectorized_message)
    confidence = max(probabilities[0])

    # Determine final prediction (you can combine or choose one model's prediction)
    final_prediction = 'spam' if prediction_nb[0] == 1 else 'not spam'

    return final_prediction, confidence


if __name__ == '__main__':
    app.run(debug=True)
