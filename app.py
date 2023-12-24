from flask import Flask, request, render_template, jsonify
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
with open('smsSpamChecking/nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('smsSpamChecking/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('smsSpamChecking/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('emailSpamChecking/email_nb_model.pkl', 'rb') as f:
    email_nb_model = pickle.load(f)

with open('emailSpamChecking/email_lr_model.pkl', 'rb') as f:
    email_lr_model = pickle.load(f)

with open('emailSpamChecking/vectorizer.pkl', 'rb') as f:
    email_vectorizer = pickle.load(f)

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/check_sms', methods=['POST'])
def check_sms():
    message = request.form['message']
    prediction, confidence = predict_sms(message)
    return jsonify({'prediction': prediction, 'confidence': confidence})

@app.route('/check_email', methods=['POST'])
def check_email():
    email = request.form['email']
    prediction, confidence = predict_email(email)
    return jsonify({'prediction': prediction, 'confidence': confidence})

def predict_sms(message):
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


    final_prediction = 'spam' if prediction_lr[0] == 1 else 'not spam'

    return final_prediction, confidence

def predict_email(email):
    # Preprocess the email
    processed_email = preprocess_text(email)

    # Vectorize the message
    vectorized_email = email_vectorizer.transform([processed_email])

    # Make a prediction
    prediction = email_nb_model.predict(vectorized_email)

    # Calculate confidence
    confidence = max(email_nb_model.predict_proba(vectorized_email)[0])

    # Determine final prediction (you can combine or choose one model's prediction)
    final_prediction = 'spam' if prediction[0] == 1 else 'not spam'

    return final_prediction, confidence

if __name__ == '__main__':
    app.run(debug=True)
