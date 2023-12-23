import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

try:
    # Reading the CSV file, ignore 3rd column onwards
    data = pd.read_csv("/Users/sabeernarula/Desktop/CS505 Final Project/CS505Project-1/smsSpamChecking/spam.csv", usecols=[0, 1], header=0, names=['label', 'message'])
except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")

# Preprocess Data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(words)

data['message'] = data['message'].apply(preprocess_text)

# Bag of Words extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Models

# Naive Bayes method
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Logistic Regression method
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Evaluate Models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

print("Naive Bayes Model Evaluation")
evaluate_model(nb_model, X_test, y_test)

print("\nLogistic Regression Model Evaluation")
evaluate_model(lr_model, X_test, y_test)

##################################
# testing on validation dataset

# split data 80:20
split_idx = int(len(data) * 0.8)

training_data = data[:split_idx]
validation_data = data[split_idx:]

# save validation dataset
validation_data.to_csv('/Users/sabeernarula/Desktop/CS505 Final Project/CS505Project-1/smsSpamChecking/validation.csv', index=False)

validation_data = pd.read_csv('/Users/sabeernarula/Desktop/CS505 Final Project/CS505Project-1/smsSpamChecking/validation.csv')

# Drop NA rows
validation_data = validation_data.dropna(subset=['message'])

validation_data['message'] = validation_data['message'].apply(preprocess_text)

# Transform the messages using CountVectorizer
X_validation = vectorizer.transform(validation_data['message'])
y_validation = validation_data['label'].apply(lambda x: 1 if x == 'spam' else 0)

def predict_and_evaluate(model, X, y_true):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(y_true, y_pred)}")

# test NB
print("Naive Bayes on Validation Data")
predict_and_evaluate(nb_model, X_validation, y_validation)

# test LR
print("\nLogistic Regression on Validation Data")
predict_and_evaluate(lr_model, X_validation, y_validation)

########################


# try sample dataset (all normal texts, no spam) - detecting false positives
new_data = pd.read_csv('/Users/sabeernarula/Desktop/CS505 Final Project/CS505Project-1/smsSpamChecking/output.csv', header=None, names=['message'])

# preprocess
new_data['message'] = new_data['message'].apply(preprocess_text)

X_new = vectorizer.transform(new_data['message'])

# Naive Bayes
nb_predictions = nb_model.predict(X_new)

# Logistic Regression
lr_predictions = lr_model.predict(X_new)

def calculate_percentages(predictions):
    counts = np.bincount(predictions)
    # Calculate percentages
    ham_percentage = (counts[0] / len(predictions)) * 100
    spam_percentage = (counts[1] / len(predictions)) * 100
    return ham_percentage, spam_percentage

# Calculate percentages for NB
nb_ham_percentage, nb_spam_percentage = calculate_percentages(nb_predictions)
print(f"Naive Bayes - Ham: {nb_ham_percentage}%, Spam: {nb_spam_percentage}%")

# Calculate percentages for LR
lr_ham_percentage, lr_spam_percentage = calculate_percentages(lr_predictions)
print(f"Logistic Regression - Ham: {lr_ham_percentage}%, Spam: {lr_spam_percentage}%")



###############################

import pickle

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)