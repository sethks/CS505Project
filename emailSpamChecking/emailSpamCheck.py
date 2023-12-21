import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

try: 
    data = pd.read_csv("emailSpamChecking\emails.csv")
    data.drop(["Email No."], axis=1, inplace=True)
except Exception as e:
    print(f"Error reading the CSV file: {e}")

x = data.iloc[:, :-1] # all columns except the last one
y = data.iloc[:, -1] # last column

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Naive Bayes method
nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)

# Logistic Regression method
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

print("Naive Bayes Model Evaluation")
evaluate_model(nb_model, x_test, y_test)

print("\nLogistic Regression Model Evaluation")
evaluate_model(lr_model, x_test, y_test)