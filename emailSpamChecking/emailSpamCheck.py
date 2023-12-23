import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer

# Function to preprocess the emails and clean up the unneccesaary data
def preprocess_email(raw_email):
    pattern = r"X-FileName:.*\n\n(.*)"  # Adjust the pattern if needed
    match = re.search(pattern, raw_email, re.DOTALL)
    email_body = match.group(1).strip() if match else raw_email.strip()
    return email_body

# Function to evaluate the model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Load and preprocess the new dataset from emails_v2.csv (emails_v2_smaller.csv is a smaller version of emails_v2.csv for testing. The whole file is emails_v2.csv)
emails_v2_smaller = pd.read_csv("emailSpamChecking/emails_v2_smaller.csv")
emails_v2_smaller['cleaned_message'] = emails_v2_smaller['message'].apply(preprocess_email)

# Load and prepare the training data from emails_v1.csv
try: 
    data = pd.read_csv("emailSpamChecking/emails_v1.csv")
    data.drop(["Email No."], axis=1, inplace=True)
except Exception as e:
    print(f"Error reading the CSV file: {e}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train the Logistic Regression classifier
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate the models
print("Naive Bayes Model Evaluation")
evaluate_model(nb_model, X_test, y_test)
print("\nLogistic Regression Model Evaluation")
evaluate_model(lr_model, X_test, y_test)

# Extract feature names from emails_v1.csv
feature_names = data.columns[:-1]  # All columns except the last one (label)

# Initialize the CountVectorizer with the same features
vectorizer = CountVectorizer(vocabulary=feature_names)

# Fit and transform the cleaned messages of emails_v2_.csv (again, this is emails_v2_smaller.csv, the real file would have the emails_v2.csv as the argument)
emails_v2_features = vectorizer.transform(emails_v2_smaller['cleaned_message'])

# Use the trained models to predict
nb_predictions = nb_model.predict(emails_v2_features)
lr_predictions = lr_model.predict(emails_v2_features)

# Print out a sample of the predictions
print("Sample NB Predictions:", nb_predictions[:10])
print("Sample LR Predictions:", lr_predictions[:10])

#ignore this
# y = data.iloc[:, -1] # last column
##ignoreee