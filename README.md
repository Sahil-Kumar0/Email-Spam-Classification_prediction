#Email Spam Classification_prediction
Email Spam Classification_prediction


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
data = pd.read_csv("emails.csv")

# Split the dataset into features (X) and target (y)
X = data.drop(columns=["Email No.", "Prediction"])  # Exclude non-relevant columns
y = data["Prediction"]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report with more metrics
report = classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"])
print("Classification Report:")
print(report)


