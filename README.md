# Spam_prediction_on_sparse-text
Spam prediction on sparse text


Predicting spam in emails is a common task in natural language processing. To perform spam prediction using the dataset named emails.csv, you can follow these steps using Python and popular libraries such as pandas, scikit-learn, and nltk:

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


Assuming your data is stored in a CSV file named emails.csv

data = pd.read_csv('emails.csv')
print(data.head())
print(data.info())


Preprocessing the Text Data:


# Download stopwords if not already downloaded
nltk.download('stopwords')

# Preprocess text data
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word.isalnum()]
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

data['Processed_Text'] = data['Text'].apply(preprocess_text)


Vectorize the Text Data:

# Convert text data to a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Processed_Text'])

# Convert labels to numeric values (0 for ham, 1 for spam)
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})
y = data['Label']


Split the Data into Train and Test Sets:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Train a Naive Bayes Classifier:

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

Make Predictions and Evaluate the Model:

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(report)

