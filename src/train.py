import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# Replace 'ham' with 'not-spam'
data["label"] = data["label"].replace({"ham": "not-spam", "spam": "spam"})

# Features and labels
X = data["message"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="spam")
recall = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")
conf_matrix = confusion_matrix(y_test, y_pred)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Save metrics to file
with open("models/metrics.txt", "w") as f:
    f.write("Model: Multinomial Naive Bayes\n")
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Precision: {precision*100:.2f}%\n")
    f.write(f"Recall: {recall*100:.2f}%\n")
    f.write(f"F1-Score: {f1*100:.2f}%\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
