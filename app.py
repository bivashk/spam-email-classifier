import streamlit as st
import joblib
import os

# Load model and vectorizer
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")

# Title
st.title("📧 Spam Email Classifier")
st.write("This tool uses **Machine Learning (Naive Bayes + TF-IDF)** to detect whether an email is **Spam** or **Not Spam**.")

# Sidebar
st.sidebar.header("🔍 Try Sample Emails")
sample_spam = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!"
sample_not_spam = "Hi Bivash, can we reschedule our meeting to tomorrow at 4 PM?"

if st.sidebar.button("Use Sample Spam"):
    st.session_state.email_text = sample_spam
if st.sidebar.button("Use Sample Not Spam"):
    st.session_state.email_text = sample_not_spam

# Input Box
email_text = st.text_area("✉️ Paste your email text below:", 
                          st.session_state.get("email_text", ""), height=200)

# Predict Button
if st.button("🚀 Classify Email"):
    if email_text.strip() != "":
        # Vectorize input
        input_data = vectorizer.transform([email_text])
        prediction = model.predict(input_data)[0]

        # Show result
        if prediction == "spam":
            st.error("🚨 This email is classified as **SPAM**.")
        else:
            st.success("✅ This email looks **NOT SPAM**.")
    else:
        st.warning("⚠️ Please enter an email before classifying.")

# Display Model Metrics
st.subheader("📊 Model Performance")
if os.path.exists("models/metrics.txt"):
    with open("models/metrics.txt", "r") as f:
        st.text(f.read())
else:
    st.info("⚠️ Metrics not found. Please run `python src/train.py` to train the model.")
