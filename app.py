import os
import joblib
import streamlit as st


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.joblib")

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")

st.title("📧 Spam Email Classifier")
st.write("Paste an SMS or email message below and I’ll classify it as **Spam** or **Ham**.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run `python src/train.py` locally and commit `models/spam_model.joblib`.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

user_text = st.text_area("Your message:", height=180, placeholder="Congratulations! You've won a prize...")
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("Classify")
with col2:
    st.link_button("View Source on GitHub", "https://github.com/your-username/your-repo", disabled=False)

if predict_btn:
    if not user_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        # model is a Pipeline(TfidfVectorizer -> Classifier) so we can call predict_proba directly
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([user_text])[0]
            classes = list(model.classes_)
            # Get index of 'spam' class probability if present
            if "spam" in classes:
                spam_idx = classes.index("spam")
                spam_prob = float(proba[spam_idx])
            else:
                # Fallback if class order differs
                spam_prob = float(max(proba))
        else:
            # Some classifiers may not have predict_proba; fallback to decision
            pred = model.predict([user_text])[0]
            spam_prob = 1.0 if pred == "spam" else 0.0

        label = "Spam" if spam_prob >= 0.5 else "Ham"
        st.subheader(f"Prediction: {label}")
        st.caption(f"Spam probability: {spam_prob:.2%}")

st.markdown("---")
st.caption("Model: TF-IDF + (Naive Bayes or Logistic Regression, whichever performed better on validation).")
