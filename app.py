import os
import pathlib
import joblib
import streamlit as st

# ----- Settings -----
REPO_URL = "https://github.com/bivashk/spam-email-classifier"  # <-- your repo URL
MODELS_DIR = pathlib.Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "spam_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.txt"

# ----- Page config -----
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")

st.title("📧 Spam Email Classifier")
st.caption("TF-IDF + (Naive Bayes or Logistic Regression) · scikit-learn · Streamlit")

# ----- Load model -----
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Model not found at **models/spam_model.joblib**.\n\n"
            "Run `python src/train.py` locally and commit the **models/** folder."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

with st.expander("How it works (short)"):
    st.write(
        "- Text is transformed into TF-IDF features.\n"
        "- A classifier (Naive Bayes or Logistic Regression — whichever scored better in training) predicts **Spam** vs **Not Spam**.\n"
        "- You can view training metrics below."
    )

import random

# ----- Input area -----
default_text = st.session_state.get("text", "")
user_text = st.text_area(
    "Paste an SMS or email message:",
    value=default_text,
    height=180,
    placeholder="Type or load a sample from the left sidebar…",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    do_predict = st.button("🔍 Classify")
with col2:
    st.link_button("View source on GitHub", REPO_URL)
with col3:
    if METRICS_PATH.exists():
        st.download_button("Download metrics", data=METRICS_PATH.read_text(), file_name="metrics.txt")
    else:
        st.button("Metrics unavailable", disabled=True)

# ----- Prediction -----
if do_predict:
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        pred_label = model.predict([user_text])[0]

        # Probability of spam (if supported)
        spam_prob = None
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            proba = model.predict_proba([user_text])[0]
            spam_prob = float(proba[classes.index("spam")]) if "spam" in classes else float(max(proba))

        if pred_label == "spam":
            st.error("🚨 Prediction: **Spam**")
        else:
            st.success("✅ Prediction: **Not Spam**")

        if spam_prob is not None:
            st.caption(f"Spam probability: {spam_prob:.2%}")
            st.progress(min(max(spam_prob, 0.0), 1.0))

st.markdown("---")
if METRICS_PATH.exists():
    with st.expander("Model metrics (from training)"):
        st.code(METRICS_PATH.read_text())

st.caption("👨‍💻 Built by Bivash Koirala · NIT Rourkela")
