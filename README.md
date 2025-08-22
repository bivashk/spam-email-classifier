# Spam Email Classifier (TF-IDF + Naive Bayes / Logistic Regression)

A simple, interview-friendly NLP project that classifies SMS/email text as **Spam** or **Ham** using scikit-learn.
Includes:
- Training script that **automatically downloads** the public SMS Spam dataset from UCI.
- Model selection between **Multinomial Naive Bayes** and **Logistic Regression** (chooses the best).
- Streamlit app for interactive demo.
- Clean, resume-ready repo structure.

## 🧱 Project Structure
```
spam-email-classifier/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .gitignore
├── README.md
├── src/
│   ├── preprocessing.py    # text cleaning helpers (optional; TF-IDF already lowercases + strips stopwords)
│   └── train.py            # download data, train models, save best pipeline
├── data/                   # dataset lives here (auto-downloaded by train.py)
└── models/                 # saved model pipeline + metrics (created by train.py)
```

## 🚀 Quickstart (Local)
```bash
# 1) Create and activate virtual env (Mac/Linux)
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
# python -m venv .venv
# .venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Train (downloads dataset automatically)
python src/train.py

# 4) Run the app
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud
1. Push this folder to a **new GitHub repo**.
2. On https://share.streamlit.io/ (Streamlit Community Cloud), connect your GitHub and select the repo.
3. Set **Main file path** to `app.py`. (No secrets needed. Python 3.10/3.11 is fine.)
4. If you didn't commit `models/spam_model.joblib` yet, run `src/train.py` locally and commit it so the app can load without training in the cloud.

## 📊 What gets saved
- `models/spam_model.joblib` → the **entire trained pipeline** (TF-IDF + classifier).
- `models/metrics.txt` → accuracy/precision/recall/F1 on test set.
- `data/SMSSpamCollection` → the raw dataset (tab-separated), downloaded from UCI.

## 🗣️ Interview Talking Points
- Why **TF-IDF** over Bag of Words.
- Why **Naive Bayes** is strong for text; when **LogReg/SVM** win.
- Handling **class imbalance**, **precision/recall** trade-offs.
- Real-world extensions: incremental training, model monitoring, adversarial/spam drift.

---

*Dataset source: SMS Spam Collection Dataset (UCI Machine Learning Repository).*

