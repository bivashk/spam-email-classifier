# 📧 Spam Email Classifier  

A simple **Machine Learning project** that classifies SMS/email messages as **Spam** or **Ham (Not Spam)**.  

- 🚀 End-to-end workflow: data preprocessing → feature extraction → model training → evaluation → deployment.  
- 🎯 Achieved **96% accuracy** using **Logistic Regression** with TF-IDF features.  
- 🌐 Interactive **Streamlit app** for real-time predictions.  

---

## 🔍 Why This Project?  
Spam detection is a classic **NLP problem** with real-world impact.  
This project helped me learn:  
- How to clean and preprocess raw text.  
- Why **TF-IDF** works better than plain Bag-of-Words.  
- How to compare classifiers (Naive Bayes vs Logistic Regression).  
- Deploying ML models in a user-friendly way with Streamlit.  

---

## 🧱 Project Structure  

```text
spam-email-classifier/
├── app.py                 # Streamlit web app (UI)
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── data/                  # Dataset (auto-downloaded in training)
├── models/                # Saved trained models + metrics
└── src/
    ├── preprocessing.py   # Text cleaning & TF-IDF vectorization
    └── train.py           # Training pipeline (fit, evaluate, save)

---

## 📊 Model Training & Results  

1. **Dataset:** [SMS Spam Collection (UCI)](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) (~5.5k labeled messages).  
2. **Preprocessing:**  
   - Lowercasing, punctuation removal, stopword removal.  
   - Converted text → TF-IDF vectors.  
3. **Models Tested:**  
   - Multinomial Naive Bayes → 95% accuracy  
   - Logistic Regression → **96% accuracy (best)**  
4. **Metrics:** Stored in `models/metrics.txt` for transparency.  

✅ Final model saved as `spam_model.joblib`.  

---

## 🖥️ How to Run Locally  

 1. Clone repository
git clone https://github.com/bivashk/spam-email-classifier.git
cd spam-email-classifier

 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

 3. Install requirements
pip install -r requirements.txt

 4. Train model
python src/train.py

 5. Run app
streamlit run app.py

## 🌐 Deployment

Deployed on Streamlit Cloud for easy access.

Live Demo --> (https://spam-email-classifier-bivashk.streamlit.app/)

## 🛠️ Tech Stack

Python 3.10+

scikit-learn (ML models & TF-IDF)

pandas, numpy (data handling)

Streamlit (deployment & UI)

joblib (model persistence)

## 📌 Dataset

Source: SMS Spam Collection Dataset (UCI)

## 👨‍💻 Author

Built by Bivash Koirala
