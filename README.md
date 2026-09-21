# ✈️ Airline Tweet Sentiment Analysis

An end-to-end Natural Language Processing (NLP) text classification project that predicts whether a tweet regarding an airline is positive, negative, or neutral. 

## 🧠 Project Architecture
* **Phase 1: EDA & Preprocessing:** Cleaned messy Twitter data using Regex and NLTK (removed URLs, mentions, and stopwords).
* **Phase 2: Feature Engineering:** Translated English text into a mathematical matrix using TF-IDF Vectorization (5,000 features).
* **Phase 3: Machine Learning:** Trained a Logistic Regression classification algorithm.
* **Phase 4: Deployment:** Serialized the model via Joblib and deployed an interactive web interface using Streamlit.

## 📊 Model Performance
* **Overall Accuracy:** 80%
* **Negative Sentiment Recall:** 94% 
* **Evaluation Metrics:** Evaluated using Precision, Recall, F1-Scores, and Confusion Matrices on a heavily imbalanced dataset.

## 🚀 How to Run Locally
1. Clone this repository.
2. Create a virtual environment and run `pip install -r requirements.txt`.
3. Navigate to the root folder and run the Streamlit app:
   ```bash
   streamlit run src/app.py
   