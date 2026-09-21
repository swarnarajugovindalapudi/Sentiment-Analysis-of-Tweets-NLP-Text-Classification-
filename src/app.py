import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


# 1. Setup the text cleaning tools
stop_words = set(stopwords.words('english'))

def clean_user_input(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_words)

# 2. Load the trained AI artifacts from the models folder
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# 3. Build the User Interface
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.write("Type a simulated airline tweet below to see if the AI thinks it is positive, negative, or neutral!")

# 4. Create an input box and a button
user_tweet = st.text_input("Enter your tweet here:")

if st.button("Predict Sentiment"):
    if user_tweet:
        # Step A: Clean the raw user input
        cleaned_tweet = clean_user_input(user_tweet)
        
        # Step B: Translate the words into mathematical numbers
        # Notice we use [cleaned_tweet] in brackets because it expects a list
        vectorized_tweet = vectorizer.transform([cleaned_tweet]).toarray()
        
        # Step C: Ask the model to make a prediction
        prediction = model.predict(vectorized_tweet)[0]
        
        # Step D: Display the result on the web page
        st.success(f"The AI predicts this sentiment is: **{prediction.upper()}**")
    else:
        st.warning("Please type a tweet first!")