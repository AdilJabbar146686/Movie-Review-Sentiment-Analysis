import streamlit as st
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Use local nltk_data directory (to avoid download issues on Streamlit Cloud)
nltk.data.path.append('./nltk_data')

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(review):
    cleaned_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([cleaned_review])
    prediction = model.predict(vectorized_review)
    return "Positive 😊" if prediction[0] == 1 else "Negative 😞"

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to find out if it's positive or negative.")

user_input = st.text_area("Your Movie Review")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment = predict_sentiment(user_input)
        st.subheader("Predicted Sentiment:")
        st.success(sentiment)
