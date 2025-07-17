import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model


words_index = imdb.get_word_index()
reversed_word_index = {value:key for key, value in words_index.items()}

model = load_model('simple_rnn_imdb.h5')
def decode_review(encoded_review): # encoded is a array of numbers, basically
    return " ".join([reversed_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [words_index.get(word, 2) + 3 for word in words]
    padding_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padding_review

#prediction
def predict_review(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


#steamlit app
st.title("IMDB movie review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative:")

#user input 
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    sentiment, prediction = predict_review(user_input)
    st.write(f"sentiment:{sentiment}")
    st.write(f"prediction score:{prediction}")
else:
    st.write("Please, enter a movie review")
