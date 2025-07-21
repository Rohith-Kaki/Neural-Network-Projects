import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load tokenizer and model
model = load_model('next_word_prediction.keras')
with open('tokenizer.pkl', "rb") as f:
    tokenizer = pickle.load(f)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    tokenized_sequence = tokenizer.texts_to_sequences([text])[0]
    if len(tokenized_sequence) >= max_sequence_len:
        tokenized_sequence = tokenized_sequence[-(max_sequence_len-1):]
    tokenized_sequence = pad_sequences([tokenized_sequence], padding='pre', maxlen=max_sequence_len-1)
    predicted = model.predict(tokenized_sequence)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

#steamlit app
st.title('Next Word Prediction with LSTM')
input_text = st.text_input('Enter your text', "to be or not to be")
if st.button("Predict next word?"):
    max_sequence_len = model.input_shape[1]
    next_word = predict_next_word(model=model, tokenizer=tokenizer, text=input_text, max_sequence_len=max_sequence_len)
    st.write(f'Next Word is: {next_word}')