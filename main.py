import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained model and tokenizer
model = load_model('sentiment_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess the input text
def preprocess_text(text):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences to ensure uniform input length
    padded_sequences = pad_sequences(sequences, maxlen=100)
    return padded_sequences

# Streamlit app
def main():
    st.title("Sentiment Analysis")
    st.write("Enter a movie review to predict its sentiment (positive or negative).")

    # Input text area for the user to enter a movie review
    user_input = st.text_area("Movie Review", "")

    if st.button("Predict Sentiment"):
        if user_input:
            # Preprocess the input text
            processed_input = preprocess_text(user_input)
            # Make a prediction using the loaded model
            prediction = model.predict(processed_input)
            prediction_class = np.argmax(prediction, axis=1)[0]

            sentiment_map = {0: "Negative", 1: "Netural", 2: "Positive"} 

            st.write(f"Predicted Sentiment: {sentiment_map[prediction_class]}")
        else:
            st.write("Please enter a movie review to predict its sentiment.")

if __name__ == "__main__":
    main()