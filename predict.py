import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_next_word(seed_text, max_len=5):
    model = load_model('outputs/model.h5')
    with open("outputs/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""
