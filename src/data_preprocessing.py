import re
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_sequences(filepath, max_seq_len=5):
    with open(filepath, 'r', encoding='utf-8') as f:
        corpus = f.read()

    cleaned = clean_text(corpus)
    lines = cleaned.split('.')  # basic sentence splitting

    # Tokenization
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(lines)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # +1 for padding or OOV

    sequences = []
    for line in lines:
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(tokens)):
            n_gram = tokens[:i+1]
            sequences.append(n_gram)

    sequences = np.array(pad_sequences(sequences, maxlen=max_seq_len, padding='pre'))

    X = sequences[:, :-1]
    y = sequences[:, -1]

    # One-hot encode y with the actual vocab size
    y = np.eye(vocab_size)[y]

    # Save tokenizer for future use
    with open("outputs/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return X, y, tokenizer
